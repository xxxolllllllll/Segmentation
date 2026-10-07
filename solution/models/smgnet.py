from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================= SMG-Net modules =============================
# Ported from SMG-Net/HuiZhenxing/SMG-Net/models/smgnet.py (MALUNet-family
# lightweight UNet). Only change vs the original: the network returns the raw
# crack logit (pre-sigmoid) so the wrapper can expose 2-class logits + feats to
# match the rest of the student interface.


class EAblock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class DilatedGatedAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * self.gate(x)


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list: Sequence[int], mode: str = "fc"):
        super().__init__()
        self.compress = nn.ModuleList([
            nn.Conv2d(c_list[0], c_list[0], 1),
            nn.Conv2d(c_list[1], c_list[1], 1),
            nn.Conv2d(c_list[2], c_list[2], 1),
            nn.Conv2d(c_list[2], c_list[2], 1),
            nn.Conv2d(c_list[3], c_list[3], 1),
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
            for c in (c_list[0], c_list[1], c_list[2], c_list[2], c_list[3])
        ])

    def forward(self, t1, t2, t3, t4, t5):
        out = []
        for i, t in enumerate((t1, t2, t3, t4, t5)):
            f = self.compress[i](t)
            out.append(f * self.gates[i](f))
        return out


class MRFE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.pool_conv = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.slice_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
        self.gated_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.main_conv(x)
        f2 = self.pool_conv(x)
        slices = [x[:, :, i::2, j::2] for i in range(2) for j in range(2)]
        f3 = self.slice_conv(torch.cat(slices, dim=1))
        f3 = F.interpolate(f3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        f4 = self.gated_conv(x)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode="bilinear", align_corners=False)
        return self.fusion(torch.cat([f1, f2, f3, f4], dim=1))


class SACP(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.fusion = nn.Sequential(nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = torch.mean(x, dim=3, keepdim=True).expand(-1, -1, H, W)
        v = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, H, W)
        n = min(H, W)
        d = torch.mean(torch.stack([x[:, :, i, i] for i in range(n)], dim=2), dim=2, keepdim=True)
        d = d.unsqueeze(-1).expand(-1, -1, H, W)
        weight = self.fusion(torch.cat([h, v, d], dim=1))
        return x * weight


class GSSFusion(nn.Module):
    def __init__(self, low_channels: int, high_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, low_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(low_channels // 4, low_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Conv2d(low_channels + high_channels, 1, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(low_channels * 2 + high_channels, low_channels, kernel_size=1),
            nn.GroupNorm(8, low_channels),
        )

    def forward(self, fh: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
        if fh.shape[2:] != fl.shape[2:]:
            fh = F.interpolate(fh, size=fl.shape[2:], mode="bilinear", align_corners=True)
        wc = self.attn(fh)
        fl_weighted = fl * wc
        ws = torch.sigmoid(self.spatial(torch.cat([fl, fh], dim=1)))
        fl_weighted = fl_weighted * ws
        return self.fuse(torch.cat([fl_weighted, fl, fh], dim=1))


class SMGNetBackbone(nn.Module):
    def __init__(self, num_classes: int = 1, input_channels: int = 3,
                 c_list: Sequence[int] = (8, 16, 24, 32, 48, 64), split_att: str = "fc", bridge: bool = True):
        super().__init__()
        self.bridge = bridge
        self.c_list = list(c_list)

        self.encoder1 = MRFE(input_channels, c_list[0])
        self.encoder2 = MRFE(c_list[0], c_list[1])
        self.encoder3 = MRFE(c_list[1], c_list[2])
        self.encoder4 = nn.Sequential(EAblock(c_list[2]), SACP(c_list[2]))
        self.encoder5 = nn.Sequential(nn.Conv2d(c_list[2], c_list[3], kernel_size=1), EAblock(c_list[3]), SACP(c_list[3]))
        self.encoder6 = nn.Sequential(nn.Conv2d(c_list[3], c_list[4], kernel_size=1), EAblock(c_list[4]), SACP(c_list[4]))
        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)

        self.decoder1 = nn.Sequential(GSSFusion(c_list[4], c_list[4]), EAblock(c_list[4]))
        self.decoder2 = nn.Sequential(GSSFusion(c_list[3], c_list[4]), EAblock(c_list[3]))
        self.decoder3 = nn.Sequential(GSSFusion(c_list[2], c_list[3]), EAblock(c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, padding=1))

        self.ebn1 = nn.BatchNorm2d(c_list[0])
        self.ebn2 = nn.BatchNorm2d(c_list[1])
        self.ebn3 = nn.BatchNorm2d(c_list[2])
        self.ebn4 = nn.BatchNorm2d(c_list[2])
        self.ebn5 = nn.BatchNorm2d(c_list[3])
        self.dbn1 = nn.GroupNorm(8, c_list[4])
        self.dbn2 = nn.BatchNorm2d(c_list[3])
        self.dbn3 = nn.BatchNorm2d(c_list[2])
        self.dbn4 = nn.BatchNorm2d(c_list[1])
        self.dbn5 = nn.BatchNorm2d(c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit, _ = self.forward_with_feats(x)
        return logit

    def forward_with_feats(self, x: torch.Tensor):
        out = self.encoder1(x)
        out = F.gelu(F.max_pool2d(self.ebn1(out), 2, 2))
        t1 = out
        out = self.encoder2(out)
        out = F.gelu(F.max_pool2d(self.ebn2(out), 2, 2))
        t2 = out
        out = self.encoder3(out)
        out = F.gelu(F.max_pool2d(self.ebn3(out), 2, 2))
        t3 = out
        out = self.encoder4(out)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2, 2))
        t4 = out
        out = self.encoder5(out)
        out = F.gelu(F.max_pool2d(self.ebn5(out), 2, 2))
        t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        t5_adjusted = F.pad(t5, [0, 0, 0, 0, 0, 16])
        out5 = self.decoder1[0](out, t5_adjusted)
        out5 = self.decoder1[1](out5)
        out5 = F.gelu(self.dbn1(out5))
        out5 = torch.add(out5, t5_adjusted)

        t4_adjusted = F.pad(t4, [0, 0, 0, 0, 0, 8])
        out4 = self.decoder2[0](out5, t4_adjusted)
        out4 = self.decoder2[1](out4)
        out4 = F.gelu(self.dbn2(out4))
        out4 = torch.add(out4, t4_adjusted)
        out4 = F.interpolate(out4, scale_factor=2, mode="bilinear", align_corners=True)

        out3 = self.decoder3[0](out4, t3)
        out3 = self.decoder3[1](out3)
        out3 = F.gelu(self.dbn3(out3))
        out3 = torch.add(out3, t3)
        out3 = F.interpolate(out3, scale_factor=2, mode="bilinear", align_corners=True)

        out2 = F.gelu(self.dbn4(self.decoder4(out3)))
        out2 = F.interpolate(out2, size=t2.shape[2:], mode="bilinear", align_corners=True)
        out2 = torch.add(out2, t2)

        out1 = F.gelu(self.dbn5(self.decoder5(out2)))
        out1 = F.interpolate(out1, size=t1.shape[2:], mode="bilinear", align_corners=True)
        out1 = torch.add(out1, t1)

        out0 = self.final(out1)
        out0 = F.interpolate(out0, scale_factor=2, mode="bilinear", align_corners=True)
        out0 = F.interpolate(out0, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True)
        return out0, (t3, t4, t5)


class SMGNetStudent(nn.Module):
    """SMG-Net adapted to the Stage-C student interface.

    Accepts 0-1 RGB input and returns ``(logits[B, num_classes, H, W], feats)``
    where ``logits`` is 2-class (background/crack) and ``feats`` are the
    post-bridge encoder maps ``(t3, t4, t5)`` (channels ``neck_channels``) for
    optional feature distillation.
    """

    def __init__(self, num_classes: int = 2, c_list: Sequence[int] = (8, 16, 24, 32, 48, 64),
                 split_att: str = "fc", bridge: bool = True, device: torch.device | None = None):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("SMGNetStudent supports num_classes=2 (background/crack) only")
        self.num_classes = 2
        self.c_list = list(c_list)
        self.backbone = SMGNetBackbone(num_classes=1, input_channels=3, c_list=self.c_list,
                                       split_att=split_att, bridge=bridge)
        self.neck_channels = (self.c_list[2], self.c_list[2], self.c_list[3])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        logit_crack, feats = self.backbone.forward_with_feats(x)
        logits = torch.cat([-logit_crack, logit_crack], dim=1)
        return logits, feats
