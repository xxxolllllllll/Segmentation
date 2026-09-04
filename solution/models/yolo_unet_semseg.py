from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTapHead(nn.Module):
    """Replacement for Ultralytics Segment head: return neck P3/P4/P5 features only."""

    def __init__(self) -> None:
        super().__init__()
        self.last_feats: list[torch.Tensor] | None = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise RuntimeError(f"FeatureTapHead expected three neck features, got {type(x)}")
        self.last_feats = [t for t in x]
        return self.last_feats


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YoloUNetDecoder(nn.Module):
    """Lightweight U-Net decoder over YOLO neck features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        decoder_channels: Sequence[int] = (256, 192, 128, 64),
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("in_channels must be (C3, C4, C5)")
        c3, c4, c5 = [int(c) for c in in_channels]
        d5, d4, d3, d2 = [int(c) for c in decoder_channels]

        self.p5_proj = ConvBlock(c5, d5)
        self.p4_proj = nn.Sequential(nn.Conv2d(c4, d4, 1, bias=False), nn.BatchNorm2d(d4), nn.SiLU(inplace=True))
        self.p3_proj = nn.Sequential(nn.Conv2d(c3, d3, 1, bias=False), nn.BatchNorm2d(d3), nn.SiLU(inplace=True))
        self.dec4 = ConvBlock(d5 + d4, d4)
        self.dec3 = ConvBlock(d4 + d3, d3)
        self.dec2 = ConvBlock(d3, d2)
        self.dec1 = ConvBlock(d2, d2)
        self.head = nn.Conv2d(d2, int(num_classes), kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        p3, p4, p5 = feats
        x = self.p5_proj(p5)

        s4 = self.p4_proj(p4)
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec4(torch.cat([x, s4], dim=1))

        s3 = self.p3_proj(p3)
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.dec1(x)
        return self.head(x)


def _copy_ultralytics_graph_meta(src: nn.Module, dst: nn.Module) -> None:
    for name in ("i", "f", "type", "np"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _segment_head_in_channels(head: nn.Module) -> tuple[int, int, int]:
    ch: list[int] = []
    for i in range(int(head.nl)):
        first = head.cv2[i][0]
        conv = getattr(first, "conv", None)
        if conv is None:
            raise TypeError(f"Cannot read Segment.cv2[{i}][0].conv.in_channels")
        ch.append(int(conv.in_channels))
    if len(ch) != 3:
        raise RuntimeError(f"Expected three Segment input channels, got {ch}")
    return ch[0], ch[1], ch[2]


def load_yolo_backbone_neck(weights: str | Path, device: torch.device) -> tuple[nn.Module, tuple[int, int, int]]:
    """Load a YOLO segmentation model and replace its Segment head by FeatureTapHead."""
    from ultralytics.nn.modules.head import Segment
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import DEFAULT_CFG

    model, _ckpt = load_checkpoint(str(weights), device=None, fuse=False)
    head = model.model[-1]
    if not isinstance(head, Segment):
        raise TypeError(f"{weights} must be an Ultralytics segmentation checkpoint with Segment head")

    channels = _segment_head_in_channels(head)
    tap = FeatureTapHead()
    _copy_ultralytics_graph_meta(head, tap)
    model.model[-1] = tap
    model.args = deepcopy(DEFAULT_CFG)
    model.task = "segment"
    model.to(device)
    return model, channels


class YoloUNetSemanticStudent(nn.Module):
    """YOLO11 backbone/neck with a U-Net semantic segmentation head."""

    def __init__(
        self,
        weights: str | Path,
        num_classes: int,
        device: torch.device,
        decoder_channels: Sequence[int] = (256, 192, 128, 64),
    ) -> None:
        super().__init__()
        self.yolo, self.neck_channels = load_yolo_backbone_neck(weights, device=device)
        self.decoder = YoloUNetDecoder(self.neck_channels, num_classes=num_classes, decoder_channels=decoder_channels)
        self.num_classes = int(num_classes)
        self.last_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.yolo(x)
        if isinstance(feats, tuple):
            feats = list(feats)
        if not isinstance(feats, list) or len(feats) != 3:
            tap = self.yolo.model[-1]
            feats = getattr(tap, "last_feats", None)
        if not isinstance(feats, list) or len(feats) != 3:
            raise RuntimeError("YOLO backbone/neck did not return P3/P4/P5 features")
        self.last_feats = (feats[0], feats[1], feats[2])
        return self.last_feats

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        feats = self.forward_features(x)
        logits = self.decoder(feats, out_hw=x.shape[-2:])
        return logits, feats
