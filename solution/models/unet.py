from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSemanticStudent(nn.Module):
    """Standard U-Net (Ronneberger et al.) for binary semantic segmentation.

    Trained from scratch (no pretrained weights) on 0-1 inputs; returns
    ``(logits[B,2,H,W], ())``. Input H/W must be divisible by 16 (4 pools).
    """

    def __init__(self, num_classes: int = 2, base: int = 64, device: torch.device | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        b = int(base)
        self.inc = DoubleConv(3, b)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(b, b * 2)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(b * 2, b * 4)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(b * 4, b * 8)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(b * 8, b * 16)
        self.up1 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.dec1 = DoubleConv(b * 16, b * 8)
        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec2 = DoubleConv(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec3 = DoubleConv(b * 4, b * 2)
        self.up4 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec4 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, self.num_classes, 1)
        self.neck_channels = (b * 2, b * 4, b * 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))
        d1 = self.dec1(torch.cat([self.up1(x5), x4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), x2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), x1], dim=1))
        return self.head(d4), ()
