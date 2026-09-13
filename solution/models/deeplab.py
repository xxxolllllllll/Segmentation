from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DeepLabSemanticStudent(nn.Module):
    """Torchvision DeepLabV3 for binary semantic segmentation.

    Trained from scratch (no pretrained / COCO weights) on 0-1 inputs; returns
    ``(logits[B,2,H,W], ())``. The classifier is replaced for ``num_classes``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        pretrained: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        from torchvision.models import segmentation as tvs

        self.num_classes = int(num_classes)
        self.backbone = str(backbone)
        weights = "DEFAULT" if pretrained else None

        if self.backbone == "resnet50":
            self.model = tvs.deeplabv3_resnet50(weights=weights, weights_backbone=None, num_classes=self.num_classes)
        elif self.backbone == "resnet101":
            self.model = tvs.deeplabv3_resnet101(weights=weights, weights_backbone=None, num_classes=self.num_classes)
        elif self.backbone in ("mobilenet_v3_large", "mobilenet"):
            self.model = tvs.deeplabv3_mobilenet_v3_large(
                weights=weights, weights_backbone=None, num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported DeepLab backbone: {self.backbone}")
        self.neck_channels = (self.num_classes, self.num_classes, self.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        return self.model(x)["out"], ()
