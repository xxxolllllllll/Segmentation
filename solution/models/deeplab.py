from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DeepLabSemanticStudent(nn.Module):
    """Torchvision DeepLabV3 for binary semantic segmentation.

    Trained from scratch (no pretrained / COCO weights) on 0-1 inputs; returns
    ``(logits[B,2,H,W], (layer2, layer3, layer4))`` resnet features at
    H/8, H/16, H/32 (shallow->deep) for feature distillation. The classifier is
    replaced for ``num_classes``.
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
        if self.backbone not in ("resnet50", "resnet101"):
            raise ValueError("Feature distillation for DeepLab currently supports resnet50/resnet101 backbones")

        # Capture resnet layer2/3/4 outputs (H/8, H/16, H/32) via forward hooks.
        self._feat_keys = ("layer2", "layer3", "layer4")
        self._feats: dict[str, torch.Tensor] = {}
        bb = self.model.backbone
        for name in self._feat_keys:
            bb[name].register_forward_hook(self._make_hook(self._feats, name))
        self.model.eval()
        with torch.no_grad():
            self._feats.clear()
            self.model(torch.zeros(1, 3, 64, 64))
            self.neck_channels = tuple(int(self._feats[k].shape[1]) for k in self._feat_keys)
            self._feats.clear()

    @staticmethod
    def _make_hook(feats: dict, key: str):
        def hook(module, inp, out):
            feats[key] = out

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        self._feats.clear()
        out = self.model(x)["out"]
        feats = tuple(self._feats[k] for k in self._feat_keys)
        return out, feats
