from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloSegSemanticStudent(nn.Module):
    """Native YOLO11m-seg (backbone/neck/Segment head) adapted to binary semantic
    crack segmentation via mask-prototype aggregation.

    Built from scratch from the Ultralytics YAML config (no pretrained weights);
    ``nc`` of the detection head is 1 (crack), background is implicit. The mask
    branch outputs per-anchor mask coefficients (``cv4``) and mask protos
    (``proto``); the per-pixel crack logit is the log-sum-exp over anchors of
    ``cls_crack + coeff @ proto``. Returns ``(logits[B,2,H,W], feats)`` so it
    plugs into the Stage-C semantic pipeline unchanged.
    """

    def __init__(
        self,
        cfg: str = "yolo11m-seg.yaml",
        num_classes: int = 2,
        device: torch.device | None = None,
        nc_head: int = 1,
    ) -> None:
        super().__init__()
        from ultralytics.nn.tasks import SegmentationModel

        if int(num_classes) != 2:
            raise ValueError("YoloSegSemanticStudent currently supports binary (num_classes=2) only")
        self.cfg = str(cfg)
        self.num_classes = int(num_classes)
        self.nc_head = int(nc_head)
        self.yolo = SegmentationModel(cfg=self.cfg, ch=3, nc=self.nc_head, verbose=False)
        head = self.yolo.model[-1]
        self.neck_channels = tuple(int(head.cv2[i][0].conv.in_channels) for i in range(3))
        self.last_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _neck_features(self, x: torch.Tensor) -> list:
        """Replicate Ultralytics graph wiring up to (excluding) the Segment head."""
        y: list = []
        for m in self.yolo.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.yolo.save else None)
        return y

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        head = self.yolo.model[-1]
        y = self._neck_features(x)
        head_in = [y[i] for i in head.f]
        self.last_feats = (head_in[0], head_in[1], head_in[2])

        proto = head.proto(head_in[0])  # [B, nm, h, w]
        mc = torch.cat([head.cv4[i](head_in[i]).flatten(2) for i in range(len(head_in))], dim=2)  # [B, nm, A]
        cls = torch.cat([head.cv3[i](head_in[i]).flatten(2) for i in range(len(head_in))], dim=2)  # [B, nc, A]

        mask = torch.einsum("bna,bnp->bap", mc, proto.flatten(2))  # [B, A, P]
        crack = torch.logsumexp(cls[:, 0:1, :].transpose(1, 2) + mask, dim=1)  # [B, P]

        h, w = x.shape[-2:]
        crack = crack.view(x.shape[0], proto.shape[-2], proto.shape[-1])
        crack = F.interpolate(crack[:, None], size=(h, w), mode="bilinear", align_corners=False)
        logits = torch.cat([torch.zeros_like(crack), crack], dim=1)  # [B, 2, H, W]
        return logits, self.last_feats
