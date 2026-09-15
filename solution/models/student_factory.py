from __future__ import annotations

from typing import Sequence

import torch

STUDENT_ARCHS = ("yolo_unet", "yolo_seg", "unet", "deeplab")


def build_student(
    arch: str,
    *,
    num_classes: int,
    device: torch.device | None = None,
    student_weights: str = "",
    decoder_channels: Sequence[int] = (256, 192, 128, 64),
    deeplab_backbone: str = "resnet50",
    unet_base: int = 64,
    yolo_seg_cfg: str = "yolo11m-seg.yaml",
    yolo_unet_cfg: str = "yolo11m-seg.yaml",
    yolo_from_scratch: bool = False,
) -> torch.nn.Module:
    """Build a student segmentation model by architecture name.

    All returned models accept 0-1 RGB input, output ``(logits[B,C,H,W], feats)``,
    and expose ``.neck_channels`` (dummy for non-YOLO backbones). ``yolo_unet``
    loads ``student_weights`` unless ``yolo_from_scratch`` is set (then the YOLO
    backbone/neck is built from ``yolo_unet_cfg`` with random init); the comparison
    baselines are initialized from scratch (YOLO-seg from YAML, UNet/DeepLab random).
    """
    arch = str(arch).strip().lower()
    if arch == "yolo_unet":
        from .yolo_unet_semseg import YoloUNetSemanticStudent

        weights = None if yolo_from_scratch else (student_weights or None)
        return YoloUNetSemanticStudent(
            weights,
            num_classes=num_classes,
            device=device,
            decoder_channels=decoder_channels,
            cfg=yolo_unet_cfg,
        )
    if arch == "yolo_seg":
        from .yolo_seg_semantic import YoloSegSemanticStudent

        return YoloSegSemanticStudent(cfg=yolo_seg_cfg, num_classes=num_classes, device=device)
    if arch == "unet":
        from .unet import UNetSemanticStudent

        return UNetSemanticStudent(num_classes=num_classes, base=unet_base, device=device)
    if arch == "deeplab":
        from .deeplab import DeepLabSemanticStudent

        return DeepLabSemanticStudent(num_classes=num_classes, backbone=deeplab_backbone, pretrained=False, device=device)
    raise ValueError(f"Unknown student arch: {arch!r}. Choose from {STUDENT_ARCHS}")
