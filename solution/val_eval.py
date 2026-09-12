"""Shared image-level crack IoU validation for early stopping and evaluation.

Used by Stage B (teacher), Stage C (student) and the final eval script so that
the model-selection metric is identical to the reported metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.labelme_crack_copy_paste import rasterize_masks

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CRACK_CLASS = 1
IGNORE_INDEX = 255


def effective_ignore(ignore_manual: np.ndarray, component: np.ndarray) -> np.ndarray:
    return ignore_manual | (~component)


@dataclass
class CrackMetrics:
    tp: int
    fp: int
    fn: int
    valid_pixels: int

    @property
    def iou(self) -> float:
        return float(self.tp) / float(self.tp + self.fp + self.fn + 1e-6)

    @property
    def precision(self) -> float:
        return float(self.tp) / float(self.tp + self.fp + 1e-6)

    @property
    def recall(self) -> float:
        return float(self.tp) / float(self.tp + self.fn + 1e-6)

    @property
    def f1(self) -> float:
        return 2.0 * self.precision * self.recall / (self.precision + self.recall + 1e-6)

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "valid_pixels": float(self.valid_pixels),
        }


def sliding_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def crop_bottom_right_pad(image: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    crop_w = min(size, w - x)
    crop_h = min(size, h - y)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:crop_h, :crop_w] = image[y : y + crop_h, x : x + crop_w]
    return patch, crop_w, crop_h


def crop_center_pad(image: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x2 = min(x + size, w)
    y2 = min(y + size, h)
    crop = image[y:y2, x:x2]
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - crop.shape[0]) // 2
    left = (size - crop.shape[1]) // 2
    patch[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    return patch, (top, left, crop.shape[0], crop.shape[1])


def image_to_tensor_imagenet(image_np: np.ndarray) -> torch.Tensor:
    x = image_np.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))


def image_to_tensor_01(image_np: np.ndarray) -> torch.Tensor:
    x = image_np.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))


def build_gt_masks(
    data: dict[str, Any],
    size: tuple[int, int],
    *,
    crack_labels: set[str],
    ignore_labels: set[str],
    component_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    crack, ignore, component, _other = rasterize_masks(
        data,
        (w, h),
        crack_labels,
        ignore_labels,
        component_labels,
        treat_empty_component_as_full_image=False,
    )
    ignore_eff = effective_ignore(ignore, component)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[crack] = CRACK_CLASS
    gt[ignore_eff] = IGNORE_INDEX
    valid = component & (~ignore_eff)
    return gt, valid


def compute_crack_metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> CrackMetrics:
    if not bool(valid.any()):
        return CrackMetrics(0, 0, 0, 0)
    gt_c = (gt == CRACK_CLASS) & valid
    pred_c = (pred == CRACK_CLASS) & valid
    tp = int(np.logical_and(gt_c, pred_c).sum())
    fp = int(np.logical_and(~gt_c, pred_c).sum())
    fn = int(np.logical_and(gt_c, ~pred_c).sum())
    return CrackMetrics(tp=tp, fp=fp, fn=fn, valid_pixels=int(valid.sum()))


def aggregate_micro(metrics_list: list[CrackMetrics]) -> CrackMetrics:
    return CrackMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        valid_pixels=sum(m.valid_pixels for m in metrics_list),
    )


def aggregate_macro(metrics_list: list[CrackMetrics]) -> dict[str, float]:
    if not metrics_list:
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    vals = [m.as_dict() for m in metrics_list if m.valid_pixels > 0]
    if not vals:
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "iou": float(np.mean([v["iou"] for v in vals])),
        "f1": float(np.mean([v["f1"] for v in vals])),
        "precision": float(np.mean([v["precision"] for v in vals])),
        "recall": float(np.mean([v["recall"] for v in vals])),
    }


@torch.no_grad()
def predict_teacher_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    *,
    imgsz: int,
    stride: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Sliding-window binary crack mask prediction for a DINOv3 Stage-B teacher."""
    h, w = image_rgb.shape[:2]
    acc = np.zeros((num_classes, h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    use_amp = device.type == "cuda"
    for y in sliding_starts(h, imgsz, stride):
        for x in sliding_starts(w, imgsz, stride):
            patch, (top, left, crop_h, crop_w) = crop_center_pad(image_rgb, x, y, imgsz)
            t = image_to_tensor_imagenet(patch).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(t)
            logits_np = logits[0, :, top : top + crop_h, left : left + crop_w].float().cpu().numpy()
            acc[:, y : y + crop_h, x : x + crop_w] += logits_np
            cnt[y : y + crop_h, x : x + crop_w] += 1.0
    acc /= np.maximum(cnt[None, :, :], 1e-6)
    pred = np.argmax(acc, axis=0).astype(np.uint8)
    if num_classes > 2:
        pred = (pred == CRACK_CLASS).astype(np.uint8)
    return pred


@torch.no_grad()
def predict_student_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    *,
    imgsz: int,
    stride: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Sliding-window binary crack mask prediction for the YOLO-U-Net student."""
    h, w = image_rgb.shape[:2]
    prob_sum = np.zeros((num_classes, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y in sliding_starts(h, imgsz, stride):
        for x in sliding_starts(w, imgsz, stride):
            patch, crop_w, crop_h = crop_bottom_right_pad(image_rgb, x, y, imgsz)
            tensor = image_to_tensor_01(patch).unsqueeze(0).to(device)
            logits, _ = model(tensor)
            prob = F.softmax(logits.float(), dim=1)[0, :, :crop_h, :crop_w].cpu().numpy()
            prob_sum[:, y : y + crop_h, x : x + crop_w] += prob
            count[y : y + crop_h, x : x + crop_w] += 1.0
    prob_sum /= np.maximum(count[None, :, :], 1.0)
    return np.argmax(prob_sum, axis=0).astype(np.uint8)
