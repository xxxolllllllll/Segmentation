# -*- coding: utf-8 -*-
"""
Stage-B: supervised fine-tuning of domain-adapted DINOv3 teacher.

Model: frozen DINOv3 backbone + adapters + U-Net-like decoder.
Data: either semantic masks from processed patches (images/*.png + masks/*.png),
or LabelMe crack annotations with ignore/component/ncp handling.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import io
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image, UnidentifiedImageError

from checkpoint_io import torch_load_compat
from dataset_seg import SegmentationPatchDataset, default_train_transforms, split_stems
from folds import fold_train_val_test, load_folds, to_tuples
from models.dino_stage_b_unet import DINOv3StageBUNet
from scripts.labelme_crack_copy_paste import (
    alpha_blend_paste,
    extract_instances,
    find_paste_location,
    parse_csv_set,
    rasterize_masks,
    read_labelme,
    transform_instance,
)
from val_eval import (
    aggregate_micro,
    build_gt_masks,
    compute_crack_metrics,
    predict_teacher_mask,
)


EPOCH_METRICS_FIELDS = ["epoch", "train_loss", "val_iou", "best_iou", "patience_counter"]


def prepare_epoch_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(EPOCH_METRICS_FIELDS)


def append_epoch_metrics_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k, "") for k in EPOCH_METRICS_FIELDS])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clamp_num_workers_windows(args: argparse.Namespace) -> None:
    if Path().anchor and "\\" in Path().anchor and args.num_workers != 0:
        print(f"[info] Windows detected, forcing num_workers from {args.num_workers} to 0")
        args.num_workers = 0


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * target_oh, dim=dims)
        union = torch.sum(probs + target_oh, dim=dims)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CrackDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        tgt = (target == 1).float()
        prob = prob[valid]
        tgt = tgt[valid]
        inter = torch.sum(prob * tgt)
        denom = torch.sum(prob) + torch.sum(tgt)
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)


def effective_ignore(ignore_manual: np.ndarray, component: np.ndarray) -> np.ndarray:
    """ignore_effective = ignore_manual ∪ (¬component) (bool masks, same H×W)."""
    return ignore_manual | (~component)


def axis_window_starts(dim: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_toplefts(height: int, width: int, patch_h: int, patch_w: int, stride_y: int, stride_x: int) -> list[tuple[int, int]]:
    ys = axis_window_starts(height, patch_h, stride_y)
    xs = axis_window_starts(width, patch_w, stride_x)
    return [(y0, x0) for y0 in ys for x0 in xs]


def crop_top_left_2d(arr: np.ndarray, y0: int, x0: int, ph: int, pw: int, pad_value: int | bool) -> np.ndarray:
    """Crop or pad-with-value a 2D array to shape (ph, pw) with top-left at (y0, x0) on the source canvas."""
    H, W = arr.shape[:2]
    out = np.full((ph, pw), pad_value, dtype=arr.dtype)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + ph), min(W, x0 + pw)
    if ys >= ye or xs >= xe:
        return out
    dst_y0, dst_x0 = ys - y0, xs - x0
    out[dst_y0 : dst_y0 + (ye - ys), dst_x0 : dst_x0 + (xe - xs)] = arr[ys:ye, xs:xe]
    return out


def crop_top_left_rgb(img: np.ndarray, y0: int, x0: int, ph: int, pw: int) -> np.ndarray:
    H, W, _ = img.shape
    out = np.zeros((ph, pw, 3), dtype=img.dtype)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + ph), min(W, x0 + pw)
    if ys >= ye or xs >= xe:
        return out
    dst_y0, dst_x0 = ys - y0, xs - x0
    out[dst_y0 : dst_y0 + (ye - ys), dst_x0 : dst_x0 + (xe - xs)] = img[ys:ye, xs:xe]
    return out


def labelme_image_size(data: dict[str, Any], img_path: Optional[Path]) -> tuple[int, int]:
    if img_path is not None:
        with Image.open(img_path) as im:
            return im.size
    im = load_labelme_image(data, None)
    return im.size


def build_weighted_sampler_weights(is_positive_mask: list[bool], positive_ratio: float) -> Optional[list[float]]:
    """
    Weights for WeightedRandomSampler so that P(draw positive index) ~= positive_ratio
    when both positive and negative patches exist.
    """
    if not (0.0 < positive_ratio < 1.0) or not is_positive_mask:
        return None
    n_pos = sum(is_positive_mask)
    n_neg = len(is_positive_mask) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = positive_ratio / n_pos
    w_neg = (1.0 - positive_ratio) / n_neg
    return [w_pos if p else w_neg for p in is_positive_mask]


def parse_class_weights(text: str, num_classes: int) -> Optional[torch.Tensor]:
    if not text.strip():
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != num_classes:
        raise ValueError(f"--class-weights requires {num_classes} values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-B supervised fine-tuning (DINOv3+Adapter+UNet)")
    p.add_argument("--data-dir", type=Path, default=None, help="processed directory with images/ and masks/")
    p.add_argument("--labelme-dir", type=Path, default=None, help="LabelMe JSON directory for crack-only training")
    p.add_argument("--images-dir", type=Path, default=None, help="Image directory corresponding to --labelme-dir")
    p.add_argument("--teacher-weights", type=str, default="", help="Local HF DINOv3 directory (with config.json)")
    p.add_argument("--stage-a-ckpt", type=str, default="", help="Optional stage-A checkpoint to initialize adapters")
    p.add_argument(
        "--no-adapters",
        action="store_true",
        help="Raw-backbone teacher: keep adapters at identity (zero-init residual) without loading Stage-A weights.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/stage_b_teacher"))
    p.add_argument("--num-classes", type=int, default=4, help="Including background")
    p.add_argument("--imgsz", type=int, default=1024, help="Resize/pad to square; must be divisible by 16")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs (num_workers>0)")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0 only)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold-split", type=Path, default=None, help="K-fold manifest JSON (folds.json). Overrides --labelme-dir discovery.")
    p.add_argument("--fold", type=int, default=0, help="Held-out fold index (0-based) when --fold-split is set.")
    p.add_argument("--early-stop-patience", type=int, default=10, help="Early stop on val crack IoU patience.")
    p.add_argument("--val-stride", type=int, default=512, help="Sliding-window stride for image-level val IoU.")
    p.add_argument("--resume", type=Path, default=None, help="Resume training from a Stage-B last.pt checkpoint.")
    p.add_argument("--train-adapters", action="store_true", help="Also train adapters in Stage B (default: frozen to preserve Stage A features).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--lambda-dice", type=float, default=1.0)
    p.add_argument("--class-weights", type=str, default="", help="Comma-separated CE class weights")
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--ncp-label", type=str, default="ncp")
    p.add_argument("--enable-copy-paste", action="store_true")
    p.add_argument("--copy-paste-prob", type=float, default=0.7)
    p.add_argument("--min-pastes", type=int, default=1)
    p.add_argument("--max-pastes", type=int, default=3)
    p.add_argument("--cp-search-radius", type=int, default=400)
    p.add_argument("--cp-max-tries", type=int, default=150)
    p.add_argument("--cp-min-crack-area", type=int, default=30)
    p.add_argument("--cp-bbox-padding", type=int, default=16)
    p.add_argument("--cp-max-rotate-deg", type=float, default=10.0)
    p.add_argument("--cp-scale-min", type=float, default=0.9)
    p.add_argument("--cp-scale-max", type=float, default=1.1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="Debug: limit train batches per epoch")
    p.add_argument("--max-val-batches", type=int, default=0, help="Debug: limit validation batches per epoch")
    p.add_argument("--labelme-sliding-window", action="store_true", help="LabelMe: sliding-window patches (5.7.1) instead of resizing full image")
    p.add_argument("--window-size", type=int, default=0, help="Square window side in pixels (0 = use --imgsz)")
    p.add_argument("--window-stride", type=int, default=800, help="Sliding stride for H and W (LabelMe sliding-window mode)")
    p.add_argument(
        "--keep-outside-component-patches",
        action="store_true",
        help="Keep patches with no component pixels (default: discard such patches)",
    )
    p.add_argument(
        "--positive-patch-ratio",
        type=float,
        default=-1.0,
        help="If in (0,1), train loader uses weighted sampling so each draw is crack-positive with ~this probability (needs both pos & neg patches)",
    )
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".JPEG")


def has_label(data: dict[str, Any], label: str) -> bool:
    wanted = label.strip().lower()
    for shape in data.get("shapes", []):
        if isinstance(shape, dict) and str(shape.get("label", "")).strip().lower() == wanted:
            return True
    return False


def resolve_labelme_image(images_dir: Path, ann_path: Path, data: dict[str, Any]) -> Optional[Path]:
    search_dirs = [ann_path.parent, images_dir]
    sub_images = images_dir / "images"
    if sub_images.is_dir():
        search_dirs.append(sub_images)
    deduped_dirs: list[Path] = []
    for root in search_dirs:
        if root not in deduped_dirs:
            deduped_dirs.append(root)
    search_dirs = deduped_dirs
    candidates: list[Path] = []
    image_path = data.get("imagePath")
    for root in search_dirs:
        if isinstance(image_path, str) and image_path.strip():
            candidates.append(root / Path(image_path).name)
        for suffix in IMAGE_SUFFIXES:
            candidates.append(root / f"{ann_path.stem}{suffix}")
        if ".rf." in ann_path.stem:
            rf_suffix = ann_path.stem.split(".rf.", 1)[-1]
            candidates.extend(sorted(root.glob(f"*.rf.{rf_suffix}.*")))
            candidates.extend(sorted(root.glob(f"*{rf_suffix}*")))
    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.is_file() and p.suffix.lower() in {s.lower() for s in IMAGE_SUFFIXES}:
            return p
    return None


def load_labelme_image(data: dict[str, Any], img_path: Optional[Path]) -> Image.Image:
    if img_path is not None:
        return Image.open(img_path).convert("RGB")
    image_data = data.get("imageData")
    if isinstance(image_data, str) and image_data.strip():
        raw = base64.b64decode(image_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise FileNotFoundError("No external image path and no embedded imageData in LabelMe JSON")


def maybe_copy_paste_labelme(
    rng: random.Random,
    enable_copy_paste: bool,
    ncp_label: str,
    data: dict[str, Any],
    image_np: np.ndarray,
    crack: np.ndarray,
    ignore_blocking: np.ndarray,
    component: np.ndarray,
    other: np.ndarray,
    copy_paste_prob: float,
    min_pastes: int,
    max_pastes: int,
    cp_search_radius: int,
    cp_max_tries: int,
    cp_min_crack_area: int,
    cp_bbox_padding: int,
    cp_max_rotate_deg: float,
    cp_scale_min: float,
    cp_scale_max: float,
) -> np.ndarray:
    """ignore_blocking should be ignore_effective (manual ∪ ¬component) for paste blocking."""
    if not enable_copy_paste or has_label(data, ncp_label):
        return crack
    if rng.random() > copy_paste_prob:
        return crack
    instances = extract_instances(image_np, crack, cp_min_crack_area, cp_bbox_padding)
    if not instances:
        return crack
    out_crack = crack.copy()
    n_pastes = rng.randint(min_pastes, max_pastes)
    for _ in range(n_pastes):
        inst = instances[rng.randrange(len(instances))]
        src_patch, src_mask = transform_instance(
            inst["patch"],
            inst["mask"],
            rng,
            cp_max_rotate_deg,
            cp_scale_min,
            cp_scale_max,
            cp_bbox_padding,
        )
        if int(src_mask.sum()) < cp_min_crack_area:
            continue
        loc = find_paste_location(
            image_np,
            src_patch,
            src_mask,
            inst["center"],
            out_crack,
            ignore_blocking,
            component,
            other,
            rng,
            cp_max_tries,
            cp_search_radius,
            0.98,
            0.05,
            0.02,
            25.0,
            20.0,
            30.0,
        )
        if loc is None:
            continue
        x, y = loc
        alpha_blend_paste(image_np, src_patch, src_mask, x, y, 3, 7)
        h, w = src_mask.shape
        out_crack[y : y + h, x : x + w] |= src_mask.astype(bool)
    return out_crack


def tensorize_labelme_pil(
    img: Image.Image,
    mask: Image.Image,
    transform,
    imgsz: int,
    stem: str,
) -> dict:
    if transform is not None:
        img, mask = transform(img, mask)
    if imgsz > 0 and img.size != (imgsz, imgsz):
        img = img.resize((imgsz, imgsz), Image.BILINEAR)
        mask = mask.resize((imgsz, imgsz), Image.NEAREST)
    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    mask_np = np.array(mask, dtype=np.int64)
    return {
        "image": torch.from_numpy(np.transpose(img_np, (2, 0, 1))),
        "mask": torch.from_numpy(mask_np),
        "stem": stem,
    }


def build_sliding_window_entries(
    samples: list[tuple[Path, Optional[Path]]],
    crack_labels: set[str],
    ignore_labels: set[str],
    component_labels: set[str],
    window_h: int,
    window_w: int,
    stride_y: int,
    stride_x: int,
    discard_no_component: bool,
) -> tuple[list[tuple[Path, Optional[Path], int, int]], list[bool]]:
    entries: list[tuple[Path, Optional[Path], int, int]] = []
    flags: list[bool] = []
    for ann_path, img_path in samples:
        data = read_labelme(ann_path)
        w, h = labelme_image_size(data, img_path)
        crack, ignore, comp, oth = rasterize_masks(
            data,
            (w, h),
            crack_labels,
            ignore_labels,
            component_labels,
            treat_empty_component_as_full_image=False,
        )
        ie = effective_ignore(ignore, comp)
        for y0, x0 in sliding_window_toplefts(h, w, window_h, window_w, stride_y, stride_x):
            comp_win = crop_top_left_2d(comp, y0, x0, window_h, window_w, False)
            if discard_no_component and not np.any(comp_win):
                continue
            crack_win = crop_top_left_2d(crack, y0, x0, window_h, window_w, False)
            ie_win = crop_top_left_2d(ie, y0, x0, window_h, window_w, True)
            flags.append(bool(np.any(crack_win & ~ie_win)))
            entries.append((ann_path, img_path, y0, x0))
    return entries, flags


class LabelMeCrackDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Optional[Path]]],
        *,
        transform=None,
        enable_copy_paste: bool = False,
        copy_paste_prob: float = 0.7,
        crack_labels: Optional[set[str]] = None,
        ignore_labels: Optional[set[str]] = None,
        component_labels: Optional[set[str]] = None,
        ncp_label: str = "ncp",
        min_pastes: int = 1,
        max_pastes: int = 3,
        cp_search_radius: int = 400,
        cp_max_tries: int = 150,
        cp_min_crack_area: int = 30,
        cp_bbox_padding: int = 16,
        cp_max_rotate_deg: float = 10.0,
        cp_scale_min: float = 0.9,
        cp_scale_max: float = 1.1,
        imgsz: int = 1024,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.enable_copy_paste = enable_copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.crack_labels = crack_labels or {"crack"}
        self.ignore_labels = ignore_labels or {"ignore"}
        self.component_labels = component_labels or {"component", "wood"}
        self.ncp_label = ncp_label.strip().lower()
        self.min_pastes = min_pastes
        self.max_pastes = max_pastes
        self.cp_search_radius = cp_search_radius
        self.cp_max_tries = cp_max_tries
        self.cp_min_crack_area = cp_min_crack_area
        self.cp_bbox_padding = cp_bbox_padding
        self.cp_max_rotate_deg = cp_max_rotate_deg
        self.cp_scale_min = cp_scale_min
        self.cp_scale_max = cp_scale_max
        self.imgsz = int(imgsz)
        self.rng = random.Random(seed)
        self.is_positive_mask: list[bool] = []
        for ann_path, img_path in self.samples:
            data = read_labelme(ann_path)
            w, h = labelme_image_size(data, img_path)
            crack, ignore, comp, _ = rasterize_masks(
                data,
                (w, h),
                self.crack_labels,
                self.ignore_labels,
                self.component_labels,
                treat_empty_component_as_full_image=False,
            )
            ie = effective_ignore(ignore, comp)
            self.is_positive_mask.append(bool(np.any(crack & ~ie)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ann_path, img_path = self.samples[idx]
        data = read_labelme(ann_path)
        img = load_labelme_image(data, img_path)
        w, h = img.size
        image_np = np.array(img, dtype=np.uint8)
        crack, ignore, component, other = rasterize_masks(
            data,
            (w, h),
            self.crack_labels,
            self.ignore_labels,
            self.component_labels,
            treat_empty_component_as_full_image=False,
        )
        ie = effective_ignore(ignore, component)
        crack = maybe_copy_paste_labelme(
            self.rng,
            self.enable_copy_paste,
            self.ncp_label,
            data,
            image_np,
            crack,
            ie,
            component,
            other,
            self.copy_paste_prob,
            self.min_pastes,
            self.max_pastes,
            self.cp_search_radius,
            self.cp_max_tries,
            self.cp_min_crack_area,
            self.cp_bbox_padding,
            self.cp_max_rotate_deg,
            self.cp_scale_min,
            self.cp_scale_max,
        )
        mask_np = np.zeros((h, w), dtype=np.uint8)
        mask_np[crack] = 1
        mask_np[ie] = 255
        pil_img = Image.fromarray(image_np)
        pil_mask = Image.fromarray(mask_np, mode="L")
        return tensorize_labelme_pil(pil_img, pil_mask, self.transform, self.imgsz, ann_path.stem)


class LabelMeSlidingWindowDataset(Dataset):
    """One sample = one (image, label) patch from 5.7.1-style sliding windows."""

    def __init__(
        self,
        entries: list[tuple[Path, Optional[Path], int, int]],
        is_positive_mask: list[bool],
        *,
        window_h: int,
        window_w: int,
        transform=None,
        enable_copy_paste: bool = False,
        copy_paste_prob: float = 0.7,
        crack_labels: Optional[set[str]] = None,
        ignore_labels: Optional[set[str]] = None,
        component_labels: Optional[set[str]] = None,
        ncp_label: str = "ncp",
        min_pastes: int = 1,
        max_pastes: int = 3,
        cp_search_radius: int = 400,
        cp_max_tries: int = 150,
        cp_min_crack_area: int = 30,
        cp_bbox_padding: int = 16,
        cp_max_rotate_deg: float = 10.0,
        cp_scale_min: float = 0.9,
        cp_scale_max: float = 1.1,
        imgsz: int = 1024,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.is_positive_mask = is_positive_mask
        self.window_h = int(window_h)
        self.window_w = int(window_w)
        self.transform = transform
        self.enable_copy_paste = enable_copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.crack_labels = crack_labels or {"crack"}
        self.ignore_labels = ignore_labels or {"ignore"}
        self.component_labels = component_labels or {"component", "wood"}
        self.ncp_label = ncp_label.strip().lower()
        self.min_pastes = min_pastes
        self.max_pastes = max_pastes
        self.cp_search_radius = cp_search_radius
        self.cp_max_tries = cp_max_tries
        self.cp_min_crack_area = cp_min_crack_area
        self.cp_bbox_padding = cp_bbox_padding
        self.cp_max_rotate_deg = cp_max_rotate_deg
        self.cp_scale_min = cp_scale_min
        self.cp_scale_max = cp_scale_max
        self.imgsz = int(imgsz)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        ann_path, img_path, y0, x0 = self.entries[idx]
        data = read_labelme(ann_path)
        img = load_labelme_image(data, img_path)
        w, h = img.size
        image_np = np.array(img, dtype=np.uint8)
        crack, ignore, component, other = rasterize_masks(
            data,
            (w, h),
            self.crack_labels,
            self.ignore_labels,
            self.component_labels,
            treat_empty_component_as_full_image=False,
        )
        ie = effective_ignore(ignore, component)
        patch_img = crop_top_left_rgb(image_np, y0, x0, self.window_h, self.window_w)
        crack_p = crop_top_left_2d(crack, y0, x0, self.window_h, self.window_w, False)
        comp_p = crop_top_left_2d(component, y0, x0, self.window_h, self.window_w, False)
        ie_p = crop_top_left_2d(ie, y0, x0, self.window_h, self.window_w, True)
        other_p = crop_top_left_2d(other, y0, x0, self.window_h, self.window_w, False)
        crack_p = maybe_copy_paste_labelme(
            self.rng,
            self.enable_copy_paste,
            self.ncp_label,
            data,
            patch_img,
            crack_p,
            ie_p,
            comp_p,
            other_p,
            self.copy_paste_prob,
            self.min_pastes,
            self.max_pastes,
            self.cp_search_radius,
            self.cp_max_tries,
            self.cp_min_crack_area,
            self.cp_bbox_padding,
            self.cp_max_rotate_deg,
            self.cp_scale_min,
            self.cp_scale_max,
        )
        mask_np = np.zeros((self.window_h, self.window_w), dtype=np.uint8)
        mask_np[crack_p] = 1
        mask_np[ie_p] = 255
        stem = f"{ann_path.stem}_y{y0}_x{x0}"
        pil_img = Image.fromarray(patch_img)
        pil_mask = Image.fromarray(mask_np, mode="L")
        return tensorize_labelme_pil(pil_img, pil_mask, self.transform, self.imgsz, stem)


def resize_batch(batch: dict, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["image"]
    y = batch["mask"]
    if x.shape[-2:] != (imgsz, imgsz):
        x = F.interpolate(x, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    if y.shape[-2:] != (imgsz, imgsz):
        y = F.interpolate(y.unsqueeze(1).float(), size=(imgsz, imgsz), mode="nearest").squeeze(1).long()
    return x, y


def is_valid_image_mask_pair(images_dir: Path, masks_dir: Path, stem: str) -> bool:
    ip = images_dir / f"{stem}.png"
    mp = masks_dir / f"{stem}.png"
    if not ip.is_file() or not mp.is_file():
        return False
    try:
        with Image.open(ip) as im:
            im.verify()
        with Image.open(mp) as mm:
            mm.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    clamp_num_workers_windows(args)
    set_seed(args.seed)
    if args.imgsz % 16 != 0:
        raise ValueError("--imgsz must be divisible by 16 for ViT-B/16")
    if args.labelme_sliding_window and args.labelme_dir is None:
        raise ValueError("--labelme-sliding-window requires --labelme-dir")
    if args.labelme_sliding_window and args.window_stride <= 0:
        raise ValueError("--window-stride must be positive when using --labelme-sliding-window")

    if args.labelme_dir is not None:
        if args.images_dir is None:
            raise ValueError("--images-dir is required when using --labelme-dir")
        labelme_dir = args.labelme_dir.expanduser().resolve()
        images_dir = args.images_dir.expanduser().resolve()
        if not labelme_dir.is_dir():
            raise FileNotFoundError(f"LabelMe dir not found: {labelme_dir}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        all_samples: list[tuple[Path, Optional[Path]]] = []
        skipped = 0
        embedded = 0
        if args.fold_split is not None:
            split = load_folds(args.fold_split.expanduser().resolve())
            train_dicts, val_dicts, _test = fold_train_val_test(
                split, int(args.fold), float(args.val_ratio), int(args.seed)
            )
            train_samples: list[tuple[Path, Optional[Path]]] = to_tuples(train_dicts)
            val_samples: list[tuple[Path, Optional[Path]]] = to_tuples(val_dicts)
        else:
            for ann_path in sorted(labelme_dir.glob("*.json")):
                data = read_labelme(ann_path)
                img_path = resolve_labelme_image(images_dir, ann_path, data)
                if img_path is None:
                    if isinstance(data.get("imageData"), str) and data.get("imageData"):
                        embedded += 1
                    else:
                        skipped += 1
                        continue
                all_samples.append((ann_path, img_path))
            if not all_samples:
                raise RuntimeError(f"No LabelMe annotations with matching images under {labelme_dir}")
            rng = random.Random(args.seed)
            rng.shuffle(all_samples)
            n_val = max(1, int(len(all_samples) * args.val_ratio)) if len(all_samples) > 1 else 0
            val_samples = all_samples[:n_val]
            train_samples = all_samples[n_val:] or all_samples
        crack_labels = parse_csv_set(args.crack_labels)
        ignore_labels = parse_csv_set(args.ignore_labels)
        component_labels = parse_csv_set(args.component_labels)
        lm_common = dict(
            transform=default_train_transforms(),
            enable_copy_paste=args.enable_copy_paste,
            copy_paste_prob=args.copy_paste_prob,
            crack_labels=crack_labels,
            ignore_labels=ignore_labels,
            component_labels=component_labels,
            ncp_label=args.ncp_label,
            min_pastes=args.min_pastes,
            max_pastes=args.max_pastes,
            cp_search_radius=args.cp_search_radius,
            cp_max_tries=args.cp_max_tries,
            cp_min_crack_area=args.cp_min_crack_area,
            cp_bbox_padding=args.cp_bbox_padding,
            cp_max_rotate_deg=args.cp_max_rotate_deg,
            cp_scale_min=args.cp_scale_min,
            cp_scale_max=args.cp_scale_max,
            imgsz=args.imgsz,
        )
        if args.labelme_sliding_window:
            wh = int(args.window_size) if int(args.window_size) > 0 else int(args.imgsz)
            if wh <= 0:
                raise ValueError("Sliding window size invalid: set --window-size > 0 or --imgsz > 0")
            sy = sx = int(args.window_stride)
            discard_no_component = not bool(args.keep_outside_component_patches)
            train_entries, train_pos_flags = build_sliding_window_entries(
                train_samples,
                crack_labels,
                ignore_labels,
                component_labels,
                wh,
                wh,
                sy,
                sx,
                discard_no_component,
            )
            val_src = val_samples if val_samples else train_samples[: max(1, len(train_samples) // 5)]
            val_entries, val_pos_flags = build_sliding_window_entries(
                val_src,
                crack_labels,
                ignore_labels,
                component_labels,
                wh,
                wh,
                sy,
                sx,
                discard_no_component,
            )
            if not train_entries:
                raise RuntimeError(
                    "No training sliding-window patches (check component polygons and --keep-outside-component-patches)"
                )
            if not val_entries:
                raise RuntimeError("No validation sliding-window patches")
            train_ds = LabelMeSlidingWindowDataset(
                train_entries,
                train_pos_flags,
                window_h=wh,
                window_w=wh,
                seed=args.seed,
                **lm_common,
            )
            val_lm = {**lm_common, "transform": None, "enable_copy_paste": False, "seed": args.seed + 1}
            val_ds = LabelMeSlidingWindowDataset(
                val_entries,
                val_pos_flags,
                window_h=wh,
                window_w=wh,
                **val_lm,
            )
            n_pos_tr = sum(train_pos_flags)
            print(
                f"[info] LabelMe sliding-window: window={wh}, stride={sy}, train_patches={len(train_ds)} "
                f"(positive≈{n_pos_tr}), val_patches={len(val_ds)}, discard_no_component={discard_no_component}, "
                f"embedded_images={embedded}, skipped_no_image={skipped}, copy_paste={args.enable_copy_paste}"
            )
        else:
            train_ds = LabelMeCrackDataset(train_samples, seed=args.seed, **lm_common)
            val_ds = LabelMeCrackDataset(
                val_samples if val_samples else train_samples[: max(1, len(train_samples) // 5)],
                transform=None,
                enable_copy_paste=False,
                crack_labels=crack_labels,
                ignore_labels=ignore_labels,
                component_labels=component_labels,
                ncp_label=args.ncp_label,
                imgsz=args.imgsz,
                seed=args.seed + 1,
            )
            print(
                f"[info] LabelMe samples: train={len(train_ds)}, val={len(val_ds)}, "
                f"embedded_images={embedded}, skipped_no_image={skipped}, copy_paste={args.enable_copy_paste}"
            )
    else:
        if args.data_dir is None:
            raise ValueError("Use either --data-dir or --labelme-dir/--images-dir")
        data_dir = args.data_dir.expanduser().resolve()
        img_dir = data_dir / "images"
        msk_dir = data_dir / "masks"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            raise FileNotFoundError(f"Need images/ and masks/ under {data_dir}")

        train_stems, val_stems = split_stems(img_dir, val_ratio=args.val_ratio, seed=args.seed)
        train_stems = [s for s in train_stems if is_valid_image_mask_pair(img_dir, msk_dir, s)]
        val_stems = [s for s in val_stems if is_valid_image_mask_pair(img_dir, msk_dir, s)]
        n_valid = len(train_stems) + len(val_stems)
        if n_valid == 0:
            raise RuntimeError("No valid image/mask PNG pairs found after integrity check.")
        print(f"[info] valid stems after integrity check: train={len(train_stems)}, val={len(val_stems)}")
        train_ds = SegmentationPatchDataset(
            images_dir=img_dir,
            masks_dir=msk_dir,
            image_list=train_stems,
            transform=default_train_transforms(),
            num_classes=args.num_classes,
        )
        val_list = val_stems if val_stems else train_stems[: max(1, len(train_stems) // 5)]
        val_ds = SegmentationPatchDataset(
            images_dir=img_dir,
            masks_dir=msk_dir,
            image_list=val_list,
            transform=None,
            num_classes=args.num_classes,
        )

    device = torch.device(args.device)
    wts_list = build_weighted_sampler_weights(
        list(getattr(train_ds, "is_positive_mask", [])),
        float(args.positive_patch_ratio),
    )
    train_sampler: Optional[WeightedRandomSampler] = None
    if wts_list is not None:
        gen = torch.Generator().manual_seed(int(args.seed))
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(wts_list, dtype=torch.double),
            num_samples=len(wts_list),
            replacement=True,
            generator=gen,
        )
        print(
            f"[info] WeightedRandomSampler: positive_patch_ratio={args.positive_patch_ratio}, "
            f"n_train={len(wts_list)}, n_pos={sum(train_ds.is_positive_mask)}"
        )
    elif 0.0 < float(args.positive_patch_ratio) < 1.0:
        print("[info] positive-patch-ratio ignored (need both crack-positive and negative patches in train set)")

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Train loader empty; lower --batch-size")

    tw = args.teacher_weights.strip() or None
    model = DINOv3StageBUNet(
        weights_dir=tw,
        pretrained=True,
        device=device,
        num_classes=args.num_classes,
        bottleneck_dim=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
    ).to(device)
    if args.no_adapters:
        if args.stage_a_ckpt.strip():
            print("[info] --no-adapters set: ignoring --stage-a-ckpt (raw-backbone teacher)")
        print("[info] raw-backbone teacher: adapters kept at identity (no Stage-A weights loaded)")
    elif args.stage_a_ckpt.strip():
        model.load_stage_a_adapters(args.stage_a_ckpt.strip())
        print(f"[info] loaded stage-A adapters from: {args.stage_a_ckpt}")

    class_weights = parse_class_weights(args.class_weights, args.num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_index)
    dice_loss: nn.Module
    if args.num_classes == 2:
        dice_loss = CrackDiceLoss(ignore_index=args.ignore_index)
    else:
        dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(freeze_adapters=not args.train_adapters),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_dict[k] = str(v)
        else:
            args_dict[k] = v
    (args.output_dir / "config.json").write_text(json.dumps(args_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    start_epoch = 1
    best_iou = -1.0
    patience_counter = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch_load_compat(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and isinstance(ckpt["scaler"], dict) and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_iou" in ckpt:
            best_iou = float(ckpt["best_iou"])
            patience_counter = int(ckpt.get("patience_counter", 0))
        else:
            # Backward-compat for older checkpoints (no best_iou/patience fields):
            # recover from the sibling best.pt when available.
            best_iou = float(ckpt.get("val_iou", -1.0))
            patience_counter = 0
            best_path = resume_path.parent / "best.pt"
            if best_path.is_file():
                try:
                    bck = torch_load_compat(best_path, map_location="cpu", weights_only=False)
                    best_iou = float(bck.get("val_iou", best_iou))
                    patience_counter = max(0, int(ckpt.get("epoch", 0)) - int(bck.get("epoch", 0)))
                except Exception:
                    pass
        print(
            f"[resume] {resume_path} -> start_epoch={start_epoch} "
            f"best_iou={best_iou:.6f} patience_counter={patience_counter}",
            flush=True,
        )
        if start_epoch > args.epochs or patience_counter >= args.early_stop_patience:
            reason = (
                f"start_epoch={start_epoch} > epochs={args.epochs}"
                if start_epoch > args.epochs
                else f"patience_counter={patience_counter} >= {args.early_stop_patience}"
            )
            print(f"[resume] already finished/early-stopped ({reason}); nothing to train.", flush=True)
            (args.output_dir / ".completed").write_text(
                json.dumps({"last_epoch": int(ckpt.get("epoch", 0)), "target_epochs": args.epochs}),
                encoding="utf-8",
            )
            return

    epoch_metrics_csv = args.output_dir / "epoch_metrics.csv"
    prepare_epoch_metrics_csv(epoch_metrics_csv)
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        model.train()
        train_sum = 0.0
        n_train = 0
        for bi, batch in enumerate(train_loader, start=1):
            if args.max_steps > 0 and bi > args.max_steps:
                break
            x, y = resize_batch(batch, args.imgsz)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                l_ce = ce_loss(logits, y)
                if args.num_classes == 2:
                    l_dice = dice_loss(logits, y)
                else:
                    l_dice = dice_loss(logits, y, num_classes=args.num_classes)
                loss = args.lambda_ce * l_ce + args.lambda_dice * l_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach().cpu().item())
            n_train += 1
            if bi % args.log_every == 0:
                print(
                    f"[epoch {epoch:03d}] step {bi:04d}/{len(train_loader)} "
                    f"loss={loss.item():.5f} ce={l_ce.item():.5f} dice={l_dice.item():.5f}"
                )

        model.eval()
        if args.labelme_dir is not None:
            val_metrics = []
            with torch.no_grad():
                for vi, (ann_path, img_path) in enumerate(val_samples, start=1):
                    if args.max_val_batches > 0 and vi > args.max_val_batches:
                        break
                    data = read_labelme(ann_path)
                    img = load_labelme_image(data, img_path)
                    w, h = img.size
                    image_rgb = np.asarray(img, dtype=np.uint8)
                    gt, valid = build_gt_masks(
                        data,
                        (w, h),
                        crack_labels=crack_labels,
                        ignore_labels=ignore_labels,
                        component_labels=component_labels,
                    )
                    if not bool(valid.any()):
                        continue
                    pred = predict_teacher_mask(
                        model,
                        image_rgb,
                        imgsz=args.imgsz,
                        stride=args.val_stride,
                        num_classes=args.num_classes,
                        device=device,
                    )
                    val_metrics.append(compute_crack_metrics(gt, pred, valid))
            val_iou = aggregate_micro(val_metrics).iou if val_metrics else 0.0
        else:
            val_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for vi, batch in enumerate(val_loader, start=1):
                    if args.max_val_batches > 0 and vi > args.max_val_batches:
                        break
                    x, y = resize_batch(batch, args.imgsz)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x)
                        l_ce = ce_loss(logits, y)
                        if args.num_classes == 2:
                            l_dice = dice_loss(logits, y)
                        else:
                            l_dice = dice_loss(logits, y, num_classes=args.num_classes)
                        loss = args.lambda_ce * l_ce + args.lambda_dice * l_dice
                    val_sum += float(loss.detach().cpu().item())
                    n_val += 1
            val_loss = val_sum / max(1, n_val)
            # Legacy data-dir path: negate loss so the shared "maximize val_iou"
            # early-stopping below selects the lowest validation loss.
            val_iou = -val_loss

        train_avg = train_sum / max(1, n_train)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_avg:.6f} val_iou={val_iou:.6f}")

        improved = val_iou > best_iou
        if improved:
            best_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1

        append_epoch_metrics_csv(
            epoch_metrics_csv,
            {
                "epoch": epoch,
                "train_loss": train_avg,
                "val_iou": val_iou,
                "best_iou": best_iou,
                "patience_counter": patience_counter,
            },
        )

        latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "train_loss": train_avg,
            "val_iou": val_iou,
            "best_iou": best_iou,
            "patience_counter": patience_counter,
            "args": vars(args),
        }
        torch.save(latest, args.output_dir / "last.pt")
        if improved:
            torch.save(latest, args.output_dir / "best.pt")
            print(f"[ckpt] best updated: val_iou={best_iou:.6f}")
        if epoch % args.save_every == 0:
            torch.save(latest, args.output_dir / f"epoch_{epoch:03d}.pt")
        if patience_counter >= args.early_stop_patience:
            print(f"[early-stop] no improvement for {args.early_stop_patience} epochs, stopping at epoch {epoch}")
            break

    (args.output_dir / ".completed").write_text(
        json.dumps({"last_epoch": last_epoch, "target_epochs": args.epochs}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

