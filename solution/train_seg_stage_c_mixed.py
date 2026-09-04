#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage C: curated-only YOLO11-U-Net semantic distillation.

Implements docs/DINOv3_古建木构件教师域适应与异构蒸馏方案.md §6:
  - curated LabelMe only, no rough / pseudo-label branch;
  - semantic mask: 0=background, 1=crack, 255=ignore;
  - component_mask is preserved and used to build valid supervision V;
  - CE/Dice, feature distillation and attention distillation share V / V_l;
  - feature distillation is masked by V_l to avoid learning from component-outside regions.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checkpoint_io import torch_load_compat  # noqa: E402
from distill_modules import AdaptiveTeacherFusion, StudentChannelAlign  # noqa: E402
from models.dino_stage_b_unet import DINOv3StageBUNet  # noqa: E402
from models.teacher_vit import build_teacher  # noqa: E402
from models.yolo_unet_semseg import YoloUNetSemanticStudent  # noqa: E402
from scripts.labelme_crack_copy_paste import (  # noqa: E402
    alpha_blend_paste,
    extract_instances,
    find_paste_location,
    parse_csv_set,
    rasterize_masks,
    read_labelme,
    transform_instance,
)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".JPEG")
IGNORE_INDEX = 255


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clamp_num_workers_windows(args: argparse.Namespace) -> None:
    if Path().anchor and "\\" in Path().anchor and args.num_workers != 0:
        print(f"[info] Windows detected, forcing num_workers from {args.num_workers} to 0", flush=True)
        args.num_workers = 0


def parse_float_list(text: str, expected: int, name: str) -> Optional[list[float]]:
    text = (text or "").strip()
    if not text:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated floats, got {vals}")
    return vals


def effective_ignore(ignore_manual: np.ndarray, component: np.ndarray) -> np.ndarray:
    return ignore_manual | (~component)


def axis_window_starts(dim: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_toplefts(height: int, width: int, patch: int, stride: int) -> list[tuple[int, int]]:
    ys = axis_window_starts(height, patch, stride)
    xs = axis_window_starts(width, patch, stride)
    return [(y0, x0) for y0 in ys for x0 in xs]


def crop_top_left_2d(arr: np.ndarray, y0: int, x0: int, ph: int, pw: int, pad_value: int | bool) -> np.ndarray:
    h, w = arr.shape[:2]
    out = np.full((ph, pw), pad_value, dtype=arr.dtype)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + ph), min(w, x0 + pw)
    if ys >= ye or xs >= xe:
        return out
    dy, dx = ys - y0, xs - x0
    out[dy : dy + (ye - ys), dx : dx + (xe - xs)] = arr[ys:ye, xs:xe]
    return out


def crop_top_left_rgb(img: np.ndarray, y0: int, x0: int, ph: int, pw: int) -> np.ndarray:
    h, w, _ = img.shape
    out = np.zeros((ph, pw, 3), dtype=img.dtype)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + ph), min(w, x0 + pw)
    if ys >= ye or xs >= xe:
        return out
    dy, dx = ys - y0, xs - x0
    out[dy : dy + (ye - ys), dx : dx + (xe - xs)] = img[ys:ye, xs:xe]
    return out


def resolve_labelme_image(images_dir: Path, ann_path: Path, data: dict[str, Any]) -> Optional[Path]:
    search_dirs = [ann_path.parent, images_dir]
    sub_images = images_dir / "images"
    if sub_images.is_dir():
        search_dirs.append(sub_images)
    deduped: list[Path] = []
    for root in search_dirs:
        if root not in deduped:
            deduped.append(root)

    candidates: list[Path] = []
    image_path = data.get("imagePath")
    for root in deduped:
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
    raise FileNotFoundError("No external image and no embedded imageData in LabelMe JSON")


def labelme_image_size(data: dict[str, Any], img_path: Optional[Path]) -> tuple[int, int]:
    if img_path is not None:
        with Image.open(img_path) as im:
            return im.size
    with load_labelme_image(data, None) as im:
        return im.size


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def build_labelme_samples(labelme_dir: Path, images_dir: Path) -> tuple[list[tuple[Path, Optional[Path]]], int, int]:
    samples: list[tuple[Path, Optional[Path]]] = []
    embedded = 0
    skipped = 0
    for ann_path in sorted(labelme_dir.glob("*.json")):
        data = read_labelme(ann_path)
        img_path = resolve_labelme_image(images_dir, ann_path, data)
        if img_path is None:
            if isinstance(data.get("imageData"), str) and data.get("imageData"):
                embedded += 1
            else:
                skipped += 1
                continue
        elif not is_valid_image(img_path):
            skipped += 1
            continue
        samples.append((ann_path, img_path))
    return samples, embedded, skipped


def split_samples(samples: list[tuple[Path, Optional[Path]]], val_ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    out = list(samples)
    rng.shuffle(out)
    n_val = max(1, int(len(out) * val_ratio)) if len(out) > 1 else 0
    val = out[:n_val]
    train = out[n_val:] or out
    return train, val


def build_window_entries(
    samples: list[tuple[Path, Optional[Path]]],
    *,
    crack_labels: set[str],
    ignore_labels: set[str],
    component_labels: set[str],
    window_size: int,
    window_stride: int,
    discard_no_component: bool,
) -> tuple[list[tuple[Path, Optional[Path], int, int]], list[bool]]:
    entries: list[tuple[Path, Optional[Path], int, int]] = []
    is_positive: list[bool] = []
    for ann_path, img_path in samples:
        data = read_labelme(ann_path)
        w, h = labelme_image_size(data, img_path)
        crack, ignore, component, _other = rasterize_masks(
            data,
            (w, h),
            crack_labels,
            ignore_labels,
            component_labels,
            treat_empty_component_as_full_image=False,
        )
        ignore_eff = effective_ignore(ignore, component)
        for y0, x0 in sliding_window_toplefts(h, w, window_size, window_stride):
            component_win = crop_top_left_2d(component, y0, x0, window_size, window_size, False)
            if discard_no_component and not bool(component_win.any()):
                continue
            crack_win = crop_top_left_2d(crack, y0, x0, window_size, window_size, False)
            ignore_win = crop_top_left_2d(ignore_eff, y0, x0, window_size, window_size, True)
            entries.append((ann_path, img_path, y0, x0))
            is_positive.append(bool(np.any(crack_win & ~ignore_win)))
    return entries, is_positive


def labelme_has_any_label(data: dict[str, Any], labels: set[str]) -> bool:
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list) or not labels:
        return False
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip().lower()
        if label in labels:
            return True
    return False


def apply_same_image_copy_paste_patch(
    *,
    image: np.ndarray,
    crack_mask: np.ndarray,
    ignore_mask: np.ndarray,
    component_mask: np.ndarray,
    other_label_mask: np.ndarray,
    rng: random.Random,
    num_pastes: int,
    instance_attempt_multiplier: int,
    min_crack_area: int,
    bbox_padding: int,
    search_radius: int,
    max_tries_per_instance: int,
    inside_component_threshold: float,
    max_crack_overlap: float,
    max_other_overlap: float,
    brightness_mean_threshold: float,
    brightness_std_threshold: float,
    texture_angle_threshold: float,
    max_rotate_deg: float,
    scale_min: float,
    scale_max: float,
    alpha_dilate: int,
    alpha_blur: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Online same-image Copy-Paste on a cropped training patch.

    The pasted pixels are constrained by component/ignore/other-label masks and
    the semantic label update uses the transformed crack mask, not the feathered alpha.
    """
    source_crack = crack_mask & component_mask & (~ignore_mask)
    if num_pastes <= 0 or not bool(source_crack.any()) or not bool(component_mask.any()):
        return image, crack_mask, 0

    instances = extract_instances(image, source_crack, min_crack_area, bbox_padding)
    if not instances:
        return image, crack_mask, 0

    out_image = image.copy()
    out_crack = crack_mask.copy()
    pasted = 0
    attempts = max(num_pastes, 1) * max(instance_attempt_multiplier, 1)
    for _ in range(attempts):
        if pasted >= num_pastes:
            break
        instance = instances[rng.randrange(len(instances))]
        src_patch, src_mask = transform_instance(
            instance["patch"],
            instance["mask"],
            rng,
            max_rotate_deg,
            scale_min,
            scale_max,
            bbox_padding,
        )
        if int(src_mask.sum()) < min_crack_area:
            continue
        location = find_paste_location(
            out_image,
            src_patch,
            src_mask,
            instance["center"],
            out_crack,
            ignore_mask,
            component_mask,
            other_label_mask,
            rng,
            max_tries_per_instance,
            search_radius,
            inside_component_threshold,
            max_crack_overlap,
            max_other_overlap,
            brightness_mean_threshold,
            brightness_std_threshold,
            texture_angle_threshold,
        )
        if location is None:
            continue
        x, y = location
        alpha_blend_paste(out_image, src_patch, src_mask, x, y, alpha_dilate, alpha_blur)
        ph, pw = src_mask.shape
        out_crack[y : y + ph, x : x + pw] |= src_mask.astype(bool)
        pasted += 1
    return out_image, out_crack, pasted


class LabelMeStageCDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[Path, Optional[Path], int, int]],
        is_positive_mask: list[bool],
        *,
        crack_labels: set[str],
        ignore_labels: set[str],
        component_labels: set[str],
        window_size: int,
        imgsz: int,
        train: bool,
        seed: int,
        ncp_labels: set[str],
        copy_paste_prob: float,
        copy_paste_num_pastes: int,
        copy_paste_attempt_multiplier: int,
        copy_paste_min_crack_area: int,
        copy_paste_bbox_padding: int,
        copy_paste_search_radius: int,
        copy_paste_max_tries: int,
        copy_paste_inside_component_threshold: float,
        copy_paste_max_crack_overlap: float,
        copy_paste_max_other_overlap: float,
        copy_paste_brightness_mean_threshold: float,
        copy_paste_brightness_std_threshold: float,
        copy_paste_texture_angle_threshold: float,
        copy_paste_max_rotate_deg: float,
        copy_paste_scale_min: float,
        copy_paste_scale_max: float,
        copy_paste_alpha_dilate: int,
        copy_paste_alpha_blur: int,
    ) -> None:
        self.entries = entries
        self.is_positive_mask = is_positive_mask
        self.crack_labels = crack_labels
        self.ignore_labels = ignore_labels
        self.component_labels = component_labels
        self.window_size = int(window_size)
        self.imgsz = int(imgsz)
        self.train = bool(train)
        self.rng = random.Random(seed)
        self.ncp_labels = ncp_labels
        self.copy_paste_prob = float(copy_paste_prob)
        self.copy_paste_num_pastes = int(copy_paste_num_pastes)
        self.copy_paste_attempt_multiplier = int(copy_paste_attempt_multiplier)
        self.copy_paste_min_crack_area = int(copy_paste_min_crack_area)
        self.copy_paste_bbox_padding = int(copy_paste_bbox_padding)
        self.copy_paste_search_radius = int(copy_paste_search_radius)
        self.copy_paste_max_tries = int(copy_paste_max_tries)
        self.copy_paste_inside_component_threshold = float(copy_paste_inside_component_threshold)
        self.copy_paste_max_crack_overlap = float(copy_paste_max_crack_overlap)
        self.copy_paste_max_other_overlap = float(copy_paste_max_other_overlap)
        self.copy_paste_brightness_mean_threshold = float(copy_paste_brightness_mean_threshold)
        self.copy_paste_brightness_std_threshold = float(copy_paste_brightness_std_threshold)
        self.copy_paste_texture_angle_threshold = float(copy_paste_texture_angle_threshold)
        self.copy_paste_max_rotate_deg = float(copy_paste_max_rotate_deg)
        self.copy_paste_scale_min = float(copy_paste_scale_min)
        self.copy_paste_scale_max = float(copy_paste_scale_max)
        self.copy_paste_alpha_dilate = int(copy_paste_alpha_dilate)
        self.copy_paste_alpha_blur = int(copy_paste_alpha_blur)

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
        ignore_eff = effective_ignore(ignore, component)

        image_p = crop_top_left_rgb(image_np, y0, x0, self.window_size, self.window_size)
        crack_p = crop_top_left_2d(crack, y0, x0, self.window_size, self.window_size, False)
        component_p = crop_top_left_2d(component, y0, x0, self.window_size, self.window_size, False)
        ignore_p = crop_top_left_2d(ignore_eff, y0, x0, self.window_size, self.window_size, True)
        other_p = crop_top_left_2d(other, y0, x0, self.window_size, self.window_size, False)

        can_copy_paste = (
            self.train
            and self.copy_paste_prob > 0.0
            and self.copy_paste_num_pastes > 0
            and self.rng.random() < self.copy_paste_prob
            and not labelme_has_any_label(data, self.ncp_labels)
        )
        if can_copy_paste:
            image_p, crack_p, _num_pasted = apply_same_image_copy_paste_patch(
                image=image_p,
                crack_mask=crack_p,
                ignore_mask=ignore_p,
                component_mask=component_p,
                other_label_mask=other_p,
                rng=self.rng,
                num_pastes=self.copy_paste_num_pastes,
                instance_attempt_multiplier=self.copy_paste_attempt_multiplier,
                min_crack_area=self.copy_paste_min_crack_area,
                bbox_padding=self.copy_paste_bbox_padding,
                search_radius=self.copy_paste_search_radius,
                max_tries_per_instance=self.copy_paste_max_tries,
                inside_component_threshold=self.copy_paste_inside_component_threshold,
                max_crack_overlap=self.copy_paste_max_crack_overlap,
                max_other_overlap=self.copy_paste_max_other_overlap,
                brightness_mean_threshold=self.copy_paste_brightness_mean_threshold,
                brightness_std_threshold=self.copy_paste_brightness_std_threshold,
                texture_angle_threshold=self.copy_paste_texture_angle_threshold,
                max_rotate_deg=self.copy_paste_max_rotate_deg,
                scale_min=self.copy_paste_scale_min,
                scale_max=self.copy_paste_scale_max,
                alpha_dilate=self.copy_paste_alpha_dilate,
                alpha_blur=self.copy_paste_alpha_blur,
            )

        mask_p = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        mask_p[crack_p] = 1
        mask_p[ignore_p] = IGNORE_INDEX

        if self.train and self.rng.random() < 0.5:
            image_p = np.ascontiguousarray(image_p[:, ::-1])
            mask_p = np.ascontiguousarray(mask_p[:, ::-1])
            component_p = np.ascontiguousarray(component_p[:, ::-1])

        pil_img = Image.fromarray(image_p)
        pil_mask = Image.fromarray(mask_p, mode="L")
        pil_component = Image.fromarray(component_p.astype(np.uint8) * 255, mode="L")
        if self.imgsz > 0 and pil_img.size != (self.imgsz, self.imgsz):
            pil_img = pil_img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            pil_mask = pil_mask.resize((self.imgsz, self.imgsz), Image.NEAREST)
            pil_component = pil_component.resize((self.imgsz, self.imgsz), Image.NEAREST)

        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        mask_np = np.array(pil_mask, dtype=np.int64)
        component_np = np.array(pil_component, dtype=np.uint8) > 0
        valid_np = component_np & (mask_np != IGNORE_INDEX)
        stem = f"{ann_path.stem}_y{y0}_x{x0}"
        return {
            "img": torch.from_numpy(np.transpose(img_np, (2, 0, 1))),
            "mask": torch.from_numpy(mask_np).long(),
            "component": torch.from_numpy(component_np.astype(np.uint8)).bool(),
            "valid": torch.from_numpy(valid_np.astype(np.uint8)).bool(),
            "stem": stem,
        }


def collate_curated(batch: list[dict]) -> dict:
    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0).long(),
        "component": torch.stack([b["component"] for b in batch], dim=0).bool(),
        "valid": torch.stack([b["valid"] for b in batch], dim=0).bool(),
        "stem": [b["stem"] for b in batch],
    }


def build_weighted_sampler_weights(is_positive_mask: list[bool], positive_ratio: float) -> Optional[list[float]]:
    if not (0.0 < positive_ratio < 1.0) or not is_positive_mask:
        return None
    n_pos = sum(is_positive_mask)
    n_neg = len(is_positive_mask) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = positive_ratio / n_pos
    w_neg = (1.0 - positive_ratio) / n_neg
    return [w_pos if p else w_neg for p in is_positive_mask]


class CrackDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        tgt = (target == 1).float()
        prob = prob[valid]
        tgt = tgt[valid]
        inter = (prob * tgt).sum()
        denom = prob.sum() + tgt.sum()
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def amp_autocast(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def resolve_stage_b_config(ckpt_path: Path) -> dict:
    cfg = ckpt_path.parent / "config.json"
    return read_json(cfg) if cfg.is_file() else {}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


EPOCH_METRICS_FIELDS = [
    "epoch",
    "train_total",
    "train_ce",
    "train_dice",
    "train_feat",
    "train_attn",
    "val_seg",
    "best_val",
]


def prepare_epoch_metrics_csv(path: Path, start_epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_METRICS_FIELDS)
            writer.writeheader()
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        rows = []

    keep_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            epoch = int(str(row.get("epoch", "")).strip())
        except Exception:
            continue
        if epoch < start_epoch:
            keep_rows.append({k: row.get(k, "") for k in EPOCH_METRICS_FIELDS})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(keep_rows)


def append_epoch_metrics_csv(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_METRICS_FIELDS)
        writer.writerow({k: row.get(k, "") for k in EPOCH_METRICS_FIELDS})


def load_stage_b_teacher(args: argparse.Namespace, device: torch.device) -> DINOv3StageBUNet:
    ckpt_path = args.teacher_stage_b_ckpt.expanduser().resolve()
    cfg = resolve_stage_b_config(ckpt_path)
    teacher_weights = args.teacher_weights.strip() or cfg.get("teacher_weights", "")
    if not teacher_weights:
        raise ValueError("Teacher weights path is required via --teacher-weights or Stage-B config.json")
    teacher = DINOv3StageBUNet(
        weights_dir=teacher_weights,
        pretrained=not args.teacher_no_pretrained,
        device=device,
        num_classes=int(cfg.get("num_classes", args.num_classes)),
        bottleneck_dim=int(cfg.get("adapter_bottleneck", 64)),
        adapter_dropout=float(cfg.get("adapter_dropout", 0.1)),
    ).to(device)
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_raw_vit_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    teacher_weights = args.teacher_weights.strip()
    if not teacher_weights:
        raise ValueError("--teacher-weights is required when --teacher-mode raw_vit")
    teacher = build_teacher(
        img_size=args.teacher_img_size,
        pretrained=not args.teacher_no_pretrained,
        weights_dir=teacher_weights,
        device=device,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.teacher_mode == "stage_b":
        if args.teacher_stage_b_ckpt is None:
            raise ValueError("--teacher-stage-b-ckpt is required when --teacher-mode stage_b")
        return load_stage_b_teacher(args, device=device)
    if args.teacher_mode == "raw_vit":
        return load_raw_vit_teacher(args, device=device)
    raise ValueError(f"Unsupported teacher mode: {args.teacher_mode}")


def teacher_feature_dim(teacher: nn.Module) -> int:
    if hasattr(teacher, "hidden_size"):
        return int(getattr(teacher, "hidden_size"))
    if hasattr(teacher, "embed_dim"):
        return int(getattr(teacher, "embed_dim"))
    raise AttributeError("Teacher is missing hidden_size/embed_dim, cannot infer distillation channels")


def extract_teacher_feature_maps(teacher: nn.Module, x: torch.Tensor) -> Sequence[torch.Tensor]:
    if hasattr(teacher, "extract_adapted_feature_maps"):
        return teacher.extract_adapted_feature_maps(x)
    return teacher(x)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask[:, None].float(), size=size, mode="nearest").squeeze(1)


def masked_smooth_l1(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if valid.dim() == 3:
        valid = valid[:, None]
    valid = valid.to(device=student.device, dtype=student.dtype)
    denom = valid.sum() * student.shape[1]
    if float(denom.detach().cpu()) <= 0.0:
        return student.sum() * 0.0
    loss = F.smooth_l1_loss(student, teacher.detach(), reduction="none")
    return (loss * valid).sum() / (denom + eps)


def spatial_attention(feat: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    att = feat.float().pow(2).mean(dim=1, keepdim=True)
    if valid.dim() == 3:
        valid = valid[:, None]
    valid = valid.to(device=att.device, dtype=att.dtype)
    att = att * valid
    denom = torch.sqrt((att.pow(2).sum(dim=(2, 3), keepdim=True) / (valid.sum(dim=(2, 3), keepdim=True) + eps)) + eps)
    return att / denom


def compute_distill_losses(
    *,
    images_01: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    student_feats: Sequence[torch.Tensor],
    teacher: nn.Module,
    align: StudentChannelAlign,
    fusion: nn.Module,
    teacher_img_size: int,
    lambdas: Sequence[float],
    attn_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_feats = align(student_feats[0], student_feats[1], student_feats[2])
    target_sizes = [(s.shape[2], s.shape[3]) for s in s_feats]
    x_t = F.interpolate(images_01, size=(teacher_img_size, teacher_img_size), mode="bilinear", align_corners=False)
    x_t = imagenet_normalize(x_t)
    with torch.no_grad():
        t_feats = extract_teacher_feature_maps(teacher, x_t)
    t_targets = fusion(t_feats, target_sizes)

    feat_total = torch.zeros((), device=images_01.device, dtype=s_feats[0].dtype)
    attn_total = torch.zeros_like(feat_total)
    crack = mask == 1
    for i, (s_l, t_l) in enumerate(zip(s_feats, t_targets)):
        alpha = float(lambdas[i])
        if alpha <= 0.0:
            continue
        v_l = resize_mask(valid, s_l.shape[-2:]) > 0.5
        if not bool(v_l.any()):
            continue
        feat_total = feat_total + alpha * masked_smooth_l1(s_l, t_l, v_l)

        m_l = resize_mask(crack, s_l.shape[-2:]) > 0.5
        a_s = spatial_attention(s_l, v_l)
        a_t = spatial_attention(t_l.detach(), v_l)
        weight = v_l[:, None].to(dtype=a_s.dtype, device=a_s.device) * (1.0 + float(attn_gamma) * m_l[:, None].to(dtype=a_s.dtype, device=a_s.device))
        denom = weight.sum()
        if float(denom.detach().cpu()) > 0.0:
            attn_total = attn_total + alpha * (weight * (a_s - a_t).pow(2)).sum() / (denom + 1e-6)
    return feat_total, attn_total


def needs_teacher_distill(args: argparse.Namespace) -> bool:
    return float(args.lambda_feat_curated) > 0.0 or float(args.lambda_attn_curated) > 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C: curated-only YOLO11-U-Net semantic distillation")
    p.add_argument("--curated-labelme-dir", type=Path, default=None, help="Curated LabelMe JSON dir; preferred Stage-C input")
    p.add_argument("--curated-dir", type=Path, default=None, help="Alias/fallback for --curated-labelme-dir")
    p.add_argument("--images-dir", type=Path, default=None, help="Image dir corresponding to LabelMe JSONs; default: labelme dir")
    p.add_argument("--student-weights", type=str, default="solution/yolo11m-seg.pt")
    p.add_argument("--teacher-mode", type=str, default="stage_b", choices=("stage_b", "raw_vit"))
    p.add_argument("--teacher-stage-b-ckpt", type=Path, default=None)
    p.add_argument("--teacher-weights", type=str, default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--teacher-img-size", type=int, default=1024)
    p.add_argument("--window-size", type=int, default=0, help="0 = use --imgsz")
    p.add_argument("--window-stride", type=int, default=800)
    p.add_argument("--keep-outside-component-patches", action="store_true")
    p.add_argument("--positive-patch-ratio", type=float, default=-1.0)
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--ncp-labels", type=str, default="ncp", help="Labels that disable Copy-Paste for the whole image")
    p.add_argument("--copy-paste-prob", type=float, default=0.0, help="Online same-image Copy-Paste probability for train patches")
    p.add_argument("--copy-paste-num-pastes", type=int, default=3)
    p.add_argument("--copy-paste-attempt-multiplier", type=int, default=4)
    p.add_argument("--copy-paste-min-crack-area", type=int, default=30)
    p.add_argument("--copy-paste-bbox-padding", type=int, default=16)
    p.add_argument("--copy-paste-search-radius", type=int, default=400)
    p.add_argument("--copy-paste-max-tries", type=int, default=150)
    p.add_argument("--copy-paste-inside-component-threshold", type=float, default=0.98)
    p.add_argument("--copy-paste-max-crack-overlap", type=float, default=0.05)
    p.add_argument("--copy-paste-max-other-overlap", type=float, default=0.02)
    p.add_argument("--copy-paste-brightness-mean-threshold", type=float, default=25.0)
    p.add_argument("--copy-paste-brightness-std-threshold", type=float, default=20.0)
    p.add_argument("--copy-paste-texture-angle-threshold", type=float, default=30.0)
    p.add_argument("--copy-paste-max-rotate-deg", type=float, default=10.0)
    p.add_argument("--copy-paste-scale-min", type=float, default=0.9)
    p.add_argument("--copy-paste-scale-max", type=float, default=1.1)
    p.add_argument("--copy-paste-alpha-dilate", type=int, default=3)
    p.add_argument("--copy-paste-alpha-blur", type=int, default=7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size-curated", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs (num_workers>0)")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0 only)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--lambda-dice", type=float, default=1.0)
    p.add_argument("--lambda-feat-curated", type=float, default=0.5)
    p.add_argument("--lambda-attn-curated", type=float, default=0.2)
    p.add_argument("--attn-crack-gamma", type=float, default=3.0)
    p.add_argument("--lambda-l1", type=float, default=0.5)
    p.add_argument("--lambda-l2", type=float, default=0.3)
    p.add_argument("--lambda-l3", type=float, default=0.2)
    p.add_argument("--class-weights", type=str, default="", help="Optional CE class weights, e.g. 0.2,2.0")
    p.add_argument("--student-feat-channels", type=str, default="", help="Optional override C3,C4,C5")
    p.add_argument("--decoder-channels", type=str, default="256,192,128,64")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--teacher-no-pretrained", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clamp_num_workers_windows(args)
    set_seed(args.seed)
    use_teacher = needs_teacher_distill(args)
    if args.num_classes != 2:
        raise ValueError("Current Stage-C crack semantic plan expects --num-classes 2")
    if args.imgsz % 16 != 0 or args.teacher_img_size % 16 != 0:
        raise ValueError("--imgsz and --teacher-img-size must be divisible by 16")
    if use_teacher and args.teacher_mode == "stage_b" and args.teacher_stage_b_ckpt is None:
        raise ValueError("Set --teacher-stage-b-ckpt when --teacher-mode stage_b")
    if use_teacher and args.teacher_mode == "raw_vit" and not args.teacher_weights.strip():
        raise ValueError("Set --teacher-weights when --teacher-mode raw_vit")

    labelme_dir = args.curated_labelme_dir or args.curated_dir
    if labelme_dir is None:
        raise ValueError("Set --curated-labelme-dir (or --curated-dir as alias)")
    labelme_dir = labelme_dir.expanduser().resolve()
    images_dir = (args.images_dir or labelme_dir).expanduser().resolve()
    if not labelme_dir.is_dir():
        raise FileNotFoundError(f"LabelMe dir not found: {labelme_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    all_samples, embedded, skipped = build_labelme_samples(labelme_dir, images_dir)
    if not all_samples:
        raise RuntimeError(f"No LabelMe annotations with matching images under {labelme_dir}")
    train_samples, val_samples = split_samples(all_samples, args.val_ratio, args.seed)

    crack_labels = parse_csv_set(args.crack_labels)
    ignore_labels = parse_csv_set(args.ignore_labels)
    component_labels = parse_csv_set(args.component_labels)
    ncp_labels = parse_csv_set(args.ncp_labels)
    if not (0.0 <= args.copy_paste_prob <= 1.0):
        raise ValueError("--copy-paste-prob must be in [0, 1]")
    if args.copy_paste_scale_min <= 0.0 or args.copy_paste_scale_max <= 0.0:
        raise ValueError("--copy-paste-scale-min/max must be positive")
    if args.copy_paste_scale_min > args.copy_paste_scale_max:
        raise ValueError("--copy-paste-scale-min must be <= --copy-paste-scale-max")
    window_size = int(args.window_size) if int(args.window_size) > 0 else int(args.imgsz)
    discard_no_component = not bool(args.keep_outside_component_patches)
    train_entries, train_pos = build_window_entries(
        train_samples,
        crack_labels=crack_labels,
        ignore_labels=ignore_labels,
        component_labels=component_labels,
        window_size=window_size,
        window_stride=args.window_stride,
        discard_no_component=discard_no_component,
    )
    val_src = val_samples if val_samples else train_samples[: max(1, len(train_samples) // 5)]
    val_entries, val_pos = build_window_entries(
        val_src,
        crack_labels=crack_labels,
        ignore_labels=ignore_labels,
        component_labels=component_labels,
        window_size=window_size,
        window_stride=args.window_stride,
        discard_no_component=discard_no_component,
    )
    if not train_entries:
        raise RuntimeError("No training patches. Check component labels or use --keep-outside-component-patches.")
    if not val_entries:
        raise RuntimeError("No validation patches. Check component labels or use --keep-outside-component-patches.")

    train_ds = LabelMeStageCDataset(
        train_entries,
        train_pos,
        crack_labels=crack_labels,
        ignore_labels=ignore_labels,
        component_labels=component_labels,
        window_size=window_size,
        imgsz=args.imgsz,
        train=True,
        seed=args.seed,
        ncp_labels=ncp_labels,
        copy_paste_prob=args.copy_paste_prob,
        copy_paste_num_pastes=args.copy_paste_num_pastes,
        copy_paste_attempt_multiplier=args.copy_paste_attempt_multiplier,
        copy_paste_min_crack_area=args.copy_paste_min_crack_area,
        copy_paste_bbox_padding=args.copy_paste_bbox_padding,
        copy_paste_search_radius=args.copy_paste_search_radius,
        copy_paste_max_tries=args.copy_paste_max_tries,
        copy_paste_inside_component_threshold=args.copy_paste_inside_component_threshold,
        copy_paste_max_crack_overlap=args.copy_paste_max_crack_overlap,
        copy_paste_max_other_overlap=args.copy_paste_max_other_overlap,
        copy_paste_brightness_mean_threshold=args.copy_paste_brightness_mean_threshold,
        copy_paste_brightness_std_threshold=args.copy_paste_brightness_std_threshold,
        copy_paste_texture_angle_threshold=args.copy_paste_texture_angle_threshold,
        copy_paste_max_rotate_deg=args.copy_paste_max_rotate_deg,
        copy_paste_scale_min=args.copy_paste_scale_min,
        copy_paste_scale_max=args.copy_paste_scale_max,
        copy_paste_alpha_dilate=args.copy_paste_alpha_dilate,
        copy_paste_alpha_blur=args.copy_paste_alpha_blur,
    )
    val_ds = LabelMeStageCDataset(
        val_entries,
        val_pos,
        crack_labels=crack_labels,
        ignore_labels=ignore_labels,
        component_labels=component_labels,
        window_size=window_size,
        imgsz=args.imgsz,
        train=False,
        seed=args.seed + 1,
        ncp_labels=ncp_labels,
        copy_paste_prob=0.0,
        copy_paste_num_pastes=0,
        copy_paste_attempt_multiplier=args.copy_paste_attempt_multiplier,
        copy_paste_min_crack_area=args.copy_paste_min_crack_area,
        copy_paste_bbox_padding=args.copy_paste_bbox_padding,
        copy_paste_search_radius=args.copy_paste_search_radius,
        copy_paste_max_tries=args.copy_paste_max_tries,
        copy_paste_inside_component_threshold=args.copy_paste_inside_component_threshold,
        copy_paste_max_crack_overlap=args.copy_paste_max_crack_overlap,
        copy_paste_max_other_overlap=args.copy_paste_max_other_overlap,
        copy_paste_brightness_mean_threshold=args.copy_paste_brightness_mean_threshold,
        copy_paste_brightness_std_threshold=args.copy_paste_brightness_std_threshold,
        copy_paste_texture_angle_threshold=args.copy_paste_texture_angle_threshold,
        copy_paste_max_rotate_deg=args.copy_paste_max_rotate_deg,
        copy_paste_scale_min=args.copy_paste_scale_min,
        copy_paste_scale_max=args.copy_paste_scale_max,
        copy_paste_alpha_dilate=args.copy_paste_alpha_dilate,
        copy_paste_alpha_blur=args.copy_paste_alpha_blur,
    )

    sampler = None
    weights = build_weighted_sampler_weights(train_ds.is_positive_mask, args.positive_patch_ratio)
    if weights is not None:
        gen = torch.Generator().manual_seed(int(args.seed))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=gen)
        print(
            f"[info] WeightedRandomSampler: positive_patch_ratio={args.positive_patch_ratio}, "
            f"n_train={len(weights)}, n_pos={sum(train_ds.is_positive_mask)}",
            flush=True,
        )
    elif 0.0 < args.positive_patch_ratio < 1.0:
        print("[info] positive-patch-ratio ignored (need both positive and negative patches)", flush=True)

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size_curated,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_curated,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size_curated,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_curated,
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Train loader empty; lower --batch-size-curated")

    decoder_channels = parse_float_list(args.decoder_channels, 4, "--decoder-channels")
    assert decoder_channels is not None
    student = YoloUNetSemanticStudent(
        args.student_weights,
        num_classes=args.num_classes,
        device=device,
        decoder_channels=[int(x) for x in decoder_channels],
    ).to(device)
    if args.student_feat_channels.strip():
        c3, c4, c5 = [int(x.strip()) for x in args.student_feat_channels.split(",")]
    else:
        c3, c4, c5 = student.neck_channels

    teacher: nn.Module | None = None
    align: StudentChannelAlign | None = None
    fusion: AdaptiveTeacherFusion | None = None
    teacher_dim: int | None = None
    if use_teacher:
        teacher = load_teacher(args, device=device)
        teacher_dim = teacher_feature_dim(teacher)
        align = StudentChannelAlign(in_channels=(c3, c4, c5), out_channels=teacher_dim).to(device)
        fusion = AdaptiveTeacherFusion(channels=teacher_dim).to(device)

    class_weights = parse_float_list(args.class_weights, args.num_classes, "--class-weights")
    ce_weight = torch.tensor(class_weights, device=device, dtype=torch.float32) if class_weights else None
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=IGNORE_INDEX)
    dice_loss_fn = CrackDiceLoss(ignore_index=IGNORE_INDEX).to(device)

    params = list(student.parameters())
    if align is not None:
        params += list(align.parameters())
    if fusion is not None:
        params += list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lambdas = (args.lambda_l1, args.lambda_l2, args.lambda_l3)

    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        ckpt = torch_load_compat(resume_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["student"], strict=True)
        if align is not None and "align" in ckpt:
            align.load_state_dict(ckpt["align"], strict=True)
        if fusion is not None and "fusion" in ckpt:
            fusion.load_state_dict(ckpt["fusion"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and isinstance(ckpt["scaler"], dict):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[resume] {resume_path} -> start_epoch={start_epoch} best_val={best_val:.4f}", flush=True)

    config = {
        **vars(args),
        "curated_labelme_dir_resolved": str(labelme_dir),
        "images_dir_resolved": str(images_dir),
        "neck_channels": (c3, c4, c5),
        "teacher_enabled": use_teacher,
        "teacher_mode": args.teacher_mode if use_teacher else "disabled",
        "teacher_feature_dim": teacher_dim,
        "amp": use_amp,
        "window_size_resolved": window_size,
        "train_patches": len(train_ds),
        "val_patches": len(val_ds),
        "train_positive_patches": int(sum(train_ds.is_positive_mask)),
        "val_positive_patches": int(sum(val_ds.is_positive_mask)),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    epoch_metrics_csv = args.output_dir / "epoch_metrics.csv"
    prepare_epoch_metrics_csv(epoch_metrics_csv, start_epoch=start_epoch)

    print(
        f"[StageC] train_patches={len(train_ds)} val_patches={len(val_ds)} "
        f"positive={sum(train_ds.is_positive_mask)} window={window_size} stride={args.window_stride} "
        f"copy_paste_prob={args.copy_paste_prob} copy_paste_num={args.copy_paste_num_pastes} "
        f"teacher_enabled={use_teacher} teacher_mode={args.teacher_mode if use_teacher else 'disabled'} teacher_dim={teacher_dim} "
        f"embedded_images={embedded} skipped={skipped} neck={(c3, c4, c5)} amp={use_amp} out={args.output_dir}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        if align is not None:
            align.train()
        if fusion is not None:
            fusion.train()
        running = {"ce": 0.0, "dice": 0.0, "feat": 0.0, "attn": 0.0, "total": 0.0}
        n_batches = 0
        n_train = len(train_loader) if args.max_steps <= 0 else min(len(train_loader), args.max_steps)
        print(f"\n======== Epoch {epoch}/{args.epochs} ({n_train} batches) ========", flush=True)

        for bi, batch in enumerate(train_loader, start=1):
            if args.max_steps > 0 and bi > args.max_steps:
                break
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, use_amp):
                logits, feats = student(img)
                ce = ce_loss_fn(logits, mask)
                dice = dice_loss_fn(logits, mask)
                feat = torch.zeros((), device=device, dtype=logits.dtype)
                attn = torch.zeros((), device=device, dtype=logits.dtype)
                if use_teacher:
                    assert teacher is not None and align is not None and fusion is not None
                    feat, attn = compute_distill_losses(
                        images_01=img,
                        mask=mask,
                        valid=valid,
                        student_feats=feats,
                        teacher=teacher,
                        align=align,
                        fusion=fusion,
                        teacher_img_size=args.teacher_img_size,
                        lambdas=lambdas,
                        attn_gamma=args.attn_crack_gamma,
                    )
                total = (
                    args.lambda_ce * ce
                    + args.lambda_dice * dice
                    + args.lambda_feat_curated * feat
                    + args.lambda_attn_curated * attn
                )

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            running["ce"] += float(ce.detach())
            running["dice"] += float(dice.detach())
            running["feat"] += float(feat.detach())
            running["attn"] += float(attn.detach())
            running["total"] += float(total.detach())
            n_batches += 1

            if args.log_every > 0 and (bi == 1 or bi % args.log_every == 0):
                print(
                    f"  batch {bi}/{n_train} total={running['total']/n_batches:.4f} "
                    f"ce={running['ce']/n_batches:.4f} dice={running['dice']/n_batches:.4f} "
                    f"feat={running['feat']/n_batches:.4f} attn={running['attn']/n_batches:.4f}",
                    flush=True,
                )

        student.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for vi, batch in enumerate(val_loader, start=1):
                if args.max_val_batches > 0 and vi > args.max_val_batches:
                    break
                img = batch["img"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with amp_autocast(device, use_amp):
                    logits, _feats = student(img)
                    val_loss = args.lambda_ce * ce_loss_fn(logits, mask) + args.lambda_dice * dice_loss_fn(logits, mask)
                val_losses.append(float(val_loss.detach().cpu()))
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0

        print(
            f"Epoch {epoch}/{args.epochs} train_total={running['total']/max(n_batches,1):.4f} "
            f"train_ce={running['ce']/max(n_batches,1):.4f} train_dice={running['dice']/max(n_batches,1):.4f} "
            f"train_feat={running['feat']/max(n_batches,1):.4f} train_attn={running['attn']/max(n_batches,1):.4f} "
            f"val_seg={mean_val:.4f}",
            flush=True,
        )

        improved = mean_val <= best_val
        if improved:
            best_val = mean_val
        append_epoch_metrics_csv(
            epoch_metrics_csv,
            {
                "epoch": epoch,
                "train_total": running["total"] / max(n_batches, 1),
                "train_ce": running["ce"] / max(n_batches, 1),
                "train_dice": running["dice"] / max(n_batches, 1),
                "train_feat": running["feat"] / max(n_batches, 1),
                "train_attn": running["attn"] / max(n_batches, 1),
                "val_seg": mean_val,
                "best_val": best_val,
            },
        )
        ckpt = {
            "epoch": epoch,
            "student": student.state_dict(),
            "align": align.state_dict() if align is not None else None,
            "fusion": fusion.state_dict() if fusion is not None else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "best_val": best_val,
            "args": vars(args),
            "neck_channels": (c3, c4, c5),
            "model_type": "yolo_unet_semantic",
            "valid_region": "component_mask & (mask != 255)",
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if improved:
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"[ckpt] best updated: val_seg={best_val:.4f}", flush=True)

    print(f"[done] best val_seg={best_val:.4f} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
