# -*- coding: utf-8 -*-
"""
Stage-A: DINOv3 teacher domain adaptation with self-supervised training.

Key choices (aligned with docs/DINOv3_古建木构件教师域适应与异构蒸馏方案.md):
- frozen DINOv3 backbone
- bottleneck residual adapters on blocks 3/4, 7/8, 11/12
- DINO-style EMA self-distillation with multi-crop augmentation
- aspect-ratio-aware multi-crop for elongated ancient timber images
- uses all domain images; supports both image files and LabelMe JSON inputs
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.dino_stage_a import DINOv3StageAModel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clamp_num_workers_windows(args: argparse.Namespace) -> None:
    if os.name == "nt" and args.num_workers != 0:
        print(f"[info] Windows detected, forcing num_workers from {args.num_workers} to 0")
        args.num_workers = 0


def _parse_float_pair(text: str, name: str) -> tuple[float, float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'min,max', got: {text}")
    a, b = float(parts[0]), float(parts[1])
    if not (0.0 < a < b <= 1.0):
        raise ValueError(f"{name} must satisfy 0 < min < max <= 1, got: {text}")
    return a, b


@dataclass(frozen=True)
class CropSpec:
    target_size: int
    num_crops: int
    normal_side_frac: tuple[float, float]
    short_side_frac: tuple[float, float]
    long_side_frac: tuple[float, float]


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    ann_boxes: tuple[tuple[int, int, int, int], ...] = ()


def _parse_int_list(text: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("adapter-indices cannot be empty")
    return out


def _resolve_roots(raw_roots: Sequence[str]) -> list[Path]:
    roots: list[Path] = []
    for item in raw_roots:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"input path does not exist: {p}")
            roots.append(p)
    if not roots:
        raise ValueError("No valid --input-roots provided.")
    return roots


def _load_labelme_ann_boxes(path: Path) -> tuple[tuple[int, int, int, int], ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8-sig"))

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return ()

    boxes: list[tuple[int, int, int, int]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
            except Exception:
                continue
        if not xs or not ys:
            continue
        x1 = int(math.floor(min(xs)))
        y1 = int(math.floor(min(ys)))
        x2 = int(math.ceil(max(xs)))
        y2 = int(math.ceil(max(ys)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return tuple(boxes)


def discover_samples(input_roots: Sequence[Path]) -> list[SampleRecord]:
    image_map: dict[tuple[Path, str], SampleRecord] = {}
    json_map: dict[tuple[Path, str], SampleRecord] = {}

    def _key(p: Path) -> tuple[Path, str]:
        return (p.parent.resolve(), p.stem.lower())

    def _add_path(p: Path) -> None:
        suf = p.suffix.lower()
        if suf in IMAGE_SUFFIXES:
            image_map.setdefault(_key(p), SampleRecord(path=p))
        elif suf == ".json":
            json_map[_key(p)] = SampleRecord(path=p, ann_boxes=_load_labelme_ann_boxes(p))

    for root in input_roots:
        if root.is_file():
            _add_path(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            _add_path(p)

    # Prefer JSON records when both image and JSON exist for the same stem in the same folder.
    merged: dict[tuple[Path, str], SampleRecord] = dict(image_map)
    merged.update(json_map)
    samples = sorted(merged.values(), key=lambda s: str(s.path))
    if not samples:
        joined = ", ".join(str(p) for p in input_roots)
        raise FileNotFoundError(f"No images/LabelMe JSON found under: {joined}")
    return samples


def load_rgb_from_sample(path: Path) -> Image.Image:
    suf = path.suffix.lower()
    if suf in IMAGE_SUFFIXES:
        with Image.open(path) as img:
            return img.convert("RGB")

    if suf != ".json":
        raise ValueError(f"Unsupported sample file: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8-sig"))

    image_data = data.get("imageData")
    if isinstance(image_data, str) and image_data.strip():
        raw = base64.b64decode(image_data)
        with Image.open(BytesIO(raw)) as img:
            return img.convert("RGB")

    image_path = data.get("imagePath")
    if isinstance(image_path, str) and image_path.strip():
        raw_image_path = Path(image_path)
        image_name = raw_image_path.name
        candidate_dirs = [
            path.parent,
            path.parent.parent,
            path.parent.parent / "images",
            path.parent.parent / "image",
            path.parent.parent / "train",
            path.parent.parent / "valid",
            path.parent.parent / "val",
            path.parent.parent / "test",
        ]
        candidate_paths: list[Path] = []
        candidate_paths.append((path.parent / raw_image_path).resolve())
        candidate_paths.append((path.parent.parent / raw_image_path).resolve())
        for d in candidate_dirs:
            candidate_paths.append((d / image_name).resolve())

        stem = Path(image_name).stem if image_name else path.stem
        for d in candidate_dirs:
            for suf in IMAGE_SUFFIXES:
                candidate_paths.append((d / f"{stem}{suf}").resolve())
                candidate_paths.append((d / f"{stem}{suf.upper()}").resolve())

        seen: set[Path] = set()
        for candidate in candidate_paths:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                with Image.open(candidate) as img:
                    return img.convert("RGB")

    raise FileNotFoundError(f"LabelMe JSON has no usable imageData/imagePath: {path}")


def _resize_with_aspect_and_pad(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize tensor to fit target square while keeping aspect ratio, then pad."""
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    _, h, w = x.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid tensor spatial size: {(h, w)}")
    scale = min(target_size / h, target_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0).squeeze(0)
    return x


def _sample_crop_box(
    width: int,
    height: int,
    *,
    spec: CropSpec,
    elongated_ratio_threshold: float,
    ann_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    include_ann_prob: float = 0.8,
    max_crop_aspect: float = 1.6,
) -> tuple[int, int, int, int]:
    short_side = min(width, height)
    long_side = max(width, height)
    aspect = long_side / max(1, short_side)

    if aspect >= elongated_ratio_threshold:
        crop_short = int(round(random.uniform(*spec.short_side_frac) * short_side))
        crop_long = int(round(random.uniform(*spec.long_side_frac) * long_side))
        if width >= height:
            crop_w, crop_h = crop_long, crop_short
        else:
            crop_w, crop_h = crop_short, crop_long
    else:
        side = int(round(random.uniform(*spec.normal_side_frac) * short_side))
        side = max(16, min(side, short_side))
        crop_w = side
        crop_h = side

    if max_crop_aspect > 1.0:
        if crop_w >= crop_h:
            crop_w = min(crop_w, int(round(crop_h * max_crop_aspect)))
        else:
            crop_h = min(crop_h, int(round(crop_w * max_crop_aspect)))

    chosen_box: tuple[int, int, int, int] | None = None
    if ann_boxes and random.random() < include_ann_prob:
        chosen_box = random.choice(list(ann_boxes))

    crop_w = max(16, min(crop_w, width))
    crop_h = max(16, min(crop_h, height))

    if chosen_box is None:
        left = 0 if crop_w >= width else random.randint(0, width - crop_w)
        top = 0 if crop_h >= height else random.randint(0, height - crop_h)
    else:
        bx1, by1, bx2, by2 = chosen_box
        cx = int(round((bx1 + bx2) / 2.0))
        cy = int(round((by1 + by2) / 2.0))
        # Bias crops around the annotation center instead of forcing full-box containment,
        # which can create overly elongated crops and excessive black padding.
        jitter_x = max(8, crop_w // 8)
        jitter_y = max(8, crop_h // 8)
        target_cx = cx + random.randint(-jitter_x, jitter_x)
        target_cy = cy + random.randint(-jitter_y, jitter_y)
        target_cx = max(0, min(width - 1, target_cx))
        target_cy = max(0, min(height - 1, target_cy))

        left = max(0, min(width - crop_w, target_cx - crop_w // 2))
        top = max(0, min(height - crop_h, target_cy - crop_h // 2))
    return left, top, left + crop_w, top + crop_h


def _build_crop_transform(
    *,
    spec: CropSpec,
    elongated_ratio_threshold: float,
    color_jitter_strength: float = 0.4,
    include_ann_prob: float = 0.8,
    max_crop_aspect: float = 1.6,
):
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF

    cj = T.ColorJitter(
        brightness=0.8 * color_jitter_strength,
        contrast=0.8 * color_jitter_strength,
        saturation=0.8 * color_jitter_strength,
        hue=0.2 * color_jitter_strength,
    )
    blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    gray = T.Grayscale(num_output_channels=3)

    def _transform(image: Image.Image, ann_boxes: Sequence[tuple[int, int, int, int]] | None = None) -> torch.Tensor:
        crop_box = _sample_crop_box(
            image.width,
            image.height,
            spec=spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            ann_boxes=ann_boxes,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
        )
        crop = image.crop(crop_box)
        if random.random() < 0.5:
            crop = TF.hflip(crop)
        if random.random() < 0.8:
            crop = cj(crop)
        if random.random() < 0.2:
            crop = gray(crop)
        if random.random() < 0.3:
            crop = blur(crop)

        x = TF.to_tensor(crop)
        x = _resize_with_aspect_and_pad(x, spec.target_size)
        x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return x

    return _transform


class MultiCropAug:
    def __init__(
        self,
        *,
        global_spec: CropSpec,
        mid_spec: CropSpec,
        local_spec: CropSpec,
        elongated_ratio_threshold: float = 2.5,
        include_ann_prob: float = 0.8,
        max_crop_aspect: float = 1.6,
        num_global_crops: int = 2,
        num_mid_crops: int = 2,
        num_local_crops: int = 6,
    ) -> None:
        self.num_global_crops = num_global_crops
        self.num_mid_crops = num_mid_crops
        self.num_local_crops = num_local_crops
        self.global_tf = _build_crop_transform(
            spec=global_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
        )
        self.mid_tf = _build_crop_transform(
            spec=mid_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
        )
        self.local_tf = _build_crop_transform(
            spec=local_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
        )
        self.global_spec = global_spec
        self.mid_spec = mid_spec
        self.local_spec = local_spec

    def __call__(self, image: Image.Image, ann_boxes: Sequence[tuple[int, int, int, int]] | None = None) -> list[torch.Tensor]:
        crops: list[torch.Tensor] = []
        for _ in range(self.num_global_crops):
            crops.append(self.global_tf(image, ann_boxes))
        for _ in range(self.num_mid_crops):
            crops.append(self.mid_tf(image, ann_boxes))
        for _ in range(self.num_local_crops):
            crops.append(self.local_tf(image, ann_boxes))
        return crops

    def view_names(self) -> list[str]:
        names: list[str] = []
        names.extend([f"global_{i+1}" for i in range(self.num_global_crops)])
        names.extend([f"mid_{i+1}" for i in range(self.num_mid_crops)])
        names.extend([f"local_{i+1}" for i in range(self.num_local_crops)])
        return names


def _tensor_to_pil_denorm(x: torch.Tensor) -> Image.Image:
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = (x.detach().cpu() * std.cpu() + mean.cpu()).clamp(0.0, 1.0)
    arr = (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _fit_to_box(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (size, size), fill)
    fitted = ImageOps.contain(img, (size, size))
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def _fit_with_boxes(
    img: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    size: int,
    fill=(255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGB", (size, size), fill)
    fitted = ImageOps.contain(img, (size, size))
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    if not boxes:
        return canvas

    sx = fitted.width / max(1, img.width)
    sy = fitted.height / max(1, img.height)
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2 in boxes:
        fx1 = left + int(round(x1 * sx))
        fy1 = top + int(round(y1 * sy))
        fx2 = left + int(round(x2 * sx))
        fy2 = top + int(round(y2 * sy))
        draw.rectangle((fx1, fy1, fx2, fy2), outline=(255, 0, 0), width=2)
    return canvas


def save_multicrop_previews(
    samples: Sequence[SampleRecord],
    transform: MultiCropAug,
    output_dir: Path,
    max_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = transform.view_names()
    cell = max(192, transform.global_spec.target_size // 2)
    margin = 16
    title_h = 22
    cols = 3

    for idx, sample in enumerate(samples[: max_samples]):
        image = load_rgb_from_sample(sample.path)
        crops = transform(image, sample.ann_boxes)
        panels: list[tuple[str, Image.Image]] = [("original", _fit_with_boxes(image, sample.ann_boxes, cell))]
        for name, crop in zip(names, crops):
            panels.append((name, _fit_to_box(_tensor_to_pil_denorm(crop), cell)))

        rows = math.ceil(len(panels) / cols)
        canvas_w = cols * (cell + margin) + margin
        canvas_h = rows * (cell + title_h + margin) + margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)

        for i, (label, panel) in enumerate(panels):
            r = i // cols
            c = i % cols
            x = margin + c * (cell + margin)
            y = margin + r * (cell + title_h + margin)
            draw.text((x, y), label, fill=(20, 20, 20))
            canvas.paste(panel, (x, y + title_h))

        stem = sample.path.stem.replace(" ", "_")
        out_path = output_dir / f"{idx:03d}_{stem}_multicrop_preview.png"
        canvas.save(out_path)
    print(f"[info] saved {min(max_samples, len(samples))} multi-crop previews to {output_dir}")


class UnlabeledMultiCropDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        transform: MultiCropAug,
        *,
        cache_mode: str = "none",
        cache_max_items: int = 0,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.cache_mode = cache_mode
        self.cache_max_items = cache_max_items
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        if self.cache_mode not in {"none", "ram"}:
            raise ValueError(f"Unknown cache_mode: {self.cache_mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_put(self, key: str, arr: np.ndarray) -> None:
        if self.cache_mode != "ram":
            return
        if key in self._cache:
            return
        self._cache[key] = arr
        self._cache_order.append(key)
        if self.cache_max_items > 0:
            while len(self._cache_order) > self.cache_max_items:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)

    def _load_image(self, sample: SampleRecord) -> Image.Image:
        key = str(sample.path)
        if self.cache_mode == "ram":
            arr = self._cache.get(key)
            if arr is not None:
                return Image.fromarray(arr, mode="RGB")
        image = load_rgb_from_sample(sample.path)
        if self.cache_mode == "ram":
            self._cache_put(key, np.asarray(image, dtype=np.uint8))
        return image

    def warmup_cache(self) -> None:
        if self.cache_mode != "ram":
            return
        total = len(self.samples)
        if total == 0:
            return
        print(f"[info] warming up RAM cache for {total} samples...")
        for i, sample in enumerate(self.samples, start=1):
            _ = self._load_image(sample)
            if i % 200 == 0 or i == total:
                print(f"[cache] {i}/{total}")
        print(f"[info] RAM cache ready: {len(self._cache)} items")

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        sample = self.samples[index]
        image = self._load_image(sample)
        return self.transform(image, sample.ann_boxes)


class DINOLikeLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits: list[torch.Tensor], teacher_logits: list[torch.Tensor]) -> torch.Tensor:
        """
        student_logits: all crops, list len = n_global + n_local
        teacher_logits: global crops only, list len = n_global
        """
        t_cat = torch.cat(teacher_logits, dim=0)
        t_probs = F.softmax((t_cat - self.center) / self.teacher_temp, dim=-1).detach()
        t_probs_list = list(t_probs.chunk(len(teacher_logits), dim=0))

        total = torch.zeros((), device=t_cat.device, dtype=t_cat.dtype)
        n_terms = 0
        for iq, q in enumerate(t_probs_list):
            for iv, v in enumerate(student_logits):
                # DINO-style: skip matching the same global view index.
                if iv == iq:
                    continue
                log_p = F.log_softmax(v / self.student_temp, dim=-1)
                total = total + torch.mean(torch.sum(-q * log_p, dim=-1))
                n_terms += 1
        if n_terms == 0:
            raise RuntimeError("No valid terms for DINO loss; check crop counts.")
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_logits: list[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_logits, dim=0).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))


@dataclass
class EmaModules:
    adapters: nn.ModuleDict
    projector: nn.Module

    @torch.no_grad()
    def update_from(self, student_adapters: nn.ModuleDict, student_projector: nn.Module, momentum: float) -> None:
        s_state = dict(student_adapters.named_parameters())
        t_state = dict(self.adapters.named_parameters())
        for name, t in t_state.items():
            s = s_state[name]
            t.data.mul_(momentum).add_(s.data * (1.0 - momentum))

        s_buf = dict(student_adapters.named_buffers())
        t_buf = dict(self.adapters.named_buffers())
        for name, t in t_buf.items():
            t.data.copy_(s_buf[name].data)

        s_state = dict(student_projector.named_parameters())
        t_state = dict(self.projector.named_parameters())
        for name, t in t_state.items():
            s = s_state[name]
            t.data.mul_(momentum).add_(s.data * (1.0 - momentum))

        s_buf = dict(student_projector.named_buffers())
        t_buf = dict(self.projector.named_buffers())
        for name, t in t_buf.items():
            t.data.copy_(s_buf[name].data)


def build_ema_modules(model: DINOv3StageAModel, device: torch.device) -> EmaModules:
    import copy

    ema_adapters: nn.ModuleDict = copy.deepcopy(model.adapters).to(device)
    ema_projector: nn.Module = copy.deepcopy(model.projector).to(device)
    for p in ema_adapters.parameters():
        p.requires_grad_(False)
    for p in ema_projector.parameters():
        p.requires_grad_(False)
    ema_adapters.eval()
    ema_projector.eval()
    return EmaModules(adapters=ema_adapters, projector=ema_projector)


def _to_device_crop_list(crops: Sequence[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [c.to(device, non_blocking=True) for c in crops]


def cosine_ema_momentum(epoch: int, epochs: int, base_m: float) -> float:
    # Gradually increase EMA momentum to 1.0.
    return 1.0 - 0.5 * (1.0 - base_m) * (1.0 + math.cos(math.pi * epoch / max(1, epochs - 1)))


def _save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: DINOv3StageAModel,
    ema: EmaModules,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    ckpt = {
        "epoch": epoch,
        "weights_dir": model.weights_dir,
        "adapter_indices": model.adapter_indices,
        "hidden_size": model.hidden_size,
        "student_adapters": model.adapters.state_dict(),
        "student_projector": model.projector.state_dict(),
        "teacher_adapters_ema": ema.adapters.state_dict(),
        "teacher_projector_ema": ema.projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    path = output_dir / f"stage_a_epoch_{epoch:03d}.pt"
    torch.save(ckpt, path)
    latest = output_dir / "stage_a_last.pt"
    torch.save(ckpt, latest)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-A self-supervised DINOv3 adapter training")
    p.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more roots. Supports image files and LabelMe JSON files.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/stage_a"))
    p.add_argument("--teacher-weights", type=str, default="", help="Local HF DINOv3 directory with config.json")
    p.add_argument("--teacher-no-pretrained", action="store_true")
    p.add_argument("--adapter-indices", type=str, default="3,4,7,8,11,12")
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--proj-hidden-dim", type=int, default=2048)
    p.add_argument("--proj-out-dim", type=int, default=1024)
    p.add_argument("--proj-dropout", type=float, default=0.0)
    p.add_argument("--num-global-crops", type=int, default=2)
    p.add_argument("--num-mid-crops", type=int, default=2)
    p.add_argument("--num-local-crops", type=int, default=4)
    p.add_argument("--elongated-ratio-threshold", type=float, default=2.5)
    p.add_argument("--include-ann-prob", type=float, default=0.85, help="Probability of sampling a crop that covers an annotation box when LabelMe shapes are available")
    p.add_argument("--max-crop-aspect", type=float, default=1.6, help="Upper bound of crop box aspect ratio (long/short) before square padding")
    p.add_argument("--global-crop-size", type=int, default=448)
    p.add_argument("--mid-crop-size", type=int, default=320)
    p.add_argument("--local-crop-size", type=int, default=160)
    p.add_argument("--global-normal-side-frac", type=str, default="0.45,0.80")
    p.add_argument("--mid-normal-side-frac", type=str, default="0.25,0.50")
    p.add_argument("--local-normal-side-frac", type=str, default="0.10,0.25")
    p.add_argument("--global-short-side-frac", type=str, default="0.70,1.00")
    p.add_argument("--global-long-side-frac", type=str, default="0.20,0.45")
    p.add_argument("--mid-short-side-frac", type=str, default="0.40,0.80")
    p.add_argument("--mid-long-side-frac", type=str, default="0.10,0.25")
    p.add_argument("--local-short-side-frac", type=str, default="0.20,0.50")
    p.add_argument("--local-long-side-frac", type=str, default="0.05,0.12")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-images", type=str, default="none", choices=["none", "ram"], help="Image loading cache mode")
    p.add_argument("--cache-max-items", type=int, default=0, help="RAM cache max samples; 0 means unlimited")
    p.add_argument("--warmup-cache", action="store_true", help="Preload all samples into RAM cache before training")
    p.add_argument("--persistent-workers", action="store_true", help="Enable persistent DataLoader workers (num_workers>0)")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0 only)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--student-temp", type=float, default=0.1)
    p.add_argument("--teacher-temp", type=float, default=0.04)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--viz-crops-dir", type=Path, default=None, help="Optional directory to save multi-crop previews")
    p.add_argument("--viz-samples", type=int, default=0, help="How many samples to preview before training")
    p.add_argument("--preview-only", action="store_true", help="Only export multi-crop previews, then exit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast+GradScaler")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clamp_num_workers_windows(args)
    set_seed(args.seed)

    roots = _resolve_roots(args.input_roots)
    samples = discover_samples(roots)
    print(f"[info] found {len(samples)} samples from {len(roots)} roots")

    adapter_indices = _parse_int_list(args.adapter_indices)
    global_spec = CropSpec(
        target_size=args.global_crop_size,
        num_crops=args.num_global_crops,
        normal_side_frac=_parse_float_pair(args.global_normal_side_frac, "global-normal-side-frac"),
        short_side_frac=_parse_float_pair(args.global_short_side_frac, "global-short-side-frac"),
        long_side_frac=_parse_float_pair(args.global_long_side_frac, "global-long-side-frac"),
    )
    mid_spec = CropSpec(
        target_size=args.mid_crop_size,
        num_crops=args.num_mid_crops,
        normal_side_frac=_parse_float_pair(args.mid_normal_side_frac, "mid-normal-side-frac"),
        short_side_frac=_parse_float_pair(args.mid_short_side_frac, "mid-short-side-frac"),
        long_side_frac=_parse_float_pair(args.mid_long_side_frac, "mid-long-side-frac"),
    )
    local_spec = CropSpec(
        target_size=args.local_crop_size,
        num_crops=args.num_local_crops,
        normal_side_frac=_parse_float_pair(args.local_normal_side_frac, "local-normal-side-frac"),
        short_side_frac=_parse_float_pair(args.local_short_side_frac, "local-short-side-frac"),
        long_side_frac=_parse_float_pair(args.local_long_side_frac, "local-long-side-frac"),
    )

    aug = MultiCropAug(
        global_spec=global_spec,
        mid_spec=mid_spec,
        local_spec=local_spec,
        elongated_ratio_threshold=args.elongated_ratio_threshold,
        include_ann_prob=args.include_ann_prob,
        max_crop_aspect=args.max_crop_aspect,
        num_global_crops=args.num_global_crops,
        num_mid_crops=args.num_mid_crops,
        num_local_crops=args.num_local_crops,
    )
    if args.viz_samples > 0:
        viz_dir = args.viz_crops_dir if args.viz_crops_dir is not None else args.output_dir / "multicrop_preview"
        save_multicrop_previews(samples, aug, viz_dir.expanduser().resolve(), args.viz_samples)
        if args.preview_only:
            print("[info] preview-only enabled, exiting before training.")
            return
    ds = UnlabeledMultiCropDataset(
        samples,
        transform=aug,
        cache_mode=args.cache_images,
        cache_max_items=args.cache_max_items,
    )
    if args.warmup_cache:
        ds.warmup_cache()
    if args.cache_images == "ram" and args.num_workers > 0:
        print(
            "[warn] cache-images=ram with num_workers>0 duplicates cache per worker process; "
            "consider num_workers=0 or cache-images=none if RAM usage is high."
        )
    device = torch.device(args.device)
    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty. Lower --batch-size or provide more samples.")

    tw = args.teacher_weights.strip() or None
    model = DINOv3StageAModel(
        weights_dir=tw,
        pretrained=not args.teacher_no_pretrained,
        device=device,
        adapter_indices=adapter_indices,
        bottleneck_dim=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        proj_dropout=args.proj_dropout,
        freeze_backbone=True,
    ).to(device)
    model.train()
    model.backbone.eval()

    ema = build_ema_modules(model, device=device)
    criterion = DINOLikeLoss(
        out_dim=args.proj_out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        center_momentum=args.center_momentum,
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output dir: {args.output_dir}")

    global_views = args.num_global_crops
    all_views = args.num_global_crops + args.num_mid_crops + args.num_local_crops
    if global_views < 1 or all_views <= global_views:
        raise ValueError("Need at least 1 global crop and at least one additional crop.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        running = 0.0
        n_steps = 0
        ema_m = cosine_ema_momentum(epoch - 1, args.epochs, args.ema_momentum)

        for crops in loader:
            # default collate stacks to list[tensor(B,C,H,W)] with len = n_views
            if not isinstance(crops, list):
                crops = list(crops)
            if len(crops) != all_views:
                raise RuntimeError(f"Unexpected number of crops: {len(crops)} != {all_views}")

            crops = _to_device_crop_list(crops, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # teacher on global crops only
                teacher_logits: list[torch.Tensor] = []
                for i in range(global_views):
                    hs = model.backbone_hidden_states(crops[i])
                    adapted = []
                    for idx in model.adapter_indices:
                        x = hs[idx]
                        x = ema.adapters[str(idx)](x)
                        adapted.append(x)
                    pooled = model.pool_cls(adapted)
                    teacher_logits.append(ema.projector(pooled))

                # student on all crops
                student_logits: list[torch.Tensor] = []
                for i in range(all_views):
                    hs = model.backbone_hidden_states(crops[i])
                    student_logits.append(model.project_from_hidden_states(hs))

                loss = criterion(student_logits=student_logits, teacher_logits=teacher_logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                criterion.update_center(teacher_logits)
                ema.update_from(model.adapters, model.projector, momentum=ema_m)

            running += float(loss.detach().cpu().item())
            n_steps += 1

        epoch_loss = running / max(1, n_steps)
        print(f"[epoch {epoch:03d}/{args.epochs}] loss={epoch_loss:.6f} ema_m={ema_m:.6f}")

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            path = _save_checkpoint(args.output_dir, epoch, model, ema, optimizer, args)
            print(f"[ckpt] saved: {path}")


if __name__ == "__main__":
    main()

