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
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checkpoint_io import torch_load_compat
from models.dino_stage_a import DINOv3StageAModel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dist_info() -> tuple[int, int, int, bool]:
    """(rank, world_size, local_rank, is_dist) from the torchrun environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank, world_size > 1


def _setup_distributed() -> tuple[int, int, int, bool]:
    rank, world_size, local_rank, is_dist = _dist_info()
    if is_dist:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, is_dist


def _broadcast_module_(module: nn.Module, src: int = 0) -> None:
    for p in module.parameters():
        torch.distributed.broadcast(p.data, src=src)
    for b in module.buffers():
        torch.distributed.broadcast(b.data, src=src)


def _all_reduce_grads_(params, world_size: int) -> None:
    if world_size <= 1:
        return
    for p in params:
        if p.grad is None:
            continue
        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
        p.grad.div_(world_size)


def _broadcast_tensor_(t: torch.Tensor, src: int = 0) -> None:
    torch.distributed.broadcast(t, src=src)


def _all_reduce_mean_(t: torch.Tensor) -> None:
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    t.div_(torch.distributed.get_world_size())


def clamp_num_workers_windows(args: argparse.Namespace) -> None:
    if os.name == "nt" and args.num_workers != 0:
        print(f"[info] Windows detected, forcing num_workers from {args.num_workers} to 0")
        args.num_workers = 0


def worker_init_fn(worker_id: int) -> None:
    # Forked DataLoader workers inherit the CUDA-context memory mappings of the
    # main process. Pin each worker to a single thread so OpenMP/MKL/blas don't
    # oversubscribe CPUs and inflate per-worker memory further.
    torch.set_num_threads(1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


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


def _pil_pad_to_square(img: Image.Image, target_size: int) -> Image.Image:
    """Pad a (native-resolution) crop to a fixed square with no scaling."""
    w, h = img.size
    if w > target_size or h > target_size:
        img = img.crop((0, 0, min(w, target_size), min(h, target_size)))
        w, h = img.size
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(img, ((target_size - w) // 2, (target_size - h) // 2))
    return canvas


def _pil_resize_with_aspect_and_pad(img: Image.Image, target_size: int) -> Image.Image:
    """PIL version of aspect-preserving resize+pad to a fixed square.

    Doing the resize in PIL (rather than resizing an arbitrary-size torch tensor)
    keeps DataLoader workers from creating/fragmenting many variable-size CPU
    tensors, which otherwise makes worker RSS grow without bound.
    """
    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    scale = min(target_size / h, target_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


def _sample_crop_box(
    width: int,
    height: int,
    *,
    spec: CropSpec,
    elongated_ratio_threshold: float,
    ann_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    include_ann_prob: float = 0.8,
    max_crop_aspect: float = 1.6,
    fixed_size: int | None = None,
) -> tuple[int, int, int, int]:
    if fixed_size is not None:
        # Fixed native square crop (no scaling): side = min(target, image dims).
        side = max(16, min(int(fixed_size), width, height))
        crop_w = crop_h = side
    else:
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


class CropTransform:
    """Picklable single-crop transform.

    Implemented as a module-level class (instead of a closure) so it can be
    pickled when the DataLoader uses the ``spawn`` multiprocessing context,
    which avoids fork-inheriting the parent CUDA context in each worker.
    """

    def __init__(
        self,
        *,
        spec: CropSpec,
        elongated_ratio_threshold: float,
        color_jitter_strength: float = 0.4,
        include_ann_prob: float = 0.8,
        max_crop_aspect: float = 1.6,
        resize: bool = True,
    ) -> None:
        from torchvision import transforms as T

        self.spec = spec
        self.elongated_ratio_threshold = float(elongated_ratio_threshold)
        self.include_ann_prob = float(include_ann_prob)
        self.max_crop_aspect = float(max_crop_aspect)
        self.resize = bool(resize)
        self.cj = T.ColorJitter(
            brightness=0.8 * color_jitter_strength,
            contrast=0.8 * color_jitter_strength,
            saturation=0.8 * color_jitter_strength,
            hue=0.2 * color_jitter_strength,
        )
        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        self.gray = T.Grayscale(num_output_channels=3)

    def __call__(
        self,
        image: Image.Image,
        ann_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> torch.Tensor:
        from torchvision.transforms import functional as TF

        crop_box = _sample_crop_box(
            image.width,
            image.height,
            spec=self.spec,
            elongated_ratio_threshold=self.elongated_ratio_threshold,
            ann_boxes=ann_boxes,
            include_ann_prob=self.include_ann_prob,
            max_crop_aspect=self.max_crop_aspect,
            fixed_size=self.spec.target_size if not self.resize else None,
        )
        crop = image.crop(crop_box)
        if random.random() < 0.5:
            crop = TF.hflip(crop)
        if random.random() < 0.8:
            crop = self.cj(crop)
        if random.random() < 0.2:
            crop = self.gray(crop)
        if random.random() < 0.3:
            crop = self.blur(crop)

        if self.resize:
            crop = _pil_resize_with_aspect_and_pad(crop, self.spec.target_size)
        else:
            # Native-resolution crop: only pad to square, never rescale.
            crop = _pil_pad_to_square(crop, self.spec.target_size)
        x = TF.to_tensor(crop)
        x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return x


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
        resize: bool = True,
    ) -> None:
        self.num_global_crops = num_global_crops
        self.num_mid_crops = num_mid_crops
        self.num_local_crops = num_local_crops
        self.global_tf = CropTransform(
            spec=global_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
            resize=resize,
        )
        self.mid_tf = CropTransform(
            spec=mid_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
            resize=resize,
        )
        self.local_tf = CropTransform(
            spec=local_spec,
            elongated_ratio_threshold=elongated_ratio_threshold,
            include_ann_prob=include_ann_prob,
            max_crop_aspect=max_crop_aspect,
            resize=resize,
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
    def update_center(self, teacher_logits: list[torch.Tensor], reduce: bool = False) -> None:
        batch_center = torch.cat(teacher_logits, dim=0).mean(dim=0, keepdim=True)
        if reduce:
            _all_reduce_mean_(batch_center)
        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))


def blockwise_mask(gh: int, gw: int, mask_ratio: float, min_num_patches: int, max_num_patches: int) -> torch.Tensor:
    """BEiT-style block-wise mask. Returns a flattened bool mask [gh*gw], True = masked."""
    n_total = gh * gw
    num_masking = max(int(min_num_patches), int(round(n_total * mask_ratio)))
    num_masking = min(num_masking, n_total)
    mask = torch.zeros(gh, gw, dtype=torch.bool)
    if num_masking <= 0:
        return mask.flatten()
    max_num_patches = max(int(min_num_patches), min(int(max_num_patches), num_masking))
    count = 0
    while count < num_masking:
        max_mask = min(max_num_patches, num_masking - count)
        max_mask = max(max_mask, int(min_num_patches))
        mh = min(random.randint(int(min_num_patches), max_mask), gh)
        mw = min(random.randint(int(min_num_patches), max_mask), gw)
        top = random.randint(0, gh - mh)
        left = random.randint(0, gw - mw)
        cur = mask[top : top + mh, left : left + mw]
        n_unmasked = int((~cur).sum())
        if n_unmasked == 0:
            continue
        if n_unmasked > max_mask:
            flat = cur.flatten()
            idx = (~flat).nonzero(as_tuple=False).flatten()
            sel = idx[torch.randperm(idx.numel())[:max_mask]]
            flat[sel] = True
            mask[top : top + mh, left : left + mw] = flat.reshape(mh, mw)
            count += max_mask
        else:
            mask[top : top + mh, left : left + mw] = True
            count += n_unmasked
    return mask.flatten()


def build_block_masks(
    batch: int,
    crop_size: int,
    patch_size: int,
    mask_ratio: float,
    min_num_patches: int,
    max_num_patches: int,
    device: torch.device,
) -> torch.Tensor:
    gh = gw = crop_size // patch_size
    masks = [blockwise_mask(gh, gw, mask_ratio, min_num_patches, max_num_patches) for _ in range(batch)]
    return torch.stack(masks, dim=0).to(device)


class IBOTLoss(nn.Module):
    """Dense (iBOT) self-distillation over masked patch tokens, per scale group (P3/P4/P5)."""

    def __init__(
        self,
        out_dim: int,
        num_groups: int = 3,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.num_groups = int(num_groups)
        self.register_buffer("centers", torch.zeros(self.num_groups, out_dim))

    def forward(
        self,
        student_patch_logits: list[list[torch.Tensor]],
        teacher_patch_logits: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        layer_groups: Sequence[int],
    ) -> torch.Tensor:
        """Each ``*_patch_logits[v]`` is a list over adapter layers with shape [B, P, K]."""
        total = torch.zeros((), device=student_patch_logits[0][0].device, dtype=torch.float32)
        n_terms = 0
        for s_layers, t_layers, m in zip(student_patch_logits, teacher_patch_logits, masks):
            if int(m.sum()) == 0:
                continue
            for i, (s, t) in enumerate(zip(s_layers, t_layers)):
                g = int(layer_groups[i])
                s_m = s[m].float()
                t_m = t[m].detach().float()
                target = F.softmax((t_m - self.centers[g].unsqueeze(0)) / self.teacher_temp, dim=-1)
                logp = F.log_softmax(s_m / self.student_temp, dim=-1)
                total = total + (-target * logp).sum(-1).mean()
                n_terms += 1
        if n_terms == 0:
            return student_patch_logits[0][0].sum() * 0.0
        return total / n_terms

    @torch.no_grad()
    def update_center(
        self,
        teacher_patch_logits: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        layer_groups: Sequence[int],
        reduce: bool = False,
    ) -> None:
        vals: dict[int, list[torch.Tensor]] = defaultdict(list)
        for t_layers, m in zip(teacher_patch_logits, masks):
            if int(m.sum()) == 0:
                continue
            for i, t in enumerate(t_layers):
                vals[int(layer_groups[i])].append(t[m].detach().float())
        if not reduce:
            for g, chunks in vals.items():
                if not chunks:
                    continue
                batch_center = torch.cat(chunks, dim=0).mean(dim=0)
                self.centers[g].mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))
            return
        # Distributed: fixed collective count across ranks (avoid deadlock when a
        # rank has no masked tokens for some group).
        for g in range(self.num_groups):
            chunks = vals.get(g, [])
            batch_center = torch.cat(chunks, dim=0).mean(dim=0) if chunks else torch.zeros_like(self.centers[g])
            count = torch.tensor([1.0 if chunks else 0.0], device=self.centers.device)
            _all_reduce_mean_(batch_center)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            if float(count.item()) > 0.0:
                self.centers[g].mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))

    @torch.no_grad()
    def collapse_stats(
        self,
        student_patch_logits: list[list[torch.Tensor]],
        teacher_patch_logits: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        layer_groups: Sequence[int],
    ) -> dict[str, dict[str, float]]:
        """Per-group collapse indicators computed on the masked patch tokens."""
        t_by_g: dict[int, list[torch.Tensor]] = defaultdict(list)
        s_by_g: dict[int, list[torch.Tensor]] = defaultdict(list)
        for s_layers, t_layers, m in zip(student_patch_logits, teacher_patch_logits, masks):
            if int(m.sum()) == 0:
                continue
            for i, (s, t) in enumerate(zip(s_layers, t_layers)):
                g = int(layer_groups[i])
                t_by_g[g].append(t[m].detach().float())
                s_by_g[g].append(s[m].detach().float())
        stats: dict[str, dict[str, float]] = {}
        for g in sorted(t_by_g):
            t_all = torch.cat(t_by_g[g], dim=0)
            s_all = torch.cat(s_by_g[g], dim=0)
            target = F.softmax((t_all - self.centers[g].unsqueeze(0)) / self.teacher_temp, dim=-1)
            marg = target.mean(dim=0)
            marg_entropy = float(-(marg * (marg + 1e-8).log()).sum() / math.log(marg.numel()))
            stats[str(g)] = {
                "n_tokens": float(t_all.shape[0]),
                "marg_entropy": marg_entropy,
                "mean_max_prob": float(target.max(dim=-1).values.mean()),
                "student_logit_std": float(s_all.std(dim=0).mean()),
                "center_norm": float(self.centers[g].norm()),
            }
        return stats


@torch.no_grad()
def _ema_update_(target: nn.Module, source: nn.Module, momentum: float) -> None:
    t_params = dict(target.named_parameters())
    s_params = dict(source.named_parameters())
    for name, t in t_params.items():
        t.data.mul_(momentum).add_(s_params[name].data * (1.0 - momentum))
    t_buf = dict(target.named_buffers())
    s_buf = dict(source.named_buffers())
    for name, t in t_buf.items():
        t.data.copy_(s_buf[name].data)


@dataclass
class EmaModules:
    adapters: nn.ModuleDict
    projector: nn.Module
    ibot_heads: nn.ModuleDict

    @torch.no_grad()
    def update_from(
        self,
        student_adapters: nn.ModuleDict,
        student_projector: nn.Module,
        student_ibot_heads: nn.ModuleDict,
        momentum: float,
    ) -> None:
        _ema_update_(self.adapters, student_adapters, momentum)
        _ema_update_(self.projector, student_projector, momentum)
        for key in self.ibot_heads:
            _ema_update_(self.ibot_heads[key], student_ibot_heads[key], momentum)


def build_ema_modules(model: DINOv3StageAModel, device: torch.device) -> EmaModules:
    import copy

    ema_adapters: nn.ModuleDict = copy.deepcopy(model.adapters).to(device)
    ema_projector: nn.Module = copy.deepcopy(model.projector).to(device)
    ema_ibot_heads: nn.ModuleDict = copy.deepcopy(model.ibot_heads).to(device)
    for module in (ema_adapters, ema_projector, ema_ibot_heads):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
    return EmaModules(adapters=ema_adapters, projector=ema_projector, ibot_heads=ema_ibot_heads)


def _to_device_crop_list(crops: Sequence[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [c.to(device, non_blocking=True) for c in crops]


def cosine_ema_momentum(epoch: int, epochs: int, base_m: float) -> float:
    # Gradually increase EMA momentum to 1.0.
    return 1.0 - 0.5 * (1.0 - base_m) * (1.0 + math.cos(math.pi * epoch / max(1, epochs - 1)))


LOSS_STEP_FIELDS = ["global_step", "epoch", "loss"]
LOSS_EPOCH_FIELDS = ["epoch", "loss", "ema_m"]


def _prepare_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)


def _append_csv_row(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: DINOv3StageAModel,
    ema: EmaModules,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dino_loss: DINOLikeLoss,
    ibot_loss: IBOTLoss,
    scaler: torch.amp.GradScaler | None = None,
) -> Path:
    ckpt = {
        "epoch": epoch,
        "weights_dir": model.weights_dir,
        "adapter_indices": model.adapter_indices,
        "hidden_size": model.hidden_size,
        "student_adapters": model.adapters.state_dict(),
        "student_projector": model.projector.state_dict(),
        "student_ibot_head": model.ibot_heads.state_dict(),
        "teacher_adapters_ema": ema.adapters.state_dict(),
        "teacher_projector_ema": ema.projector.state_dict(),
        "teacher_ibot_head_ema": ema.ibot_heads.state_dict(),
        "dino_center": dino_loss.center.detach().cpu(),
        "ibot_centers": ibot_loss.centers.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
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
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from a Stage-A checkpoint path, or 'auto' for <output-dir>/stage_a_last.pt",
    )
    p.add_argument("--teacher-weights", type=str, default="", help="Local HF DINOv3 directory with config.json")
    p.add_argument("--teacher-no-pretrained", action="store_true")
    p.add_argument("--adapter-indices", type=str, default="3,4,7,8,11,12")
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--proj-hidden-dim", type=int, default=2048)
    p.add_argument("--proj-out-dim", type=int, default=1024)
    p.add_argument("--proj-dropout", type=float, default=0.0)
    p.add_argument("--ibot-hidden-dim", type=int, default=2048)
    p.add_argument("--ibot-out-dim", type=int, default=2048)
    p.add_argument("--ibot-dropout", type=float, default=0.0)
    p.add_argument("--lambda-dino", type=float, default=1.0, help="Weight for the CLS-level DINO loss")
    p.add_argument("--lambda-ibot", type=float, default=1.0, help="Weight for the dense (iBOT) masked-patch loss")
    p.add_argument("--ibot-student-temp", type=float, default=0.1)
    p.add_argument("--ibot-teacher-temp", type=float, default=0.04)
    p.add_argument("--ibot-center-momentum", type=float, default=0.9)
    p.add_argument("--mask-ratio", type=float, default=0.3, help="Block-wise mask ratio for student global crops (iBOT)")
    p.add_argument("--mask-min-num-patches", type=int, default=8)
    p.add_argument("--mask-max-num-patches", type=int, default=0, help="0 = num_masking_patches")
    p.add_argument(
        "--ibot-diag-every",
        type=int,
        default=0,
        help="If >0, print per-scale iBOT collapse diagnostics every N steps",
    )
    p.add_argument("--num-global-crops", type=int, default=2)
    p.add_argument("--num-mid-crops", type=int, default=2)
    p.add_argument("--num-local-crops", type=int, default=4)
    p.add_argument("--elongated-ratio-threshold", type=float, default=2.5)
    p.add_argument("--include-ann-prob", type=float, default=0.85, help="Probability of sampling a crop that covers an annotation box when LabelMe shapes are available")
    p.add_argument("--max-crop-aspect", type=float, default=1.6, help="Upper bound of crop box aspect ratio (long/short) before square padding")
    p.add_argument("--global-crop-size", type=int, default=1024)
    p.add_argument("--mid-crop-size", type=int, default=640)
    p.add_argument("--local-crop-size", type=int, default=320)
    p.add_argument(
        "--crop-resize",
        action="store_true",
        help="Resize each crop to its target size (old behavior). Default: native-resolution crop + pad only (no scaling).",
    )
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
    p.add_argument("--log-every", type=int, default=100, help="Print per-step loss every N steps (0 disables step-level printing)")
    p.add_argument("--max-steps", type=int, default=0, help="Debug: limit steps per epoch (0 = all)")
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
    rank, world_size, local_rank, is_dist = _setup_distributed()
    is_main = rank == 0

    roots = _resolve_roots(args.input_roots)
    samples = discover_samples(roots)
    if is_main:
        print(f"[info] found {len(samples)} samples from {len(roots)} roots | world_size={world_size}", flush=True)

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
        resize=bool(args.crop_resize),
    )
    if args.viz_samples > 0:
        viz_dir = args.viz_crops_dir if args.viz_crops_dir is not None else args.output_dir / "multicrop_preview"
        if is_main:
            save_multicrop_previews(samples, aug, viz_dir.expanduser().resolve(), args.viz_samples)
        if args.preview_only:
            if is_main:
                print("[info] preview-only enabled, exiting before training.")
            if is_dist:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
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
    device = torch.device("cuda", local_rank) if is_dist else torch.device(args.device)
    sampler = None
    if is_dist:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True
        )
    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["worker_init_fn"] = worker_init_fn
        # spawn (not fork) so workers do NOT inherit the CUDA context of the
        # main process. fork-inherited CUDA mappings cost ~3GB RAM per worker
        # and cause OOM on WSL. Requires picklable dataset/transform.
        loader_kwargs["multiprocessing_context"] = "spawn"
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
        ibot_hidden_dim=args.ibot_hidden_dim,
        ibot_out_dim=args.ibot_out_dim,
        ibot_dropout=args.ibot_dropout,
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
    ibot_criterion = IBOTLoss(
        out_dim=args.ibot_out_dim,
        num_groups=model.num_ibot_groups,
        student_temp=args.ibot_student_temp,
        teacher_temp=args.ibot_teacher_temp,
        center_momentum=args.ibot_center_momentum,
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (not args.no_amp) and device.type == "cuda"
    # bf16 autocast (no GradScaler): keeps ranks consistent under DDP grad averaging.
    scaler = None
    if is_dist:
        for m in (model.adapters, model.projector, model.ibot_heads, ema.adapters, ema.projector, ema.ibot_heads):
            _broadcast_module_(m)
        _broadcast_tensor_(criterion.center)
        _broadcast_tensor_(ibot_criterion.centers)

    args.output_dir = args.output_dir.expanduser().resolve()
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        torch.distributed.barrier()
    if is_main:
        print(f"[info] output dir: {args.output_dir}")

    loss_steps_csv = args.output_dir / "loss_steps.csv"
    loss_epoch_csv = args.output_dir / "epoch_metrics.csv"
    if is_main:
        _prepare_csv(loss_steps_csv, LOSS_STEP_FIELDS)
        _prepare_csv(loss_epoch_csv, LOSS_EPOCH_FIELDS)
    if is_dist:
        torch.distributed.barrier()

    # ---- resume ----
    start_epoch = 1
    resume_path: Path | None = None
    if args.resume.strip():
        resume_path = (
            args.output_dir / "stage_a_last.pt"
            if args.resume.strip() == "auto"
            else Path(args.resume).expanduser().resolve()
        )
    if resume_path is not None and resume_path.is_file():
        ckpt = torch_load_compat(resume_path, map_location="cpu", weights_only=False)
        if "student_adapters" in ckpt and "student_ibot_head" in ckpt:
            model.adapters.load_state_dict(ckpt["student_adapters"], strict=True)
            model.projector.load_state_dict(ckpt["student_projector"], strict=True)
            model.ibot_heads.load_state_dict(ckpt["student_ibot_head"], strict=True)
            if ckpt.get("teacher_adapters_ema") is not None:
                ema.adapters.load_state_dict(ckpt["teacher_adapters_ema"], strict=True)
            if ckpt.get("teacher_projector_ema") is not None:
                ema.projector.load_state_dict(ckpt["teacher_projector_ema"], strict=True)
            if ckpt.get("teacher_ibot_head_ema") is not None:
                ema.ibot_heads.load_state_dict(ckpt["teacher_ibot_head_ema"], strict=True)
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            if ckpt.get("dino_center") is not None:
                criterion.center.copy_(ckpt["dino_center"].to(criterion.center.device))
            if ckpt.get("ibot_centers") is not None:
                ibot_criterion.centers.copy_(ckpt["ibot_centers"].to(ibot_criterion.centers.device))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[resume] {resume_path} -> start_epoch={start_epoch}", flush=True)
        else:
            print(f"[resume] incompatible checkpoint (pre-iBOT?), starting fresh: {resume_path}", flush=True)
    elif resume_path is not None:
        print(f"[resume] checkpoint not found, starting fresh: {resume_path}", flush=True)

    if start_epoch > args.epochs:
        if is_main:
            print(f"[resume] start_epoch={start_epoch} > epochs={args.epochs}; nothing to do", flush=True)
            (args.output_dir / ".completed").write_text(
                json.dumps({"last_epoch": args.epochs, "target_epochs": args.epochs}), encoding="utf-8"
            )
        if is_dist:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    global_views = args.num_global_crops
    all_views = args.num_global_crops + args.num_mid_crops + args.num_local_crops
    if global_views < 1 or all_views <= global_views:
        raise ValueError("Need at least 1 global crop and at least one additional crop.")

    global_step = 0
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        model.train()
        model.backbone.eval()
        running = 0.0
        n_steps = 0
        ema_m = cosine_ema_momentum(epoch - 1, args.epochs, args.ema_momentum)
        if sampler is not None:
            sampler.set_epoch(epoch)

        for crops in loader:
            if args.max_steps > 0 and n_steps >= args.max_steps:
                break
            # default collate stacks to list[tensor(B,C,H,W)] with len = n_views
            if not isinstance(crops, list):
                crops = list(crops)
            if len(crops) != all_views:
                raise RuntimeError(f"Unexpected number of crops: {len(crops)} != {all_views}")

            crops = _to_device_crop_list(crops, device)

            # Block-wise masks for the student global crops (iBOT dense loss).
            masks: list[torch.Tensor] = []
            for i in range(global_views):
                crop_size = int(crops[i].shape[-1])
                n_patch = (crop_size // model.patch_size) ** 2
                masks.append(
                    build_block_masks(
                        batch=int(crops[i].shape[0]),
                        crop_size=crop_size,
                        patch_size=model.patch_size,
                        mask_ratio=args.mask_ratio,
                        min_num_patches=args.mask_min_num_patches,
                        max_num_patches=args.mask_max_num_patches or int(round(n_patch * args.mask_ratio)),
                        device=device,
                    )
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Teacher: unmasked global crops -> CLS logits + dense patch logits (EMA).
                teacher_logits: list[torch.Tensor] = []
                teacher_patch_logits: list[list[torch.Tensor]] = []
                with torch.no_grad():
                    for i in range(global_views):
                        hh, ww = int(crops[i].shape[-2]), int(crops[i].shape[-1])
                        hs = model.backbone_hidden_states(crops[i])
                        adapted = [ema.adapters[str(idx)](hs[idx]) for idx in model.adapter_indices]
                        teacher_logits.append(ema.projector(model.pool_cls(adapted)))
                        teacher_patch_logits.append(
                            [
                                ema.ibot_heads[str(model.layer_groups[i])](model.tokens_to_patch_seq(tok, hh, ww))
                                for i, tok in enumerate(adapted)
                            ]
                        )

                # Student: masked global crops (dense + CLS) and unmasked other crops (CLS).
                student_logits: list[torch.Tensor] = []
                student_patch_logits: list[list[torch.Tensor]] = []
                for i in range(all_views):
                    if i < global_views:
                        cls_logits, patch_logits = model.forward_dense(
                            crops[i], bool_masked_pos=masks[i], return_patch=True
                        )
                        student_logits.append(cls_logits)
                        student_patch_logits.append(patch_logits)  # type: ignore[arg-type]
                    else:
                        cls_logits, _ = model.forward_dense(crops[i], bool_masked_pos=None, return_patch=False)
                        student_logits.append(cls_logits)

                dino_loss = criterion(student_logits=student_logits, teacher_logits=teacher_logits)
                ibot_loss = ibot_criterion(student_patch_logits, teacher_patch_logits, masks, model.layer_groups)
                loss = args.lambda_dino * dino_loss + args.lambda_ibot * ibot_loss

            loss.backward()
            _all_reduce_grads_(model.trainable_parameters(), world_size)
            optimizer.step()

            with torch.no_grad():
                criterion.update_center(teacher_logits, reduce=is_dist)
                ibot_criterion.update_center(teacher_patch_logits, masks, model.layer_groups, reduce=is_dist)
                ema.update_from(model.adapters, model.projector, model.ibot_heads, momentum=ema_m)

            if is_main and args.ibot_diag_every > 0 and (global_step % args.ibot_diag_every == 0):
                with torch.no_grad():
                    diag = ibot_criterion.collapse_stats(
                        student_patch_logits, teacher_patch_logits, masks, model.layer_groups
                    )
                parts = " | ".join(
                    f"P{g} margH={s['marg_entropy']:.3f} maxp={s['mean_max_prob']:.3f} "
                    f"logitStd={s['student_logit_std']:.3f} cNorm={s['center_norm']:.2f} n={int(s['n_tokens'])}"
                    for g, s in diag.items()
                )
                print(f"[ibot-diag] step={global_step} {parts}", flush=True)


            step_loss = float(loss.detach().cpu().item())
            running += step_loss
            n_steps += 1
            global_step += 1
            if is_main:
                _append_csv_row(loss_steps_csv, [global_step, epoch, f"{step_loss:.6f}"])
            if is_main and args.log_every > 0 and n_steps % args.log_every == 0:
                print(
                    f"[epoch {epoch:03d}/{args.epochs}] step {n_steps}/{len(loader)} "
                    f"loss={running / n_steps:.6f} ema_m={ema_m:.6f}",
                    flush=True,
                )

        epoch_loss = running / max(1, n_steps)
        if is_main:
            print(f"[epoch {epoch:03d}/{args.epochs}] loss={epoch_loss:.6f} ema_m={ema_m:.6f}")
            _append_csv_row(loss_epoch_csv, [epoch, f"{epoch_loss:.6f}", f"{ema_m:.6f}"])

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            if is_dist:
                torch.distributed.barrier()
            if is_main:
                path = _save_checkpoint(args.output_dir, epoch, model, ema, optimizer, args, criterion, ibot_criterion, scaler)
                print(f"[ckpt] saved: {path}")
            if is_dist:
                torch.distributed.barrier()

    if last_epoch >= args.epochs:
        if is_main:
            (args.output_dir / ".completed").write_text(
                json.dumps({"last_epoch": last_epoch, "target_epochs": args.epochs}, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[done] Stage-A complete at epoch {last_epoch}", flush=True)
    if is_dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

