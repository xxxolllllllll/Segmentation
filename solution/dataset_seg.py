from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _iter_image_stems(images_dir: Path) -> list[str]:
    stems: set[str] = set()
    for suffix in IMAGE_SUFFIXES:
        for path in images_dir.glob(f"*{suffix}"):
            stems.add(path.stem)
        for path in images_dir.glob(f"*{suffix.upper()}"):
            stems.add(path.stem)
    return sorted(stems)


def split_stems(images_dir: Path, val_ratio: float = 0.05, seed: int = 42) -> tuple[list[str], list[str]]:
    stems = _iter_image_stems(Path(images_dir))
    if not stems:
        return [], []
    rng = random.Random(int(seed))
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * float(val_ratio))) if len(stems) > 1 and val_ratio > 0 else 0
    val = stems[:n_val]
    train = stems[n_val:] or stems
    return train, val


class PairRandomAugment:
    """Lightweight paired augmentations for image/mask training patches."""

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.3:
            k = random.randint(1, 3)
            img = img.transpose((Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[k - 1])
            mask = mask.transpose((Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[k - 1])
        if random.random() < 0.2:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.2:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        return img, mask


def default_train_transforms() -> Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]:
    return PairRandomAugment()


def _resolve_image_path(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        cand = images_dir / f"{stem}{suffix}"
        if cand.is_file():
            return cand
        cand_upper = images_dir / f"{stem}{suffix.upper()}"
        if cand_upper.is_file():
            return cand_upper
    raise FileNotFoundError(f"Missing image for stem '{stem}' under {images_dir}")


def _normalize_mask(mask_np: np.ndarray, num_classes: int) -> np.ndarray:
    out = mask_np.astype(np.int64, copy=False)
    if num_classes <= 2:
        keep = (out == 0) | (out == 1) | (out == 255)
        out = np.where(keep, out, np.where(out == 255, 255, 1))
        return out.astype(np.int64, copy=False)
    invalid = (out != 255) & ((out < 0) | (out >= int(num_classes)))
    if np.any(invalid):
        out = out.copy()
        out[invalid] = 0
    return out.astype(np.int64, copy=False)


class SegmentationPatchDataset(Dataset):
    def __init__(
        self,
        *,
        images_dir: Path,
        masks_dir: Path,
        image_list: Iterable[str],
        transform=None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_list = [str(stem) for stem in image_list]
        self.transform = transform
        self.num_classes = int(num_classes)
        self.is_positive_mask: list[bool] = []
        for stem in self.image_list:
            mask_path = self.masks_dir / f"{stem}.png"
            with Image.open(mask_path) as mask_im:
                mask_np = _normalize_mask(np.array(mask_im, dtype=np.int64), self.num_classes)
            self.is_positive_mask.append(bool(np.any(mask_np == 1)))

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem = self.image_list[index]
        image_path = _resolve_image_path(self.images_dir, stem)
        mask_path = self.masks_dir / f"{stem}.png"
        with Image.open(image_path) as img_im:
            img = img_im.convert("RGB")
        with Image.open(mask_path) as mask_im:
            mask = mask_im.convert("L")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        mask_np = _normalize_mask(np.array(mask, dtype=np.int64), self.num_classes)
        return {
            "image": torch.from_numpy(np.transpose(img_np, (2, 0, 1))),
            "mask": torch.from_numpy(mask_np),
            "stem": stem,
        }
