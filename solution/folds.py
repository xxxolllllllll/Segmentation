"""Deterministic K-fold splitting of the LabelMe crack dataset.

The fold manifest is the single source of truth shared by teacher training
(Stage B), student distillation (Stage C) and evaluation, so every experiment
uses the exact same train/val/test assignment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError

from scripts.labelme_crack_copy_paste import read_labelme

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".JPEG")


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


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def discover_samples(labelme_dir: Path, images_dir: Optional[Path] = None) -> list[dict[str, str]]:
    """Return all LabelMe samples (json + resolvable image) as dicts."""
    images_dir = images_dir or labelme_dir
    samples: list[dict[str, str]] = []
    for ann_path in sorted(labelme_dir.glob("*.json")):
        data = read_labelme(ann_path)
        img_path = resolve_labelme_image(images_dir, ann_path, data)
        if img_path is None or not is_valid_image(img_path):
            continue
        samples.append(
            {"stem": ann_path.stem, "json": str(ann_path), "image": str(img_path)}
        )
    return samples


def build_kfold(samples: list[dict[str, str]], k: int = 5, seed: int = 42) -> dict[str, Any]:
    """Deterministically assign samples to ``k`` folds (round-robin after shuffle)."""
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(samples) < k:
        raise ValueError(f"Not enough samples ({len(samples)}) for {k} folds")
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    folds: list[list[dict[str, str]]] = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(samples[idx])
    return {
        "k": int(k),
        "seed": int(seed),
        "num_samples": len(samples),
        "folds": folds,
    }


def save_folds(split: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")


def load_folds(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def to_tuples(samples: list[dict[str, str]]) -> list[tuple[Path, Path]]:
    return [(Path(s["json"]), Path(s["image"])) for s in samples]


def fold_train_val_test(
    split: dict[str, Any],
    held_out_fold: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Return (train, val, test) sample dicts for one held-out fold.

    ``test`` = the held-out fold. ``train``/``val`` are drawn from the remaining
    folds, with ``val`` being an internal ``val_ratio`` subset used for early
    stopping only.
    """
    folds = split["folds"]
    k = int(split["k"])
    if not (0 <= held_out_fold < k):
        raise ValueError(f"held_out_fold {held_out_fold} out of range for k={k}")

    test = list(folds[held_out_fold])
    train_pool: list[dict[str, str]] = []
    for f in range(k):
        if f != held_out_fold:
            train_pool.extend(folds[f])

    rng = random.Random(seed)
    pool = list(train_pool)
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_ratio)) if len(pool) > 1 else 0
    val = pool[:n_val]
    train = pool[n_val:] if n_val > 0 else pool
    return train, val, test
