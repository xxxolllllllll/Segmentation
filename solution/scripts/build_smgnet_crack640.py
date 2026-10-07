#!/usr/bin/env python3
"""Build a paper-style (640x480) crack dataset for SMG-Net reproduction.

Paper (PLOS One 2025, journal.pone.0336125): the ancient-wooden-component crack
dataset was built by cropping crack-focused regions and uniformly resizing them
to 640x480.  This script mirrors that: sliding 640x480 windows (stride
320x240), keep windows containing cracks, cap a few crops per image, and write
``{split}/images/*.jpg`` + ``{split}/masks/*.png`` (binary 0/255 crack) at
exactly 640x480.

Fold split follows ``runs/kfold/folds.json`` (fold N val = folds[N]; train =
the rest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labelme_crack_copy_paste import rasterize_masks, read_labelme  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=Path, default=Path("runs/kfold/folds.json"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--crop-w", type=int, default=640)
    p.add_argument("--crop-h", type=int, default=480)
    p.add_argument("--stride-x", type=int, default=320)
    p.add_argument("--stride-y", type=int, default=240)
    p.add_argument("--min-crack-frac", type=float, default=0.002, help="keep windows with crack fraction >= this")
    p.add_argument("--max-per-image", type=int, default=3, help="cap crops per image (0 = no cap)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--limit-images", type=int, default=0, help="debug: limit number of images")
    return p.parse_args()


def starts(dim: int, win: int, stride: int) -> list[int]:
    if dim <= win:
        return [0]
    xs = list(range(0, dim - win + 1, stride))
    if xs[-1] != dim - win:
        xs.append(dim - win)
    return xs


def main() -> None:
    args = parse_args()
    crack_labels = {s.strip().lower() for s in args.crack_labels.split(",") if s.strip()}
    ignore_labels = {s.strip().lower() for s in args.ignore_labels.split(",") if s.strip()}
    component_labels = {s.strip().lower() for s in args.component_labels.split(",") if s.strip()}

    data = json.loads(args.folds.read_text(encoding="utf-8"))
    folds = data["folds"]
    val_entries = folds[int(args.fold)]
    val_keys = {e["json"] for e in val_entries}
    all_entries: dict[str, dict] = {}
    for fold in folds:
        for e in fold:
            all_entries[e["json"]] = e
    train_entries = [e for k, e in all_entries.items() if k not in val_keys]
    val_entries = sorted(val_entries, key=lambda e: e["stem"])
    train_entries = sorted(train_entries, key=lambda e: e["stem"])
    print(f"[build] fold={args.fold} train_images={len(train_entries)} val_images={len(val_entries)}", flush=True)

    out = args.output_dir
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    stats = {}
    for split, entries in (("train", train_entries), ("val", val_entries)):
        kept = 0
        skipped = 0
        srcs = entries[: args.limit_images] if args.limit_images > 0 else entries
        for e in srcs:
            ann_path = Path(e["json"])
            img_path = Path(e["image"]) if e.get("image") else None
            if img_path is None or not img_path.is_file():
                skipped += 1
                continue
            data_json = read_labelme(ann_path)
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            w, h = image.size
            crack, ignore, component, other = rasterize_masks(
                data_json, (w, h), crack_labels, ignore_labels, component_labels,
                treat_empty_component_as_full_image=False,
            )
            if component.any():
                crack = crack & component
            if not crack.any():
                skipped += 1
                continue
            image_np = np.asarray(image, dtype=np.uint8)
            cand: list[tuple[float, int, int]] = []
            for y0 in starts(h, args.crop_h, args.stride_y):
                for x0 in starts(w, args.crop_w, args.stride_x):
                    patch = crack[y0:y0 + args.crop_h, x0:x0 + args.crop_w]
                    frac = float(patch.mean()) if patch.size else 0.0
                    if frac >= args.min_crack_frac:
                        cand.append((frac, y0, x0))
            if not cand:
                skipped += 1
                continue
            cand.sort(key=lambda t: (-t[0], t[1], t[2]))
            if args.max_per_image > 0:
                cand = cand[: args.max_per_image]
            for k, (frac, y0, x0) in enumerate(cand):
                crop = image_np[y0:y0 + args.crop_h, x0:x0 + args.crop_w]
                mcrop = (crack[y0:y0 + args.crop_h, x0:x0 + args.crop_w].astype(np.uint8) * 255)
                if crop.shape[0] != args.crop_h or crop.shape[1] != args.crop_w:
                    ph = args.crop_h - crop.shape[0]
                    pw = args.crop_w - crop.shape[1]
                    crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode="edge")
                    mcrop = np.pad(mcrop, ((0, ph), (0, pw)), mode="constant")
                stem = f"{e['stem']}_y{y0}_x{x0}_{k}"
                Image.fromarray(crop).save(out / split / "images" / f"{stem}.jpg",
                                           quality=args.jpeg_quality, subsampling=0)
                Image.fromarray(mcrop, mode="L").save(out / split / "masks" / f"{stem}.png")
                kept += 1
        stats[split] = {"images": len(srcs), "crops": kept, "skipped": skipped}
        print(f"[build] {split}: source={stats[split]['images']} crops={kept} skipped={skipped}", flush=True)

    meta = {
        "source": "labelme ancient-wood crack; paper-style 640x480 crops",
        "fold": int(args.fold),
        "crop_w": args.crop_w, "crop_h": args.crop_h,
        "stride_x": args.stride_x, "stride_y": args.stride_y,
        "min_crack_frac": args.min_crack_frac, "max_per_image": args.max_per_image,
        "stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
