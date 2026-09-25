# -*- coding: utf-8 -*-
"""Visualize crack segmentation predictions of multiple models side by side.

Reuses the eval script's loaders/predictors (Stage-B teachers and students),
runs sliding-window inference on selected images, and saves per-image panels
with (a) a mask overlay and (b) a TP/FP/FN error map when GT is available.

Supports two data sources:
  * LabelMe dataset (has GT):  --labelme-dir [--fold-split --fold | --stems | --num-images]
  * plain image dir (no GT):   --image-roots ...            -> mask overlay only

Models can be given explicitly (--teacher NAME=PATH / --student NAME=PATH) and/or
auto-discovered under --runs-root (fold dir) with --auto-discover.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_labelme_crack_test import (  # noqa: E402
    build_gt_masks,
    compute_crack_metrics,
    load_stage_b,
    load_student,
    parse_name_path,
    predict_student,
    predict_teacher,
)
from folds import load_folds  # noqa: E402
from scripts.labelme_crack_copy_paste import parse_csv_set, read_labelme  # noqa: E402
from train_seg_stage_c_mixed import load_labelme_image, resolve_labelme_image  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
STUDENT_NAMES = ["S0", "S1", "S1_attn", "S2", "S2_attn", "YOLOSeg", "UNet", "DeepLab"]
TEACHER_NAMES = ["stage_b_raw", "stage_b_adapted"]
COLOR_PRED = (255, 0, 0)
COLOR_TP = (255, 0, 0)
COLOR_FP = (255, 220, 0)
COLOR_FN = (0, 220, 0)
COLOR_GT = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize crack segmentation predictions of multiple models.")
    # data
    p.add_argument("--labelme-dir", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--image-roots", type=Path, nargs="+", default=None, help="Plain image dirs (no GT).")
    p.add_argument("--fold-split", type=Path, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--stems", type=str, default="", help="Comma-separated stems to render")
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    # models
    p.add_argument("--teacher", action="append", default=[], help="NAME=PATH to Stage-B teacher best.pt; repeatable")
    p.add_argument("--student", action="append", default=[], help="NAME=PATH to student best.pt; repeatable")
    p.add_argument("--teacher-weights", type=str, default="weights/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--auto-discover", action="store_true", help="Scan --runs-root for teacher/student best.pt")
    p.add_argument("--runs-root", type=Path, default=None, help="Base fold dir for auto-discover (default runs/kfold/fold{fold})")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated model names to skip")
    # render
    p.add_argument("--imgsz", type=int, default=0, help="0 = use each checkpoint's imgsz")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--overlay", choices=("both", "mask", "error"), default="both")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max-side", type=int, default=1024, help="Downscale images for display only")
    p.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-mode", choices=("per_image", "sheet", "both"), default="per_image")
    p.add_argument("--output-dir", type=Path, default=Path("runs/analysis/pred_vis"))
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    return p.parse_args()


def blend(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    if m.any():
        c = np.array(color, dtype=np.float32)
        out[m] = out[m] * (1.0 - alpha) + c * alpha
    return out.astype(np.uint8)


def error_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    g = (gt == 1) & valid
    p = (pred == 1) & valid
    tp = g & p
    fp = p & ~g
    fn = g & ~p
    for m, c in ((tp, COLOR_TP), (fp, COLOR_FP), (fn, COLOR_FN)):
        if m.any():
            cc = np.array(c, dtype=np.float32)
            out[m] = out[m] * (1.0 - alpha) + cc * alpha
    return out.astype(np.uint8)


def downscale(img: np.ndarray, max_side: int, nearest: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil = Image.fromarray(img)
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(pil.resize((nw, nh), resample))


def discover_images(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            out.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
    return out


def select(items: list, stems: list[str], num_images: int, seed: int) -> list:
    if stems:
        wanted = {s.strip() for s in stems if s.strip()}
        picked = [it for it in items if _stem_of(it) in wanted]
        return picked
    if num_images > 0 and len(items) > num_images:
        rng = random.Random(seed)
        return rng.sample(items, num_images)
    return list(items)


def _stem_of(item: Any) -> str:
    if isinstance(item, Path):
        return item.stem
    return str(item["stem"])


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return [{stem, image_rgb, gt(optional), valid(optional)}]."""
    cracks = parse_csv_set(args.crack_labels)
    ignores = parse_csv_set(args.ignore_labels)
    comps = parse_csv_set(args.component_labels)
    stems = [s for s in args.stems.split(",") if s.strip()]

    if args.labelme_dir is not None:
        labelme_dir = args.labelme_dir.expanduser().resolve()
        images_dir = (args.images_dir or args.labelme_dir).expanduser().resolve()
        if args.fold_split is not None:
            split = load_folds(args.fold_split.expanduser().resolve())
            fold_samples = [(Path(s["json"]), Path(s["image"])) for s in split["folds"][int(args.fold)]]
            ann_paths = [a for a, _ in fold_samples]
        else:
            ann_paths = sorted(labelme_dir.glob("*.json"))
        ann_paths = select(ann_paths, stems, args.num_images, args.seed)
        out: list[dict[str, Any]] = []
        for ann in ann_paths:
            data = read_labelme(ann)
            img_path = resolve_labelme_image(images_dir, ann, data)
            if img_path is None:
                print(f"[warn] skip {ann.name}: image not found", flush=True)
                continue
            img = load_labelme_image(data, img_path)
            image_rgb = np.asarray(img, dtype=np.uint8)
            h, w = image_rgb.shape[:2]
            gt, valid = build_gt_masks(data, (w, h), crack_labels=cracks, ignore_labels=ignores, component_labels=comps)
            out.append({"stem": ann.stem, "image_rgb": image_rgb, "gt": gt, "valid": valid})
        return out

    if args.image_roots:
        roots = [r.expanduser().resolve() for r in args.image_roots]
        img_paths = select(discover_images(roots), stems, args.num_images, args.seed)
        out = []
        for p in img_paths:
            with Image.open(p) as im:
                image_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            out.append({"stem": p.stem, "image_rgb": image_rgb, "gt": None, "valid": None})
        return out

    raise ValueError("Provide --labelme-dir (with GT) or --image-roots (no GT).")


def load_models(args: argparse.Namespace, device) -> list[dict[str, Any]]:
    excl = {s.strip() for s in args.exclude.split(",") if s.strip()}
    specs: list[tuple[str, Path, str]] = []
    for item in args.teacher:
        name, path = parse_name_path(item)
        specs.append((name, path, "teacher"))
    for item in args.student:
        name, path = parse_name_path(item)
        specs.append((name, path, "student"))

    if args.auto_discover:
        base = (args.runs_root or Path(f"runs/kfold/fold{args.fold}")).expanduser().resolve()
        for name in TEACHER_NAMES:
            ck = base / name / "best.pt"
            if ck.is_file() and name not in [s[0] for s in specs]:
                specs.append((name, ck, "teacher"))
        for name in STUDENT_NAMES:
            ck = base / name / "best.pt"
            if ck.is_file() and name not in [s[0] for s in specs]:
                specs.append((name, ck, "student"))

    models: list[dict[str, Any]] = []
    for name, path, kind in specs:
        if name in excl:
            continue
        path = path.expanduser().resolve()
        if not path.is_file():
            print(f"[warn] skip {name}: checkpoint not found {path}", flush=True)
            continue
        if kind == "teacher":
            model, imgsz = load_stage_b(path, args.teacher_weights, device)
            num_classes = int(getattr(model, "num_classes", 2))
        else:
            model, imgsz, num_classes = load_student(path, device)
        if args.imgsz > 0:
            imgsz = int(args.imgsz)
        models.append({"name": name, "kind": kind, "model": model, "imgsz": imgsz, "num_classes": num_classes})
        print(f"[model] {name} ({kind}) imgsz={imgsz}", flush=True)
    if not models:
        raise RuntimeError("No models loaded.")
    return models


def predict(model_entry: dict[str, Any], image_rgb: np.ndarray, stride: int, device) -> np.ndarray:
    if model_entry["kind"] == "teacher":
        return predict_teacher(model_entry["model"], image_rgb, imgsz=model_entry["imgsz"], stride=stride, device=device)
    return predict_student(
        model_entry["model"], image_rgb, imgsz=model_entry["imgsz"], stride=stride,
        num_classes=model_entry["num_classes"], device=device,
    )


def render(samples, models, preds_by_img, args, out_dir: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(models)
    index: list[dict[str, Any]] = []
    for si, sample in enumerate(samples):
        stem = sample["stem"]
        image_rgb = sample["image_rgb"]
        gt = sample["gt"]
        valid = sample["valid"]
        has_gt = gt is not None and valid is not None
        overlay = args.overlay
        if not has_gt:
            overlay = "mask"
        rows = 2 if overlay == "both" else 1
        ncols = 1 + (1 if has_gt else 0) + n_models
        fig, axes = plt.subplots(rows, ncols, figsize=(3.1 * ncols, 3.3 * rows), squeeze=False)

        img_disp = downscale(image_rgb, args.max_side, nearest=False)
        gt_d = downscale(gt, args.max_side, nearest=True) if has_gt else None
        valid_d = downscale(valid.astype(np.uint8) * 255, args.max_side, nearest=True) > 0 if has_gt else None

        def draw_mask(ax, title, pred_full=None):
            vis = img_disp
            if pred_full is not None:
                vis = blend(img_disp, downscale(pred_full, args.max_side, nearest=True) == 1, COLOR_PRED, args.alpha)
            ax.imshow(vis)
            if has_gt:
                ax.contour(gt_d == 1, levels=[0.5], colors=[tuple(c / 255 for c in COLOR_GT)], linewidths=0.8)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        def draw_error(ax, title, pred_full):
            pred_d = downscale(pred_full, args.max_side, nearest=True)
            ax.imshow(error_overlay(img_disp, gt_d, pred_d, valid_d, args.alpha))
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # column 0: image
        draw_mask(axes[0][0], "image")
        col = 1
        if has_gt:
            draw_mask(axes[0][1], "GT")
            if overlay == "both":
                draw_mask(axes[1][0], "image")
                draw_mask(axes[1][1], "GT")
            col = 2

        for mi, m in enumerate(models):
            pred = preds_by_img[stem][m["name"]]
            title = m["name"]
            rec = {"stem": stem, "model": m["name"]}
            if has_gt:
                met = compute_crack_metrics(gt, pred, valid)
                rec.update(met.as_dict())
                if args.show_metrics:
                    title = f"{m['name']}\nIoU={met.iou:.3f} F1={met.f1:.3f}"
            draw_mask(axes[0][col + mi], title, pred)
            if overlay == "both":
                draw_error(axes[1][col + mi], m["name"], pred)
            index.append(rec)

        fig.tight_layout()
        out_path = out_dir / f"{stem}_panel.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[viz] {out_path}", flush=True)

    return index


def render_sheet(samples, models, preds_by_img, args, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(models)
    rows = len(samples)
    ncols = 1 + 1 + n_models  # image + GT + models (mask overlay; GT may be blank)
    fig, axes = plt.subplots(rows, ncols, figsize=(2.6 * ncols, 2.8 * rows), squeeze=False)
    for ri, sample in enumerate(samples):
        img_disp = downscale(sample["image_rgb"], args.max_side, nearest=False)
        ax = axes[ri][0]
        ax.imshow(img_disp); ax.set_title(sample["stem"], fontsize=7); ax.axis("off")
        gt = sample["gt"]
        ax = axes[ri][1]
        ax.imshow(img_disp); ax.set_title("GT", fontsize=7); ax.axis("off")
        if gt is not None:
            gt_d = downscale(gt, args.max_side, nearest=True)
            ax.contour(gt_d == 1, levels=[0.5], colors=[tuple(c / 255 for c in COLOR_GT)], linewidths=0.8)
        for mi, m in enumerate(models):
            pred = preds_by_img[sample["stem"]][m["name"]]
            pred_d = downscale(pred, args.max_side, nearest=True)
            ax = axes[ri][2 + mi]
            ax.imshow(blend(img_disp, pred_d == 1, COLOR_PRED, args.alpha))
            ax.set_title(m["name"], fontsize=7); ax.axis("off")
    fig.tight_layout()
    path = out_dir / "_sheet.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[viz] {path}", flush=True)


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device(args.device)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args)
    if not samples:
        raise RuntimeError("No samples selected.")
    print(f"[data] {len(samples)} images; any_gt={samples[0]['gt'] is not None}", flush=True)

    models = load_models(args, device)

    preds_by_img: dict[str, dict[str, np.ndarray]] = {}
    for sample in samples:
        preds_by_img[sample["stem"]] = {}
        for m in models:
            with torch.no_grad():
                preds_by_img[sample["stem"]][m["name"]] = predict(m, sample["image_rgb"], args.stride, device)

    index: list[dict[str, Any]] = []
    if args.save_mode in ("per_image", "both"):
        index = render(samples, models, preds_by_img, args, out_dir)
    if args.save_mode in ("sheet", "both"):
        render_sheet(samples, models, preds_by_img, args, out_dir)

    if index:
        (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
