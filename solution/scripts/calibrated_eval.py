#!/usr/bin/env python3
"""Calibrated whole-image evaluation: tune the decision threshold on the
internal validation split, then report on the held-out test fold.

Models compared under one protocol:
  * smgnet         : official paper model (640x480 crops, per-crop norm)
  * student        : Stage-C student from a run's best.pt (0-1, 2-class)
  * teacher_stage_b: DINOv3 Stage-B FPN teacher (ImageNet norm, 2-class)

For each image we stitch the crack probability over the full image (same
windowing as training) and accumulate TP/FP/FN for every candidate threshold,
so a single forward per image covers the whole sweep.  Metrics follow
``solution/val_eval.py`` (valid = component & ~ignore).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))
from folds import fold_train_val_test, load_folds  # noqa: E402
from scripts.labelme_crack_copy_paste import read_labelme  # noqa: E402
from val_eval import (  # noqa: E402
    CRACK_CLASS,
    build_gt_masks,
    crop_bottom_right_pad,
    crop_center_pad,
    image_to_tensor_01,
    sliding_starts,
)
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("smgnet", "student", "teacher_stage_b"), required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=Path("runs/kfold/folds.json"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--thresholds", type=str,
                   default="0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    # smgnet crop geometry / norm
    p.add_argument("--crop-w", type=int, default=640)
    p.add_argument("--crop-h", type=int, default=480)
    p.add_argument("--stride-x", type=int, default=320)
    p.add_argument("--stride-y", type=int, default=240)
    p.add_argument("--norm-mean", type=float, default=149.034)
    p.add_argument("--norm-std", type=float, default=32.022)
    p.add_argument("--bridge", action="store_true")
    # student/teacher windowing
    p.add_argument("--val-stride", type=int, default=512)
    p.add_argument("--teacher-weights", type=str, default="weights/dinov3-vitb16-pretrain-lvd1689m")
    return p.parse_args()


def build_model(args: argparse.Namespace, device: torch.device):
    if args.model == "smgnet":
        official = Path("/home/zgt/Segmentation/SMG-Net/HuiZhenxing/SMG-Net/models/smgnet.py")
        spec = importlib.util.spec_from_file_location("smgnet_official", official)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        model = mod.SMGNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                           split_att="fc", bridge=bool(args.bridge))
        state = torch.load(args.ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        return model.eval().to(device), None
    if args.model == "student":
        from scripts.eval_labelme_crack_test import load_student
        model, imgsz, _nc = load_student(args.ckpt, device)
        return model, imgsz
    from scripts.eval_labelme_crack_test import load_stage_b
    model, imgsz = load_stage_b(args.ckpt, args.teacher_weights, device)
    return model, imgsz


@torch.no_grad()
def prob_full(args: argparse.Namespace, model, imgsz, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    use_amp = device.type == "cuda"
    if args.model == "smgnet":
        img = image_rgb.astype(np.float32)
        for y0 in sliding_starts(h, args.crop_h, args.stride_y):
            for x0 in sliding_starts(w, args.crop_w, args.stride_x):
                ch = min(args.crop_h, h - y0)
                cw = min(args.crop_w, w - x0)
                crop = img[y0:y0 + ch, x0:x0 + cw]
                if ch != args.crop_h or cw != args.crop_w:
                    crop = np.pad(crop, ((0, args.crop_h - ch), (0, args.crop_w - cw), (0, 0)), mode="edge")
                x = (crop - args.norm_mean) / args.norm_std
                x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0
                t = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(device).float()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(t)
                if float(out.min()) < 0.0 or float(out.max()) > 1.0:
                    out = torch.sigmoid(out)
                acc[y0:y0 + ch, x0:x0 + cw] += out[0, 0, :ch, :cw].float().cpu().numpy()
                cnt[y0:y0 + ch, x0:x0 + cw] += 1.0
        return acc / np.maximum(cnt, 1e-6)

    for y in sliding_starts(h, imgsz, args.val_stride):
        for x in sliding_starts(w, imgsz, args.val_stride):
            if args.model == "student":
                patch, crop_w, crop_h = crop_bottom_right_pad(image_rgb, x, y, imgsz)
                tensor = image_to_tensor_01(patch).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(tensor)
                prob = F.softmax(logits.float(), dim=1)[0, 1, :crop_h, :crop_w].cpu().numpy()
            else:
                patch, (top, left, crop_h, crop_w) = crop_center_pad(image_rgb, x, y, imgsz)
                t = patch.astype(np.float32) / 255.0
                t = (t - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
                tensor = torch.from_numpy(np.transpose(t, (2, 0, 1))).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(tensor)
                prob = F.softmax(logits.float(), dim=1)[0, 1, top:top + crop_h, left:left + crop_w].cpu().numpy()
            acc[y:y + crop_h, x:x + crop_w] += prob
            cnt[y:y + crop_h, x:x + crop_w] += 1.0
    return acc / np.maximum(cnt, 1e-6)


def sweep(args, model, imgsz, entries, thresholds, device, labels, tag):
    crack_labels, ignore_labels, component_labels = labels
    acc = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    macro = {t: [] for t in thresholds}
    n = 0
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        img_p = Path(e["image"])
        if not img_p.is_file():
            continue
        data = read_labelme(Path(e["json"]))
        with Image.open(img_p) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        gt, valid = build_gt_masks(data, (w, h), crack_labels=crack_labels,
                                   ignore_labels=ignore_labels, component_labels=component_labels)
        if not bool(valid.any()):
            continue
        prob = prob_full(args, model, imgsz, image, device)
        gt_c = (gt == CRACK_CLASS) & valid
        for t in thresholds:
            pred = (prob >= t) & valid
            tp = int(np.logical_and(gt_c, pred).sum())
            fp = int(np.logical_and(~gt_c, pred).sum())
            fn = int(np.logical_and(gt_c, ~pred).sum())
            acc[t]["tp"] += tp
            acc[t]["fp"] += fp
            acc[t]["fn"] += fn
            macro[t].append(tp / max(tp + fp + fn, 1))
        n += 1
        if i % 25 == 0 or i == len(entries):
            print(f"  [{tag}] {i}/{len(entries)}", flush=True)
    rows = []
    for t in thresholds:
        tp, fp, fn = acc[t]["tp"], acc[t]["fp"], acc[t]["fn"]
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        rows.append({"threshold": t, "micro_iou": tp / max(tp + fp + fn, 1e-9),
                     "macro_iou": float(np.mean(macro[t]) if macro[t] else 0.0),
                     "f1": 2 * prec * rec / max(prec + rec, 1e-9), "precision": prec, "recall": rec,
                     "tp": tp, "fp": fp, "fn": fn})
    print(f"  [{tag}] {n} images in {time.time()-t0:.0f}s", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    model, imgsz = build_model(args, device)
    labels = (
        {s.strip().lower() for s in args.crack_labels.split(",") if s.strip()},
        {s.strip().lower() for s in args.ignore_labels.split(",") if s.strip()},
        {s.strip().lower() for s in args.component_labels.split(",") if s.strip()},
    )
    split = load_folds(args.folds)
    folds = split["folds"]
    _tr, val, test = fold_train_val_test(split, args.fold, args.val_ratio, args.seed)
    val = sorted(val, key=lambda e: e["stem"])[: args.limit or None]
    test = sorted(test, key=lambda e: e["stem"])[: args.limit or None]
    print(f"[cal] model={args.model} ckpt={args.ckpt} imgsz={imgsz} "
          f"val={len(val)} test={len(test)} thresholds={len(thresholds)}", flush=True)

    val_rows = sweep(args, model, imgsz, val, thresholds, device, labels, "val")
    best = max(val_rows, key=lambda r: r["micro_iou"])
    print(f"[cal] val-best threshold={best['threshold']} micro_iou={best['micro_iou']:.4f} "
          f"F1={best['f1']:.4f} P={best['precision']:.4f} R={best['recall']:.4f}", flush=True)

    test_rows = sweep(args, model, imgsz, test, thresholds, device, labels, "test")
    test_best = max(test_rows, key=lambda r: r["micro_iou"])
    test_at_val = next(r for r in test_rows if abs(r["threshold"] - best["threshold"]) < 1e-9)
    test_at_half = next(r for r in test_rows if abs(r["threshold"] - 0.5) < 1e-9)

    print(f"\n=== {args.model} (fold{args.fold}) ===")
    print(f"{'thr':>5} {'val IoU':>8} | {'test IoU':>8} {'F1':>7} {'P':>7} {'R':>7}")
    for vr in val_rows:
        tr = next((r for r in test_rows if r["threshold"] == vr["threshold"]), None)
        if tr:
            print(f"{vr['threshold']:>5.2f} {vr['micro_iou']:>8.4f} | {tr['micro_iou']:>8.4f} "
                  f"{tr['f1']:>7.4f} {tr['precision']:>7.4f} {tr['recall']:>7.4f}")
    print(f"\nval-tuned thr={best['threshold']:.2f} -> test IoU={test_at_val['micro_iou']:.4f} "
          f"(F1={test_at_val['f1']:.4f} P={test_at_val['precision']:.4f} R={test_at_val['recall']:.4f})")
    print(f"test-optimal thr={test_best['threshold']:.2f} -> test IoU={test_best['micro_iou']:.4f}")
    print(f"thr=0.50 -> test IoU={test_at_half['micro_iou']:.4f} "
          f"(F1={test_at_half['f1']:.4f} P={test_at_half['precision']:.4f} R={test_at_half['recall']:.4f})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "model": args.model, "ckpt": str(args.ckpt), "fold": args.fold, "imgsz": imgsz,
            "val_images": len(val), "test_images": len(test),
            "val_best_threshold": best["threshold"], "val_rows": val_rows, "test_rows": test_rows,
            "test_at_val_threshold": test_at_val, "test_at_0.5": test_at_half, "test_best": test_best,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cal] wrote {args.out}")


if __name__ == "__main__":
    main()
