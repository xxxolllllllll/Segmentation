#!/usr/bin/env python3
"""Threshold sweep on the whole-image crack metric (one forward, many thresholds).

Stitches the student's crack probability over the full image (same windowing as
training/eval) and then, for a list of thresholds, reports micro/macro IoU, F1,
precision and recall against LabelMe GT (valid = component & ~ignore).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))
from folds import load_folds  # noqa: E402
from scripts.labelme_crack_copy_paste import read_labelme  # noqa: E402
from val_eval import (  # noqa: E402
    CRACK_CLASS,
    build_gt_masks,
    crop_bottom_right_pad,
    image_to_tensor_01,
    sliding_starts,
)
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=Path("runs/kfold/folds.json"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", choices=("test", "val", "all"), default="test")
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8")
    p.add_argument("--val-stride", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def crack_prob_full(model, image_rgb: np.ndarray, *, imgsz: int, stride: int, device: torch.device) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y in sliding_starts(h, imgsz, stride):
        for x in sliding_starts(w, imgsz, stride):
            patch, crop_w, crop_h = crop_bottom_right_pad(image_rgb, x, y, imgsz)
            tensor = image_to_tensor_01(patch).unsqueeze(0).to(device)
            logits, _ = model(tensor)
            prob = F.softmax(logits.float(), dim=1)[0, 1, :crop_h, :crop_w].cpu().numpy()
            prob_sum[y:y + crop_h, x:x + crop_w] += prob
            count[y:y + crop_h, x:x + crop_w] += 1.0
    return prob_sum / np.maximum(count, 1.0)


def main() -> None:
    args = parse_args()
    from scripts.eval_labelme_crack_test import load_student
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, imgsz, num_classes = load_student(args.ckpt, device)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    print(f"[sweep] ckpt={args.ckpt} split={args.split} fold={args.fold} imgsz={imgsz} "
          f"stride={args.val_stride} thresholds={thresholds}", flush=True)

    split_data = load_folds(args.folds)
    folds = split_data["folds"]
    if args.split == "all":
        entries = sorted([e for f in folds for e in f], key=lambda e: e["stem"])
    elif args.split == "val":
        from folds import fold_train_val_test
        _tr, val, _te = fold_train_val_test(split_data, args.fold, args.val_ratio, args.seed)
        entries = sorted(val, key=lambda e: e["stem"])
    else:
        entries = sorted(folds[args.fold], key=lambda e: e["stem"])
    if args.limit > 0:
        entries = entries[: args.limit]

    crack_labels = {s.strip().lower() for s in args.crack_labels.split(",") if s.strip()}
    ignore_labels = {s.strip().lower() for s in args.ignore_labels.split(",") if s.strip()}
    component_labels = {s.strip().lower() for s in args.component_labels.split(",") if s.strip()}

    acc = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    macro_iou = {t: [] for t in thresholds}
    n_img = 0
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
        prob = crack_prob_full(model, image, imgsz=imgsz, stride=args.val_stride, device=device)
        gt_c = (gt == CRACK_CLASS) & valid
        for t in thresholds:
            pred = (prob >= t) & valid
            tp = int(np.logical_and(gt_c, pred).sum())
            fp = int(np.logical_and(~gt_c, pred).sum())
            fn = int(np.logical_and(gt_c, ~pred).sum())
            acc[t]["tp"] += tp
            acc[t]["fp"] += fp
            acc[t]["fn"] += fn
            macro_iou[t].append(tp / max(tp + fp + fn, 1))
        n_img += 1
        if i % 20 == 0 or i == len(entries):
            tbl = " ".join(f"{t}:{acc[t]['tp']/max(acc[t]['tp']+acc[t]['fp']+acc[t]['fn'],1):.3f}" for t in thresholds)
            print(f"  {i}/{len(entries)} microIoU {tbl}", flush=True)

    rows = []
    for t in thresholds:
        tp, fp, fn = acc[t]["tp"], acc[t]["fp"], acc[t]["fn"]
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        iou = tp / max(tp + fp + fn, 1e-9)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"threshold": t, "micro_iou": iou, "macro_iou": float(np.mean(macro_iou[t]) if macro_iou[t] else 0.0),
                     "f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn})
    print(f"\n=== THRESHOLD SWEEP (fold{args.fold} {args.split}, n={n_img}, {time.time()-t0:.0f}s) ===")
    print(f"{'thr':>5} {'IoU':>7} {'mIoU':>7} {'F1':>7} {'P':>7} {'R':>7}")
    for r in rows:
        print(f"{r['threshold']:>5.2f} {r['micro_iou']:>7.4f} {r['macro_iou']:>7.4f} "
              f"{r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")
    best = max(rows, key=lambda r: r["micro_iou"])
    print(f"\nbest micro IoU = {best['micro_iou']:.4f} @ thr={best['threshold']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"ckpt": str(args.ckpt), "split": args.split, "fold": args.fold,
                                        "imgsz": imgsz, "stride": args.val_stride, "n_images": n_img,
                                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
