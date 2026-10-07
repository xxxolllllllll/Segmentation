#!/usr/bin/env python3
"""Unified whole-image evaluation via crop -> stitch.

Different models train on different crops (SMG-Net: paper-style 640x480 with
per-crop normalisation; our Stage-C students/teachers: square windows).  To
compare them fairly we predict on crops, stitch the *probabilities* back into
the full image (overlapping crops are averaged), threshold, and then compute
the exact same image-level crack metrics used everywhere else
(``solution/val_eval.py``: micro/macro IoU, F1, precision, recall, plus pixel
accuracy) against the LabelMe GT with ``valid = component & ~ignore``.

Currently implemented: ``--model smgnet``.  The CLI/metric path is shared so
other backbones can be added with a new branch in ``predict_full_image``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from folds import load_folds  # noqa: E402
from scripts.labelme_crack_copy_paste import read_labelme  # noqa: E402
from val_eval import (  # noqa: E402
    CRACK_CLASS,
    aggregate_macro,
    aggregate_micro,
    build_gt_masks,
    compute_crack_metrics,
)
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("smgnet",), default="smgnet")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=Path("runs/kfold/folds.json"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", choices=("test", "val", "all"), default="test")
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--crop-w", type=int, default=640)
    p.add_argument("--crop-h", type=int, default=480)
    p.add_argument("--stride-x", type=int, default=320)
    p.add_argument("--stride-y", type=int, default=240)
    p.add_argument("--norm-mean", type=float, default=149.034)
    p.add_argument("--norm-std", type=float, default=32.022)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bridge", action="store_true", help="SMG-Net SC_Att_Bridge (paper: off)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--save-pred-dir", type=Path, default=None, help="optional: save binary preds as PNG")
    return p.parse_args()


def starts(dim: int, win: int, stride: int) -> list[int]:
    if dim <= win:
        return [0]
    xs = list(range(0, dim - win + 1, stride))
    if xs[-1] != dim - win:
        xs.append(dim - win)
    return xs


@torch.no_grad()
def predict_smgnet_full(model: torch.nn.Module, image_rgb: np.ndarray, args: argparse.Namespace,
                        device: torch.device) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    use_amp = device.type == "cuda"
    img = image_rgb.astype(np.float32)
    for y0 in starts(h, args.crop_h, args.stride_y):
        for x0 in starts(w, args.crop_w, args.stride_x):
            ch = min(args.crop_h, h - y0)
            cw = min(args.crop_w, w - x0)
            crop = img[y0:y0 + ch, x0:x0 + cw]
            if ch != args.crop_h or cw != args.crop_w:
                crop = np.pad(crop, ((0, args.crop_h - ch), (0, args.crop_w - cw), (0, 0)), mode="edge")
            x = (crop - args.norm_mean) / args.norm_std
            x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0
            t = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(device).float()
            with torch.amp.autocast("cuda", enabled=use_amp):
                prob = model(t)
            prob = torch.sigmoid(prob) if prob.min() < 0 or prob.max() > 1 else prob
            prob = prob[0, 0, :ch, :cw].float().cpu().numpy()
            acc[y0:y0 + ch, x0:x0 + cw] += prob
            cnt[y0:y0 + ch, x0:x0 + cw] += 1.0
    acc /= np.maximum(cnt, 1e-6)
    return (acc >= args.threshold).astype(np.uint8)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model == "smgnet":
        import importlib.util
        official = Path("/home/zgt/Segmentation/SMG-Net/HuiZhenxing/SMG-Net/models/smgnet.py")
        spec = importlib.util.spec_from_file_location("smgnet_official", official)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        SMGNet = mod.SMGNet
        model = SMGNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                       split_att="fc", bridge=bool(args.bridge))
        state = torch.load(args.ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval().to(device)
        return model
    raise ValueError(args.model)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_data = load_folds(args.folds)
    folds = split_data["folds"]
    if args.split == "test":
        entries = sorted(folds[args.fold], key=lambda e: e["stem"])
    elif args.split == "all":
        entries = sorted([e for f in folds for e in f], key=lambda e: e["stem"])
    else:
        entries = sorted(folds[args.fold], key=lambda e: e["stem"])  # same fold as "val" here
    if args.limit > 0:
        entries = entries[: args.limit]

    crack_labels = {s.strip().lower() for s in args.crack_labels.split(",") if s.strip()}
    ignore_labels = {s.strip().lower() for s in args.ignore_labels.split(",") if s.strip()}
    component_labels = {s.strip().lower() for s in args.component_labels.split(",") if s.strip()}

    model = build_model(args, device)
    print(f"[eval] model={args.model} ckpt={args.ckpt} split={args.split} n_images={len(entries)}", flush=True)

    metrics_list = []
    per_image = []
    total_px = tp_all = 0
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        ann = Path(e["json"])
        img_p = Path(e["image"])
        if not img_p.is_file():
            continue
        data = read_labelme(ann)
        with Image.open(img_p) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        gt, valid = build_gt_masks(data, (w, h), crack_labels=crack_labels,
                                   ignore_labels=ignore_labels, component_labels=component_labels)
        if not bool(valid.any()):
            continue
        pred = predict_smgnet_full(model, image, args, device)
        if args.save_pred_dir is not None:
            args.save_pred_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred * 255).save(args.save_pred_dir / f"{e['stem']}.png")
        m = compute_crack_metrics(gt, pred, valid)
        metrics_list.append(m)
        gt_c = (gt == CRACK_CLASS) & valid
        pr_c = (pred == CRACK_CLASS) & valid
        total_px += int(valid.sum())
        tp_all += int(np.logical_and(gt_c, pr_c).sum())
        per_image.append({"stem": e["stem"], **m.as_dict()})
        if i % 20 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)} micro_iou={aggregate_micro(metrics_list).iou:.4f}", flush=True)

    micro = aggregate_micro(metrics_list)
    macro = aggregate_macro(metrics_list)
    # pixel accuracy over valid region (crack + non-crack)
    acc_num = tp_all
    acc_den = 0
    for m, e in zip(metrics_list, [x for x in per_image]):
        acc_den += m.valid_pixels
    pa = (1.0 - (sum(m.fp + m.fn for m in metrics_list) / max(sum(m.valid_pixels for m in metrics_list), 1)))
    result = {
        "model": args.model,
        "ckpt": str(args.ckpt),
        "split": args.split,
        "fold": int(args.fold),
        "n_images": len(metrics_list),
        "micro": micro.as_dict(),
        "macro": macro,
        "pixel_accuracy": float(pa),
        "valid_pixels": int(sum(m.valid_pixels for m in metrics_list)),
        "elapsed_sec": round(time.time() - t0, 1),
        "config": {"crop_w": args.crop_w, "crop_h": args.crop_h, "stride_x": args.stride_x,
                   "stride_y": args.stride_y, "threshold": args.threshold, "bridge": bool(args.bridge)},
        "per_image": per_image,
    }
    print("\n=== RESULT ===")
    print(json.dumps({k: result[k] for k in ("model", "split", "fold", "n_images", "micro", "macro", "pixel_accuracy")},
                     ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
