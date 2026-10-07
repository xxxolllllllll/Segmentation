#!/usr/bin/env python3
"""Aggregate the supervised MACUNet baseline across folds (K=5)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/kfold"))
    p.add_argument("--run", type=str, default="MACUNet_base")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("runs/kfold/reports/macunet_kfold.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for f in args.folds:
        p = args.root / f"fold{f}" / args.run / "calibrated_eval.json"
        if not p.is_file():
            rows.append((f, None, None))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        at = d.get("test_at_val_threshold") or {}
        rows.append((f, at.get("micro_iou"), at))

    lines = [f"# {args.run} across folds (whole-image, val-tuned threshold)", "",
             "| fold | test IoU | F1 | P | R |", "|---|---|---|---|---|"]
    vals = []
    for f, iou, at in rows:
        if iou is None:
            lines.append(f"| {f} | - | - | - | - |")
            continue
        vals.append(iou)
        lines.append(f"| {f} | {iou:.4f} | {at.get('f1',0):.4f} | {at.get('precision',0):.4f} | {at.get('recall',0):.4f} |")
    if vals:
        a = np.array(vals)
        lines += ["", f"**mean = {a.mean():.4f} ± {(a.std(ddof=1) if a.size>1 else 0.0):.4f}** (n={a.size})", ""]
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"[macunet] wrote {args.out}")


if __name__ == "__main__":
    main()
