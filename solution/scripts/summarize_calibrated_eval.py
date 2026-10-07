#!/usr/bin/env python3
"""Summarize calibrated_eval.py JSON outputs into a markdown table.

Usage:
  python solution/scripts/summarize_calibrated_eval.py --inputs a.json b.json \
      --out runs/kfold/reports/calibrated_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="optional row labels (same order as --inputs)")
    return p.parse_args()


def row_from(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    at_val = d.get("test_at_val_threshold") or {}
    at_half = d.get("test_at_0.5") or {}
    return {
        "name": path.parent.name if path.parent.name else path.stem,
        "model": d.get("model", "?"),
        "thr": d.get("val_best_threshold"),
        "iou": at_val.get("micro_iou"),
        "f1": at_val.get("f1"),
        "p": at_val.get("precision"),
        "r": at_val.get("recall"),
        "iou05": at_half.get("micro_iou"),
        "val_images": d.get("val_images"),
        "test_images": d.get("test_images"),
    }


def main() -> None:
    args = parse_args()
    rows = [row_from(p) for p in args.inputs]
    if args.labels and len(args.labels) == len(rows):
        for r, lab in zip(rows, args.labels):
            r["name"] = lab
    rows.sort(key=lambda r: -(r["iou"] or 0.0))

    lines = []
    lines.append("# Fold-0 calibrated whole-image crack evaluation (val-tuned threshold -> test)")
    lines.append("")
    lines.append("| model | val-tuned thr | **test IoU** | F1 | P | R | IoU@0.5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        def f(x, n=4):
            return "-" if x is None else f"{x:.{n}f}"
        lines.append(f"| {r['name']} | {f(r['thr'],2)} | **{f(r['iou'])}** | {f(r['f1'])} | "
                     f"{f(r['p'])} | {f(r['r'])} | {f(r['iou05'])} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[summarize] wrote {args.out}")


if __name__ == "__main__":
    main()
