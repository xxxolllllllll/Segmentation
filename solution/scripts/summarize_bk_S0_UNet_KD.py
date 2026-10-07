#!/usr/bin/env python3
"""Three-way fold comparison: runs_bk S0 (no-distill yolo_unet),
runs_bk UNet (supervised U-Net), and the KD student (yolo_unet + feature KD).

All evaluated with the same calibrated whole-image protocol (val-tuned
threshold -> test) on the identical folds.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KD = {0: Path("runs/kfold/fold0/abl_fpn_inst_l1"), 1: Path("runs/kfold/fold1/baseline"),
      2: Path("runs/kfold/fold2/baseline"), 3: Path("runs/kfold/fold3/baseline"),
      4: Path("runs/kfold/fold4/baseline")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("runs/kfold/reports/bk_S0_UNet_KD.md"))
    return p.parse_args()


def iou(d: Path):
    p = d / "calibrated_eval.json"
    if not p.is_file():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    return (j.get("test_at_val_threshold") or {}).get("micro_iou")


def main() -> None:
    args = parse_args()
    rows = {f: {"S0": iou(Path(f"runs_bk/kfold/fold{f}/S0")),
                "UNet": iou(Path(f"runs_bk/kfold/fold{f}/UNet")),
                "KD": iou(KD[f])} for f in args.folds}

    lines = ["# runs_bk S0 / UNet vs KD student (K=5, calibrated whole-image test IoU)", "",
             "| fold | S0 (yolo_unet, no distill) | UNet (supervised) | KD (yolo_unet + distill) |",
             "|---|---|---|---|"]
    for f in args.folds:
        def s(k):
            v = rows[f][k]
            return f"{v:.4f}" if v is not None else "-"
        lines.append(f"| {f} | {s('S0')} | {s('UNet')} | {s('KD')} |")

    def ms(k):
        a = np.array([rows[f][k] for f in args.folds if rows[f][k] is not None], dtype=float)
        if a.size == 0:
            return "-", "-"
        return f"{a.mean():.4f}", f"{(a.std(ddof=1) if a.size > 1 else 0.0):.4f}"

    lines += ["", "| model | mean | std |", "|---|---|---|"]
    for k in ("S0", "UNet", "KD"):
        m, sd = ms(k)
        lines.append(f"| {k} | {m} | {sd} |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"[bk-3way] wrote {args.out}")


if __name__ == "__main__":
    main()
