#!/usr/bin/env python3
"""Master table over all calibrated_eval.json results under runs/kfold/fold0.

Outputs two tables to stdout (and optionally a file):
  A) calibrated headline: val-tuned threshold -> test IoU/F1/P/R + IoU@0.5
  B) precision at matched recall (the fairest "is the whole curve better?" view)
     and mean precision over the [0.5, 0.85] recall range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = {
    "abl_fpn_inst_l1": "baseline",
    "recallB1_cw": "B1 weightedCE",
    "recallB2_attn": "B2 attn0.5",
    "recallB4_shallow1": "B4 shallow-only",
    "recallB3_pos8": "B3 pos-ratio0.8",
    "scaleAug_06_125": "E1a scale0.6-1.25",
    "scaleAug_05_10": "E1b scale0.5-1.0",
    "stage_b_adapted": "teacher",
    "SMGNet_paper640": "SMGNet paper640",
}
REF_ORDER = ["baseline", "B1 weightedCE", "B2 attn0.5", "B4 shallow-only",
             "B3 pos-ratio0.8", "E1a scale0.6-1.25", "E1b scale0.5-1.0",
             "SMGNet paper640", "teacher"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/kfold/fold0"))
    p.add_argument("--out", type=Path, default=Path("runs/kfold/fold0/reports/master_table_all.md"))
    p.add_argument("--recalls", type=str, default="0.50,0.60,0.70,0.75,0.80,0.85")
    return p.parse_args()


def load(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/calibrated_eval.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = f.parent.name
        label = LABELS.get(name, name)
        at_val = d.get("test_at_val_threshold") or {}
        at_half = d.get("test_at_0.5") or {}
        rows = d.get("test_rows") or []
        R = np.array([r["recall"] for r in sorted(rows, key=lambda r: r["recall"])]) if rows else np.array([])
        P = np.array([r["precision"] for r in sorted(rows, key=lambda r: r["recall"])]) if rows else np.array([])
        out.append({
            "dir": name, "label": label, "model": d.get("model", "?"),
            "thr": d.get("val_best_threshold"),
            "iou": at_val.get("micro_iou"), "f1": at_val.get("f1"),
            "p": at_val.get("precision"), "r": at_val.get("recall"),
            "iou05": at_half.get("micro_iou"),
            "R": R, "P": P,
        })
    out.sort(key=lambda x: -(x["iou"] or 0.0))
    return out


def fmt(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def main() -> None:
    args = parse_args()
    rows = load(args.root)
    recalls = [float(x) for x in args.recalls.split(",") if x.strip()]
    lines = []
    lines.append("# Master table — fold-0 calibrated whole-image crack evaluation")
    lines.append("")
    lines.append(f"Generated from {len(rows)} `calibrated_eval.json` results under `{args.root}`.")
    lines.append("")

    # ---- Table A ----
    lines.append("## A) Calibrated headline (val-tuned threshold -> test)")
    lines.append("")
    lines.append("| model | val-tuned thr | **test IoU** | F1 | P | R | IoU@0.5 | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        pending = "" if r["iou"] is not None else "pending"
        lines.append(f"| {r['label']} | {fmt(r['thr'],2)} | **{fmt(r['iou'])}** | {fmt(r['f1'])} | "
                     f"{fmt(r['p'])} | {fmt(r['r'])} | {fmt(r['iou05'])} | {pending} |")
    lines.append("")

    # ---- Table B: precision at matched recall ----
    lines.append("## B) Precision at matched recall (higher = better curve)")
    lines.append("")
    header = "| model | " + " | ".join(f"R={x:.2f}" for x in recalls) + " | meanP(R∈[0.70,0.80]) |"
    lines.append(header)
    lines.append("|" + "---|" * (len(recalls) + 2))
    for r in rows:
        if r["R"].size == 0:
            lines.append(f"| {r['label']} | " + " | ".join(["-"] * len(recalls)) + " | - |")
            continue
        vals = []
        for x in recalls:
            if r["R"].min() - 1e-6 <= x <= r["R"].max() + 1e-6:
                vals.append(f"{np.interp(x, r['R'], r['P']):.3f}")
            else:
                vals.append("-")
        # fair curve summary: average precision over the common recall range only
        if (r["R"].min() <= 0.70 + 1e-6) and (r["R"].max() >= 0.80 - 1e-6):
            grid = np.linspace(0.70, 0.80, 11)
            meanp = f"{np.interp(grid, r['R'], r['P']).mean():.3f}"
        else:
            meanp = "-"
        lines.append(f"| {r['label']} | " + " | ".join(vals) + f" | {meanp} |")
    lines.append("")
    lines.append("Notes: '-' = the model's threshold grid does not reach that recall. "
                 "meanP is averaged only over the common R∈[0.70,0.80] range (models that do not "
                 "cover it are '-'): this avoids the unfairness of averaging over different recall spans.")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[master] wrote {args.out}")


if __name__ == "__main__":
    main()
