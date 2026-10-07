#!/usr/bin/env python3
"""Aggregate calibrated_eval.json across folds for the generalization validation.

Variants:
  baseline : fold0 -> abl_fpn_inst_l1, folds 1..4 -> baseline
  E1a      : fold0 -> scaleAug_06_125, folds 1..4 -> scaleAug_06_125

Reports per-fold test IoU (val-tuned threshold) plus mean+-std and a paired
baseline-vs-E1a comparison over the folds that have both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = {
    "baseline": {0: "abl_fpn_inst_l1", 1: "baseline", 2: "baseline", 3: "baseline", 4: "baseline"},
    "E1a_scale0.6-1.25": {0: "scaleAug_06_125", 1: "scaleAug_06_125", 2: "scaleAug_06_125",
                          3: "scaleAug_06_125", 4: "scaleAug_06_125"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/kfold"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("runs/kfold/reports/generalization_baseline_vs_e1a.md"))
    return p.parse_args()


def read_metrics(root: Path, fold: int, run: str) -> dict | None:
    f = root / f"fold{fold}" / run / "calibrated_eval.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    at = d.get("test_at_val_threshold") or {}
    return {"iou": at.get("micro_iou"), "f1": at.get("f1"), "p": at.get("precision"),
            "r": at.get("recall"), "thr": d.get("val_best_threshold")}


def main() -> None:
    args = parse_args()
    data: dict[str, dict[int, dict | None]] = {}
    for name, mapping in VARIANTS.items():
        data[name] = {f: read_metrics(args.root, f, mapping[f]) for f in args.folds}

    lines = ["# Generalization validation — baseline vs E1a (scale 0.6–1.25)",
             "",
             "Per-fold test IoU (whole-image, val-tuned threshold). `-` = not available yet.",
             "",
             "| fold | baseline IoU | E1a IoU | diff (E1a-baseline) |",
             "|---|---|---|---|"]
    diffs = []
    for f in args.folds:
        b = data["baseline"][f]
        e = data["E1a_scale0.6-1.25"][f]
        bs = f"{b['iou']:.4f}" if b and b['iou'] is not None else "-"
        es = f"{e['iou']:.4f}" if e and e['iou'] is not None else "-"
        if b and e and b['iou'] is not None and e['iou'] is not None:
            d = e['iou'] - b['iou']
            diffs.append(d)
            ds = f"{d:+.4f}"
        else:
            ds = "-"
        lines.append(f"| {f} | {bs} | {es} | {ds} |")

    def stats(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        if a.size == 0:
            return "-", "-"
        return f"{a.mean():.4f}", f"{(a.std(ddof=1) if a.size > 1 else 0.0):.4f}"

    b_all = [data["baseline"][f]["iou"] if data["baseline"][f] else None for f in args.folds]
    e_all = [data["E1a_scale0.6-1.25"][f]["iou"] if data["E1a_scale0.6-1.25"][f] else None for f in args.folds]
    bm, bs = stats(b_all)
    em, es = stats(e_all)
    lines += ["", f"**baseline**: mean={bm} ± {bs}  |  **E1a**: mean={em} ± {es}", ""]

    if len(diffs) >= 2:
        d = np.array(diffs)
        try:
            from scipy import stats as st
            t, p = st.ttest_rel(
                [data["E1a_scale0.6-1.25"][f]["iou"] for f in args.folds if data["E1a_scale0.6-1.25"][f] and data["baseline"][f]],
                [data["baseline"][f]["iou"] for f in args.folds if data["E1a_scale0.6-1.25"][f] and data["baseline"][f]],
            )
            ttxt = f"paired t-test: t={t:.3f}, p={p:.4f}"
        except Exception as ex:  # pragma: no cover
            ttxt = f"paired t-test unavailable ({ex})"
        lines += [f"mean diff = {d.mean():+.4f} ± {d.std(ddof=1):.4f} (n={len(d)}), wins={int((d>0).sum())}/{len(d)}",
                  ttxt, ""]
    else:
        lines += ["Not enough paired folds yet for a significance test.", ""]

    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"[kfold] wrote {args.out}")


if __name__ == "__main__":
    main()
