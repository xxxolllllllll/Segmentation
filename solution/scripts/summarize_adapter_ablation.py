#!/usr/bin/env python3
"""Aggregate the adapter ablation (Stage-B teacher raw vs adapted) across folds.

Reads runs/kfold/fold{f}/stage_b_{raw,adapted}/calibrated_eval.json and reports
per-fold test IoU (val-tuned threshold), mean+-std and a paired test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/kfold"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("runs/kfold/reports/adapter_ablation.md"))
    return p.parse_args()


def read_iou(root: Path, fold: int, variant: str):
    f = root / f"fold{fold}" / f"stage_b_{variant}" / "calibrated_eval.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    at = d.get("test_at_val_threshold") or {}
    return at.get("micro_iou")


def main() -> None:
    args = parse_args()
    raw = {f: read_iou(args.root, f, "raw") for f in args.folds}
    adp = {f: read_iou(args.root, f, "adapted") for f in args.folds}

    lines = ["# Adapter ablation — Stage-B teacher: raw (no adapters) vs adapted (Stage-A adapters)",
             "",
             "Per-fold test IoU (whole-image, val-tuned threshold).",
             "",
             "| fold | raw (no adapters) | adapted (Stage-A) | diff (adapted-raw) |",
             "|---|---|---|---|"]
    diffs = []
    for f in args.folds:
        r, a = raw[f], adp[f]
        rs = f"{r:.4f}" if r is not None else "-"
        as_ = f"{a:.4f}" if a is not None else "-"
        ds = f"{a - r:+.4f}" if (r is not None and a is not None) else "-"
        if r is not None and a is not None:
            diffs.append(a - r)
        lines.append(f"| {f} | {rs} | {as_} | {ds} |")

    def ms(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        if a.size == 0:
            return "-", "-"
        return f"{a.mean():.4f}", f"{(a.std(ddof=1) if a.size > 1 else 0.0):.4f}"

    rm, rs = ms([raw[f] for f in args.folds])
    am, as_ = ms([adp[f] for f in args.folds])
    lines += ["", f"**raw**: mean={rm} ± {rs}  |  **adapted**: mean={am} ± {as_}", ""]

    if len(diffs) >= 2:
        d = np.array(diffs)
        pairs = [(f, raw[f], adp[f]) for f in args.folds if raw[f] is not None and adp[f] is not None]
        try:
            from scipy import stats as st
            t, p = st.ttest_rel([a for _, _, a in pairs], [r for _, r, _ in pairs])
            sig = f"paired t-test: t={t:.3f}, p={p:.4f}"
        except Exception as ex:  # pragma: no cover
            sig = f"paired t-test unavailable ({ex})"
        lines += [f"mean diff (adapted-raw) = {d.mean():+.4f} ± {d.std(ddof=1):.4f} (n={len(d)}), "
                  f"adapted wins={int((d>0).sum())}/{len(d)}", sig, ""]
    else:
        lines += ["Not enough paired folds yet.", ""]

    lines += ["Interpretation: if adapted ≈ raw (diff within noise, p high) the Stage-A adapters "
              "do not help the teacher.", ""]
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"[adapter] wrote {args.out}")


if __name__ == "__main__":
    main()
