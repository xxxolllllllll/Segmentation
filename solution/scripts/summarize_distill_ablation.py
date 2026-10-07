#!/usr/bin/env python3
"""Distillation ablation across folds: non-distilled supervised student (S0,
runs_bk/kfold/fold*/S0) vs feature-KD student (runs/kfold/fold*/baseline;
fold0 = abl_fpn_inst_l1).

Both use the identical folds.json and the same calibrated whole-image metric, so
the only intended difference is the distillation term.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# fold -> run dir for the KD (distilled) student
KD_RUNS = {0: Path("runs/kfold/fold0/abl_fpn_inst_l1"), 1: Path("runs/kfold/fold1/baseline"),
           2: Path("runs/kfold/fold2/baseline"), 3: Path("runs/kfold/fold3/baseline"),
           4: Path("runs/kfold/fold4/baseline")}
S0_RUNS = {f: Path(f"runs_bk/kfold/fold{f}/S0") for f in range(5)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path, default=Path("runs/kfold/reports/distill_ablation_S0_vs_KD.md"))
    return p.parse_args()


def read_iou(d: Path):
    p = d / "calibrated_eval.json"
    if not p.is_file():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    return (j.get("test_at_val_threshold") or {}).get("micro_iou")


def main() -> None:
    args = parse_args()
    s0 = {f: read_iou(S0_RUNS[f]) for f in args.folds}
    kd = {f: read_iou(KD_RUNS[f]) for f in args.folds}

    lines = ["# Distillation ablation (K=5) — supervised S0 vs feature-KD student", "",
             "Whole-image test IoU (val-tuned threshold); identical folds.json, identical metric.",
             "", "| fold | S0 (no distill) | KD student | diff (KD - S0) |", "|---|---|---|---|"]
    diffs = []
    for f in args.folds:
        a, b = s0[f], kd[f]
        as_ = f"{a:.4f}" if a is not None else "-"
        bs = f"{b:.4f}" if b is not None else "-"
        ds = "-"
        if a is not None and b is not None:
            diffs.append(b - a)
            ds = f"{b - a:+.4f}"
        lines.append(f"| {f} | {as_} | {bs} | {ds} |")

    def ms(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        if a.size == 0:
            return "-", "-"
        return f"{a.mean():.4f}", f"{(a.std(ddof=1) if a.size > 1 else 0.0):.4f}"

    s0m, s0s = ms([s0[f] for f in args.folds])
    kdm, kds = ms([kd[f] for f in args.folds])
    lines += ["", f"**S0 (no distill)**: mean={s0m} ± {s0s}  |  **KD student**: mean={kdm} ± {kds}", ""]

    if len(diffs) >= 2:
        d = np.array(diffs)
        pairs = [(f, s0[f], kd[f]) for f in args.folds if s0[f] is not None and kd[f] is not None]
        try:
            from scipy import stats as st
            t, p = st.ttest_rel([b for _, _, b in pairs], [a for _, a, _ in pairs])
            sig = f"paired t-test (KD - S0): t={t:.3f}, p={p:.4f}"
        except Exception as ex:  # pragma: no cover
            sig = f"paired t-test unavailable ({ex})"
        lines += [f"mean diff (KD - S0) = {d.mean():+.4f} ± {d.std(ddof=1):.4f} (n={len(d)}), "
                  f"KD wins={int((d>0).sum())}/{len(d)}", sig, ""]
    else:
        lines += ["Not enough paired folds yet.", ""]

    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"[distill-ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
