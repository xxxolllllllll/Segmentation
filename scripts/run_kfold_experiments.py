#!/usr/bin/env python3
"""5-fold cross-validation orchestrator for the crack-segmentation distillation grid.

Pipeline:
  1. Build a deterministic K-fold split over ``data/labelme/all`` (cached manifest).
  2. Train Stage A (global, self-supervised) on the external unlabeled pool.
  3. Per fold f (0..K-1):
     a. Train the Stage A+B teacher on the 4 training folds (adapters frozen).
     b. Distill/train students S0/S1/S1_attn/S2/S2_attn/S3/S3_attn on the same 4 folds.
     c. Evaluate every model on the held-out fold f.
  4. Aggregate mean +- std over folds and write reports.

All steps are idempotent: existing checkpoints / metrics are reused.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SOLUTION_ROOT = ROOT / "solution"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from folds import build_kfold, discover_samples, load_folds, save_folds  # noqa: E402


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_ROOT = env_path("DATA_ROOT", ROOT / "data")
WEIGHTS_ROOT = env_path("WEIGHTS_ROOT", ROOT / "weights")
RUNS_ROOT = env_path("RUNS_ROOT", ROOT / "runs")

LABELME_ALL_DIR = env_path("LABELME_ALL_DIR", DATA_ROOT / "labelme" / "all")
STAGE_A_EXTERNAL_DIR = env_path("STAGE_A_EXTERNAL_DIR", DATA_ROOT / "stage_a_external")
DINO_WEIGHTS = env_path("DINO_WEIGHTS", WEIGHTS_ROOT / "dinov3-vitb16-pretrain-lvd1689m")
YOLO_WEIGHTS = env_path("YOLO_WEIGHTS", WEIGHTS_ROOT / "yolo11m-seg.pt")

KFOLD_ROOT = env_path("KFOLD_ROOT", RUNS_ROOT / "kfold")
KFOLD_SPLIT = env_path("KFOLD_SPLIT", KFOLD_ROOT / "folds.json")
STAGE_A_RUN_DIR = env_path("STAGE_A_RUN_DIR", RUNS_ROOT / "stage_a")
STAGE_A_CKPT = env_path("STAGE_A_CKPT", STAGE_A_RUN_DIR / "stage_a_last.pt")
REPORT_ROOT = KFOLD_ROOT / "reports"

K = int(os.environ.get("KFOLD_K", "5"))
KFOLD_SEED = int(os.environ.get("KFOLD_SEED", "42"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "4"))
IMGSZ = int(os.environ.get("IMGSZ", "1024"))
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "800"))
BATCH_SIZE = int(os.environ.get("KFOLD_BATCH_SIZE", "2"))
STAGE_B_BATCH_SIZE = int(os.environ.get("STAGE_B_BATCH_SIZE", "2"))
MAX_EPOCHS = int(os.environ.get("KFOLD_MAX_EPOCHS", "100"))
EARLY_STOP_PATIENCE = int(os.environ.get("KFOLD_EARLY_STOP_PATIENCE", "10"))
VAL_RATIO = float(os.environ.get("KFOLD_VAL_RATIO", "0.10"))
VAL_STRIDE = int(os.environ.get("KFOLD_VAL_STRIDE", "512"))
DEVICE = os.environ.get("DEVICE", "cuda")

TRAIN_STAGE_A = ROOT / "solution" / "train_teacher_stage_a.py"
TRAIN_STAGE_B = ROOT / "solution" / "train_teacher_stage_b.py"
TRAIN_STAGE_C = ROOT / "solution" / "train_seg_stage_c_mixed.py"
EVAL = ROOT / "solution" / "scripts" / "eval_labelme_crack_test.py"


@dataclass(frozen=True)
class Experiment:
    paper_id: str
    teacher_mode: str
    lambda_feat: float
    lambda_attn: float


EXPERIMENTS = [
    Experiment("S0", "raw_vit", 0.0, 0.0),
    Experiment("S1", "raw_vit", 0.5, 0.0),
    Experiment("S1_attn", "raw_vit", 0.5, 0.2),
    Experiment("S2", "stage_a", 0.5, 0.0),
    Experiment("S2_attn", "stage_a", 0.5, 0.2),
    Experiment("S3", "stage_b", 0.5, 0.0),
    Experiment("S3_attn", "stage_b", 0.5, 0.2),
]


def run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def fold_dir(fold: int) -> Path:
    return KFOLD_ROOT / f"fold{fold}"


def stage_b_dir(fold: int) -> Path:
    return fold_dir(fold) / "stage_b"


def exp_dir(fold: int, exp_id: str) -> Path:
    return fold_dir(fold) / exp_id


def build_folds() -> dict:
    KFOLD_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    if KFOLD_SPLIT.is_file():
        return load_folds(KFOLD_SPLIT)
    samples = discover_samples(LABELME_ALL_DIR, LABELME_ALL_DIR)
    if not samples:
        raise FileNotFoundError(f"No LabelMe samples under {LABELME_ALL_DIR}")
    split = build_kfold(samples, k=K, seed=KFOLD_SEED)
    save_folds(split, KFOLD_SPLIT)
    print(f"[folds] built K={K} over {len(samples)} samples -> {KFOLD_SPLIT}", flush=True)
    return split


def run_stage_a() -> None:
    if STAGE_A_CKPT.is_file():
        print(f"[stage-a] skip (checkpoint exists): {STAGE_A_CKPT}", flush=True)
        return
    if not STAGE_A_EXTERNAL_DIR.is_dir():
        raise FileNotFoundError(f"Stage-A external pool missing: {STAGE_A_EXTERNAL_DIR}")
    STAGE_A_RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(TRAIN_STAGE_A),
        "--input-roots", str(STAGE_A_EXTERNAL_DIR),
        "--output-dir", str(STAGE_A_RUN_DIR),
        "--teacher-weights", str(DINO_WEIGHTS),
        "--adapter-indices", "3,4,7,8,11,12",
        "--adapter-bottleneck", "64", "--adapter-dropout", "0.1",
        "--proj-hidden-dim", "2048", "--proj-out-dim", "1024", "--proj-dropout", "0.0",
        "--num-global-crops", "2", "--num-mid-crops", "2", "--num-local-crops", "4",
        "--elongated-ratio-threshold", "2.5", "--include-ann-prob", "0.85", "--max-crop-aspect", "1.6",
        "--global-crop-size", "448", "--mid-crop-size", "320", "--local-crop-size", "160",
        "--global-normal-side-frac", "0.45,0.80", "--mid-normal-side-frac", "0.25,0.50", "--local-normal-side-frac", "0.10,0.25",
        "--global-short-side-frac", "0.70,1.00", "--global-long-side-frac", "0.20,0.45",
        "--mid-short-side-frac", "0.40,0.80", "--mid-long-side-frac", "0.10,0.25",
        "--local-short-side-frac", "0.20,0.50", "--local-long-side-frac", "0.05,0.12",
        "--epochs", "80", "--batch-size", "8",
        "--num-workers", str(NUM_WORKERS), "--persistent-workers", "--prefetch-factor", str(PREFETCH_FACTOR),
        "--lr", "2e-4", "--weight-decay", "1e-4",
        "--ema-momentum", "0.996", "--student-temp", "0.1", "--teacher-temp", "0.04", "--center-momentum", "0.9",
        "--seed", "42", "--device", DEVICE,
    ]
    run_cmd(cmd)


def run_stage_b(fold: int) -> Path:
    out = stage_b_dir(fold)
    best = out / "best.pt"
    if best.is_file():
        print(f"[stage-b] fold={fold} skip (best.pt exists): {best}", flush=True)
        return best
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(TRAIN_STAGE_B),
        "--labelme-dir", str(LABELME_ALL_DIR),
        "--images-dir", str(LABELME_ALL_DIR),
        "--teacher-weights", str(DINO_WEIGHTS),
        "--stage-a-ckpt", str(STAGE_A_CKPT),
        "--output-dir", str(out),
        "--num-classes", "2",
        "--imgsz", str(IMGSZ),
        "--epochs", str(MAX_EPOCHS), "--batch-size", str(STAGE_B_BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS), "--persistent-workers", "--prefetch-factor", str(PREFETCH_FACTOR),
        "--lr", "1e-4", "--weight-decay", "1e-4",
        "--val-ratio", str(VAL_RATIO), "--seed", str(KFOLD_SEED),
        "--fold-split", str(KFOLD_SPLIT), "--fold", str(fold),
        "--early-stop-patience", str(EARLY_STOP_PATIENCE), "--val-stride", str(VAL_STRIDE),
        "--device", DEVICE,
        "--adapter-bottleneck", "64", "--adapter-dropout", "0.1",
        "--lambda-ce", "1.0", "--lambda-dice", "1.0", "--ignore-index", "255",
        "--crack-labels", "crack", "--ignore-labels", "ignore", "--component-labels", "component,wood", "--ncp-label", "ncp",
        "--labelme-sliding-window", "--window-stride", str(WINDOW_STRIDE),
        "--positive-patch-ratio", "0.6",
        "--log-every", "10",
    ]
    run_cmd(cmd)
    if not best.is_file():
        raise FileNotFoundError(f"Stage-B training finished but best.pt missing: {best}")
    return best


def run_stage_c(fold: int, exp: Experiment, stage_b_ckpt: Path | None) -> Path:
    out = exp_dir(fold, exp.paper_id)
    best = out / "best.pt"
    if best.is_file():
        print(f"[stage-c] {exp.paper_id} fold={fold} skip (best.pt exists)", flush=True)
        return best
    out.mkdir(parents=True, exist_ok=True)

    use_teacher = exp.lambda_feat > 0.0 or exp.lambda_attn > 0.0
    cmd = [
        PYTHON, str(TRAIN_STAGE_C),
        "--curated-labelme-dir", str(LABELME_ALL_DIR),
        "--images-dir", str(LABELME_ALL_DIR),
        "--student-weights", str(YOLO_WEIGHTS),
        "--teacher-mode", exp.teacher_mode,
        "--output-dir", str(out),
        "--num-classes", "2",
        "--imgsz", str(IMGSZ), "--teacher-img-size", str(IMGSZ),
        "--window-stride", str(WINDOW_STRIDE),
        "--batch-size-curated", str(BATCH_SIZE),
        "--epochs", str(MAX_EPOCHS),
        "--num-workers", str(NUM_WORKERS), "--persistent-workers", "--prefetch-factor", str(PREFETCH_FACTOR),
        "--lr", "1e-4", "--weight-decay", "1e-4",
        "--val-ratio", str(VAL_RATIO), "--seed", str(KFOLD_SEED),
        "--fold-split", str(KFOLD_SPLIT), "--fold", str(fold),
        "--early-stop-patience", str(EARLY_STOP_PATIENCE), "--val-stride", str(VAL_STRIDE),
        "--device", DEVICE,
        "--lambda-ce", "1.0", "--lambda-dice", "1.0",
        "--lambda-feat-curated", f"{exp.lambda_feat}",
        "--lambda-attn-curated", f"{exp.lambda_attn}",
        "--attn-crack-gamma", "3.0",
        "--decoder-channels", "256,192,128,64",
        "--log-every", "10",
    ]
    if use_teacher:
        cmd += ["--teacher-weights", str(DINO_WEIGHTS)]
        if exp.teacher_mode == "stage_a":
            cmd += ["--teacher-stage-a-ckpt", str(STAGE_A_CKPT)]
        elif exp.teacher_mode == "stage_b":
            assert stage_b_ckpt is not None
            cmd += ["--teacher-stage-b-ckpt", str(stage_b_ckpt)]
    run_cmd(cmd)
    if not best.is_file():
        raise FileNotFoundError(f"Stage-C training finished but best.pt missing: {best}")
    return best


def run_eval(fold: int, stage_b_ckpt: Path, exp_ids: list[str]) -> Path:
    out = fold_dir(fold) / "eval"
    metrics_path = out / "metrics.json"
    if metrics_path.is_file():
        print(f"[eval] fold={fold} skip (metrics.json exists)", flush=True)
        return metrics_path
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL),
        "--labelme-dir", str(LABELME_ALL_DIR),
        "--output-dir", str(out),
        "--stage-b-ckpt", str(stage_b_ckpt),
        "--teacher-weights", str(DINO_WEIGHTS),
        "--fold-split", str(KFOLD_SPLIT), "--fold", str(fold),
        "--stride", str(VAL_STRIDE),
        "--device", DEVICE,
    ]
    for exp_id in exp_ids:
        cmd += ["--student-ckpt", f"{exp_id}={exp_dir(fold, exp_id) / 'best.pt'}"]
    run_cmd(cmd)
    return metrics_path


def read_model_micro(metrics_path: Path, model: str) -> dict[str, float]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = data.get("results") or data.get("models") or []
    for r in results:
        if r["model"] == model:
            micro = r["micro"]
            return {
                "iou": float(micro["iou"]),
                "f1": float(micro["f1"]),
                "precision": float(micro["precision"]),
                "recall": float(micro["recall"]),
            }
    raise KeyError(f"model {model} not found in {metrics_path}")


def aggregate() -> None:
    model_names = [e.paper_id for e in EXPERIMENTS] + ["stage_b"]
    seed_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model in model_names:
        values = {m: [] for m in ("iou", "f1", "precision", "recall")}
        for fold in range(K):
            metrics_path = fold_dir(fold) / "eval" / "metrics.json"
            micro = read_model_micro(metrics_path, model)
            for k in values:
                values[k].append(micro[k])
            seed_rows.append({"model": model, "fold": fold, **micro})
        entry = {"model": model, "folds": list(range(K))}
        for k in ("iou", "f1", "precision", "recall"):
            arr = np.array(values[k], dtype=float)
            entry[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
        summary_rows.append(entry)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ROOT / "per_fold.json").write_text(json.dumps(seed_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with (REPORT_ROOT / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "iou", "f1", "precision", "recall"])
        for row in seed_rows:
            w.writerow([row["model"], row["fold"], f"{row['iou']:.6f}", f"{row['f1']:.6f}",
                        f"{row['precision']:.6f}", f"{row['recall']:.6f}"])

    with (REPORT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "iou_mean", "iou_std", "f1_mean", "f1_std", "precision_mean", "precision_std", "recall_mean", "recall_std"])
        for row in summary_rows:
            w.writerow([
                row["model"],
                f"{row['iou']['mean']:.6f}", f"{row['iou']['std']:.6f}",
                f"{row['f1']['mean']:.6f}", f"{row['f1']['std']:.6f}",
                f"{row['precision']['mean']:.6f}", f"{row['precision']['std']:.6f}",
                f"{row['recall']['mean']:.6f}", f"{row['recall']['std']:.6f}",
            ])

    lines = [
        "| model | IoU (mean +- std) | F1 (mean +- std) | Precision (mean +- std) | Recall (mean +- std) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {m} | {iou:.4f} +- {iou_std:.4f} | {f1:.4f} +- {f1_std:.4f} | "
            "{p:.4f} +- {p_std:.4f} | {r:.4f} +- {r_std:.4f} |".format(
                m=row["model"],
                iou=row["iou"]["mean"], iou_std=row["iou"]["std"],
                f1=row["f1"]["mean"], f1_std=row["f1"]["std"],
                p=row["precision"]["mean"], p_std=row["precision"]["std"],
                r=row["recall"]["mean"], r_std=row["recall"]["std"],
            )
        )
    (REPORT_ROOT / "mean_std_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[done] reports -> {REPORT_ROOT}", flush=True)


def main() -> int:
    KFOLD_ROOT.mkdir(parents=True, exist_ok=True)
    build_folds()
    run_stage_a()

    exp_ids = [e.paper_id for e in EXPERIMENTS]
    for fold in range(K):
        stage_b_ckpt = run_stage_b(fold)
        for exp in EXPERIMENTS:
            run_stage_c(fold, exp, stage_b_ckpt if exp.teacher_mode == "stage_b" else None)
        run_eval(fold, stage_b_ckpt, exp_ids)

    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
