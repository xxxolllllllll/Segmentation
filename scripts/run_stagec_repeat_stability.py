#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SOLUTION_ROOT = ROOT / "solution"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from checkpoint_io import torch_load_compat


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


TRAIN_SCRIPT = ROOT / "solution" / "train_seg_stage_c_mixed.py"
EVAL_SCRIPT = ROOT / "solution" / "scripts" / "eval_labelme_crack_test.py"

DATA_ROOT = env_path("DATA_ROOT", ROOT / "data")
WEIGHTS_ROOT = env_path("WEIGHTS_ROOT", ROOT / "weights")
RUNS_ROOT = env_path("RUNS_ROOT", ROOT / "runs")

TEST_LABELME_DIR = env_path("LABELME_TEST_DIR", DATA_ROOT / "labelme" / "test")
TRAINVAL_LABELME_DIR = env_path("LABELME_TRAINVAL_DIR", DATA_ROOT / "labelme" / "trainval")
CURATED_LABELME_DIR = env_path("LABELME_CURATED_DIR", TRAINVAL_LABELME_DIR)

TEACHER_WEIGHTS = env_path("DINO_WEIGHTS", WEIGHTS_ROOT / "dinov3-vitb16-pretrain-lvd1689m")
STUDENT_WEIGHTS = env_path("YOLO_WEIGHTS", WEIGHTS_ROOT / "yolo11m-seg.pt")
STAGE_B_CKPT = env_path("STAGE_B_FULL_CKPT", RUNS_ROOT / "stage_b_full" / "best.pt")

IMPORT_RUNS_ROOT = env_path("IMPORT_RUNS_ROOT", RUNS_ROOT / "imported")
S4_EXISTING_SEED42_CKPT = env_path(
    "S4_EXISTING_SEED42_CKPT",
    IMPORT_RUNS_ROOT / "stage_c_s4_copy_paste" / "best.pt",
)

ARTIFACT_ROOT = env_path("REPEAT_ROOT", RUNS_ROOT / "repeat_stagec_stability")
TRAIN_ROOT = ARTIFACT_ROOT / "train"
EVAL_ROOT = ARTIFACT_ROOT / "eval"
REPORT_ROOT = ARTIFACT_ROOT / "reports"

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "4"))


@dataclass(frozen=True)
class Experiment:
    paper_id: str
    existing_seed42_name: str
    existing_seed42_ckpt: Path
    curated_labelme_dir: Path
    teacher_mode: str
    batch_size_curated: int
    lambda_feat_curated: float
    lambda_attn_curated: float
    student_weights: Path = STUDENT_WEIGHTS
    teacher_stage_b_ckpt: Path | None = None
    teacher_weights: Path | None = None
    imgsz: int = 1024
    window_stride: int = 800
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.05
    num_workers: int = NUM_WORKERS
    prefetch_factor: int = PREFETCH_FACTOR
    decoder_channels: str = "256,192,128,64"
    copy_paste_prob: float = 0.0
    log_every: int = 10

    def train_output_dir(self, seed: int) -> Path:
        return TRAIN_ROOT / f"{self.paper_id}_seed{seed}"

    def eval_output_dir(self, seed: int) -> Path:
        return EVAL_ROOT / f"{self.paper_id}_seed{seed}"


EXPERIMENTS: list[Experiment] = [
    Experiment(
        paper_id="S2",
        existing_seed42_name="S2",
        existing_seed42_ckpt=IMPORT_RUNS_ROOT / "stage_c_s2_feature" / "best.pt",
        curated_labelme_dir=TRAINVAL_LABELME_DIR,
        teacher_mode="stage_b",
        batch_size_curated=2,
        lambda_feat_curated=0.5,
        lambda_attn_curated=0.0,
        epochs=25,
        teacher_stage_b_ckpt=STAGE_B_CKPT,
        teacher_weights=TEACHER_WEIGHTS,
    ),
    Experiment(
        paper_id="S3",
        existing_seed42_name="S3",
        existing_seed42_ckpt=IMPORT_RUNS_ROOT / "stage_c_s3_attn" / "best.pt",
        curated_labelme_dir=CURATED_LABELME_DIR,
        teacher_mode="stage_b",
        batch_size_curated=4,
        lambda_feat_curated=0.5,
        lambda_attn_curated=0.2,
        epochs=45,
        teacher_stage_b_ckpt=STAGE_B_CKPT,
        teacher_weights=TEACHER_WEIGHTS,
    ),
    Experiment(
        paper_id="S0",
        existing_seed42_name="S0_new",
        existing_seed42_ckpt=IMPORT_RUNS_ROOT / "stage_c_s0" / "best.pt",
        curated_labelme_dir=CURATED_LABELME_DIR,
        teacher_mode="stage_b",
        batch_size_curated=4,
        lambda_feat_curated=0.0,
        lambda_attn_curated=0.0,
        epochs=20,
        teacher_stage_b_ckpt=STAGE_B_CKPT,
        teacher_weights=TEACHER_WEIGHTS,
    ),
    Experiment(
        paper_id="S1",
        existing_seed42_name="S1",
        existing_seed42_ckpt=IMPORT_RUNS_ROOT / "stage_c_s1_raw" / "best.pt",
        curated_labelme_dir=TRAINVAL_LABELME_DIR,
        teacher_mode="raw_vit",
        batch_size_curated=2,
        lambda_feat_curated=0.5,
        lambda_attn_curated=0.0,
        epochs=20,
        teacher_weights=TEACHER_WEIGHTS,
    ),
    Experiment(
        paper_id="S4",
        existing_seed42_name="S4",
        existing_seed42_ckpt=S4_EXISTING_SEED42_CKPT,
        curated_labelme_dir=CURATED_LABELME_DIR,
        teacher_mode="stage_b",
        batch_size_curated=4,
        lambda_feat_curated=0.5,
        lambda_attn_curated=0.2,
        epochs=45,
        teacher_stage_b_ckpt=STAGE_B_CKPT,
        teacher_weights=TEACHER_WEIGHTS,
        copy_paste_prob=0.7,
    ),
]

SEEDS = [42, 52, 62]
MISSING_TRAIN_SEEDS = [52, 62]


def selected_experiments() -> list[Experiment]:
    spec = os.environ.get("EXPERIMENT_FILTER", "").strip()
    if not spec:
        return EXPERIMENTS
    wanted = {item.strip() for item in spec.split(",") if item.strip()}
    chosen = [exp for exp in EXPERIMENTS if exp.paper_id in wanted]
    missing = sorted(wanted - {exp.paper_id for exp in chosen})
    if missing:
        raise ValueError(f"Unknown experiment ids in EXPERIMENT_FILTER: {missing}")
    return chosen


def run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_metrics_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_checkpoint_epoch(path: Path) -> int:
    ckpt = torch_load_compat(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", 0))


def extract_micro(metrics_json: dict[str, Any], model_name: str) -> dict[str, float]:
    results = metrics_json.get("results")
    if results is None:
        results = metrics_json.get("models", [])
    for result in results:
        if result["model"] == model_name:
            return {
                "iou": float(result["micro"]["iou"]),
                "f1": float(result["micro"]["f1"]),
                "precision": float(result["micro"]["precision"]),
                "recall": float(result["micro"]["recall"]),
            }
    raise KeyError(f"Model {model_name} not found in metrics.json")


def ensure_existing_seed42_eval(exp: Experiment) -> dict[str, float]:
    out_dir = EVAL_ROOT / f"{exp.paper_id}_seed42_existing"
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            str(EVAL_SCRIPT),
            "--labelme-dir",
            str(TEST_LABELME_DIR),
            "--output-dir",
            str(out_dir),
            "--stage-b-ckpt",
            str(STAGE_B_CKPT),
            "--teacher-weights",
            str(TEACHER_WEIGHTS),
            "--student-ckpt",
            f"{exp.existing_seed42_name}={exp.existing_seed42_ckpt}",
        ]
        run_cmd(cmd)
    return extract_micro(read_metrics_json(metrics_path), exp.existing_seed42_name)


def train_repeat_seed(exp: Experiment, seed: int) -> Path:
    out_dir = exp.train_output_dir(seed)
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"
    target_epochs = int(exp.epochs)

    if last_ckpt.is_file():
        last_epoch = read_checkpoint_epoch(last_ckpt)
        if last_epoch >= target_epochs:
            print(
                f"[skip-train] {exp.paper_id} seed={seed} already reached epoch {last_epoch} "
                f"(target={target_epochs})",
                flush=True,
            )
            return best_ckpt if best_ckpt.is_file() else last_ckpt

    if best_ckpt.is_file() and not last_ckpt.is_file():
        print(f"[skip-train] {exp.paper_id} seed={seed} has best checkpoint without last checkpoint", flush=True)
        return best_ckpt

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(TRAIN_SCRIPT),
        "--curated-labelme-dir",
        str(exp.curated_labelme_dir),
        "--student-weights",
        str(exp.student_weights),
        "--teacher-mode",
        exp.teacher_mode,
        "--output-dir",
        str(out_dir),
        "--imgsz",
        str(exp.imgsz),
        "--window-stride",
        str(exp.window_stride),
        "--epochs",
        str(exp.epochs),
        "--batch-size-curated",
        str(exp.batch_size_curated),
        "--lr",
        str(exp.lr),
        "--weight-decay",
        str(exp.weight_decay),
        "--val-ratio",
        str(exp.val_ratio),
        "--seed",
        str(seed),
        "--num-workers",
        str(exp.num_workers),
        "--persistent-workers",
        "--prefetch-factor",
        str(exp.prefetch_factor),
        "--device",
        "cuda",
        "--lambda-feat-curated",
        str(exp.lambda_feat_curated),
        "--lambda-attn-curated",
        str(exp.lambda_attn_curated),
        "--copy-paste-prob",
        str(exp.copy_paste_prob),
        "--decoder-channels",
        exp.decoder_channels,
        "--log-every",
        str(exp.log_every),
    ]
    if exp.teacher_stage_b_ckpt is not None:
        cmd.extend(["--teacher-stage-b-ckpt", str(exp.teacher_stage_b_ckpt)])
    if exp.teacher_weights is not None:
        cmd.extend(["--teacher-weights", str(exp.teacher_weights)])
    if last_ckpt.is_file():
        last_epoch = read_checkpoint_epoch(last_ckpt)
        if last_epoch < target_epochs:
            cmd.extend(["--resume", str(last_ckpt)])
            print(
                f"[resume-train] {exp.paper_id} seed={seed} from epoch {last_epoch} to target {target_epochs}",
                flush=True,
            )
    run_cmd(cmd)
    if not best_ckpt.is_file():
        raise FileNotFoundError(f"Training finished but best checkpoint missing: {best_ckpt}")
    return best_ckpt


def eval_seed(exp: Experiment, seed: int, ckpt_path: Path, *, model_name: str) -> dict[str, float]:
    out_dir = exp.eval_output_dir(seed)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            str(EVAL_SCRIPT),
            "--labelme-dir",
            str(TEST_LABELME_DIR),
            "--output-dir",
            str(out_dir),
            "--stage-b-ckpt",
            str(STAGE_B_CKPT),
            "--teacher-weights",
            str(TEACHER_WEIGHTS),
            "--student-ckpt",
            f"{model_name}={ckpt_path}",
        ]
        run_cmd(cmd)
    return extract_micro(read_metrics_json(metrics_path), model_name)


def summarize(rows: list[dict[str, Any]], experiments: list[Experiment]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for exp in experiments:
        exp_rows = [row for row in rows if row["model"] == exp.paper_id]
        entry: dict[str, Any] = {"model": exp.paper_id, "seeds": [row["seed"] for row in exp_rows]}
        for metric in ("iou", "f1", "precision", "recall"):
            values = np.array([row[metric] for row in exp_rows], dtype=float)
            entry[metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary.append(entry)
    return summary


def write_reports(seed_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "repeat_seed_metrics.json").write_text(json.dumps(seed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ROOT / "repeat_stability_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with (REPORT_ROOT / "repeat_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "iou", "f1", "precision", "recall", "ckpt"])
        for row in seed_rows:
            writer.writerow(
                [
                    row["model"],
                    row["seed"],
                    f"{row['iou']:.6f}",
                    f"{row['f1']:.6f}",
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    row["ckpt"],
                ]
            )

    with (REPORT_ROOT / "repeat_stability_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "seed_count", "seeds", "iou_mean", "iou_std", "f1_mean", "f1_std", "precision_mean", "precision_std", "recall_mean", "recall_std"]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["model"],
                    len(row["seeds"]),
                    ",".join(str(s) for s in row["seeds"]),
                    f"{row['iou']['mean']:.6f}",
                    f"{row['iou']['std']:.6f}",
                    f"{row['f1']['mean']:.6f}",
                    f"{row['f1']['std']:.6f}",
                    f"{row['precision']['mean']:.6f}",
                    f"{row['precision']['std']:.6f}",
                    f"{row['recall']['mean']:.6f}",
                    f"{row['recall']['std']:.6f}",
                ]
            )

    lines = [
        "| 模型 | seed 数 | IoU (mean +- std) | F1 (mean +- std) | Precision (mean +- std) | Recall (mean +- std) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {seed_count} | {iou_mean:.4f} +- {iou_std:.4f} | {f1_mean:.4f} +- {f1_std:.4f} | "
            "{precision_mean:.4f} +- {precision_std:.4f} | {recall_mean:.4f} +- {recall_std:.4f} |".format(
                model=row["model"],
                seed_count=len(row["seeds"]),
                iou_mean=row["iou"]["mean"],
                iou_std=row["iou"]["std"],
                f1_mean=row["f1"]["mean"],
                f1_std=row["f1"]["std"],
                precision_mean=row["precision"]["mean"],
                precision_std=row["precision"]["std"],
                recall_mean=row["recall"]["mean"],
                recall_std=row["recall"]["std"],
            )
        )
    (REPORT_ROOT / "repeat_stability_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_inputs() -> None:
    experiments = selected_experiments()
    required_paths = [
        TEST_LABELME_DIR,
        TRAINVAL_LABELME_DIR,
        TEACHER_WEIGHTS,
        STUDENT_WEIGHTS,
        STAGE_B_CKPT,
    ]
    if any(exp.curated_labelme_dir == CURATED_LABELME_DIR for exp in experiments):
        required_paths.append(CURATED_LABELME_DIR)
    required_paths.extend(exp.existing_seed42_ckpt for exp in experiments)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main() -> int:
    validate_inputs()
    experiments = selected_experiments()
    for path in (TRAIN_ROOT, EVAL_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    for exp in experiments:
        seed42_metrics = ensure_existing_seed42_eval(exp)
        seed_rows.append({"model": exp.paper_id, "seed": 42, "ckpt": str(exp.existing_seed42_ckpt), **seed42_metrics})
        for seed in MISSING_TRAIN_SEEDS:
            ckpt_path = train_repeat_seed(exp, seed)
            metrics = eval_seed(exp, seed, ckpt_path, model_name=f"{exp.paper_id}_seed{seed}")
            seed_rows.append({"model": exp.paper_id, "seed": seed, "ckpt": str(ckpt_path), **metrics})

    summary_rows = summarize(seed_rows, experiments)
    write_reports(seed_rows, summary_rows)
    print("\n[done] repeat stability reports written to:", REPORT_ROOT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
