#!/usr/bin/env python3
"""5-fold cross-validation orchestrator for the crack-segmentation distillation grid.

Pipeline:
  1. Build a deterministic K-fold split over ``<DATA_ROOT>/labelme/all`` (cached manifest).
  2. Train Stage A (global, self-supervised) on the external unlabeled pool, only
     when a selected experiment needs a Stage-A/B teacher.
  3. Per fold f (0..K-1):
     a. Train the Stage A+B teacher on the 4 training folds (adapters frozen), only
        when a selected experiment needs a Stage-B teacher.
     b. Distill/train the selected students on the same 4 folds.
     c. Evaluate every trained model on the held-out fold f (re-run when new
        checkpoints appear, so metrics accumulate across incremental runs).
  4. Aggregate mean +- std over folds and write reports.

Selection:
  ``EXPERIMENT_FILTER`` (comma-separated paper ids) limits which experiments run,
  e.g. ``EXPERIMENT_FILTER=S0`` or ``EXPERIMENT_FILTER=S0,S1,S3``. Default: all.
  Data root is ``DATA_ROOT`` (default ``<project>/data``); outputs go under
  ``RUNS_ROOT`` (default ``<project>/runs``).

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

from checkpoint_io import torch_load_compat  # noqa: E402
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
    student_arch: str = "yolo_unet"
    student_scratch: bool = False
    teacher_ckpt: str = ""  # for teacher_mode="stage_b": "raw" | "adapted"


EXPERIMENTS = [
    Experiment("S0", "raw_vit", 0.0, 0.0, student_scratch=True),
    # S1 = Stage-B teacher WITHOUT Stage-A adaptation (raw backbone -> fusion -> bridge -> FPN decoder)
    Experiment("S1", "stage_b", 0.5, 0.0, student_scratch=True, teacher_ckpt="raw"),
    Experiment("S1_attn", "stage_b", 0.5, 0.2, student_scratch=True, teacher_ckpt="raw"),
    # S2 = Stage-B teacher WITH Stage-A adaptation
    Experiment("S2", "stage_b", 0.5, 0.0, student_scratch=True, teacher_ckpt="adapted"),
    Experiment("S2_attn", "stage_b", 0.5, 0.2, student_scratch=True, teacher_ckpt="adapted"),
    # Architecture-comparison baselines (no distillation; same protocol as S0, from scratch).
    Experiment("YOLOSeg", "raw_vit", 0.0, 0.0, student_arch="yolo_seg"),
    Experiment("UNet", "raw_vit", 0.0, 0.0, student_arch="unet"),
    Experiment("DeepLab", "raw_vit", 0.0, 0.0, student_arch="deeplab"),
]


def selected_experiments() -> list[Experiment]:
    """Experiments to run this invocation, from EXPERIMENT_FILTER (default: all)."""
    spec = os.environ.get("EXPERIMENT_FILTER", "").strip()
    if not spec:
        return list(EXPERIMENTS)
    by_id = {e.paper_id: e for e in EXPERIMENTS}
    wanted = [x.strip() for x in spec.replace(" ", ",").split(",") if x.strip()]
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise ValueError(f"Unknown EXPERIMENT_FILTER ids: {missing}. Available: {list(by_id)}")
    return [by_id[w] for w in wanted]


def experiment_uses_teacher(exp: Experiment) -> bool:
    return exp.lambda_feat > 0.0 or exp.lambda_attn > 0.0


def teacher_stage_requirements(experiments: list[Experiment]) -> tuple[bool, set[str]]:
    """Return (need_stage_a, stage_b_variants) for the selected experiments.

    ``stage_b_variants`` is the set of Stage-B teacher variants to train, e.g.
    {"raw", "adapted"} ("raw" = no Stage-A adapters; "adapted" = load Stage-A).
    """
    teach = [e for e in experiments if experiment_uses_teacher(e)]
    variants = {e.teacher_ckpt for e in teach if e.teacher_mode == "stage_b"}
    need_stage_a = "adapted" in variants
    return need_stage_a, variants


def available_models(fold: int) -> list[str]:
    """All checkpoints currently present for a fold (students + optional stage_b teachers)."""
    models: list[str] = []
    for exp in EXPERIMENTS:
        if (exp_dir(fold, exp.paper_id) / "best.pt").is_file():
            models.append(exp.paper_id)
    for variant in ("raw", "adapted"):
        if (stage_b_dir(fold, variant) / "best.pt").is_file():
            models.append(f"stage_b_{variant}")
    return models


def run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _metrics_state(out: Path) -> tuple[int, int] | None:
    """(last_epoch, patience_counter) recovered from out/epoch_metrics.csv."""
    csv_path = out / "epoch_metrics.csv"
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    last_epoch = 0
    best_epoch = 0
    prev_best: float | None = None
    for row in rows:
        try:
            ep = int(float(str(row.get("epoch", "")).strip()))
        except Exception:
            continue
        bv: float | None = None
        for key in ("best_val", "best_iou"):
            raw = row.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                bv = float(str(raw).strip())
                break
            except Exception:
                continue
        if bv is None:
            continue
        last_epoch = max(last_epoch, ep)
        if prev_best is None or bv > prev_best:
            prev_best = bv
            best_epoch = ep
    if last_epoch <= 0:
        return None
    if best_epoch <= 0:
        best_epoch = last_epoch
    return last_epoch, max(0, last_epoch - best_epoch)


def _ckpt_state(path: Path) -> tuple[int, int] | None:
    """(epoch, patience_counter) from a checkpoint.

    Recovers patience from the sibling ``best.pt`` (last_epoch - best_epoch) when
    the checkpoint predates the ``patience_counter`` field.
    """
    if not path.is_file():
        return None
    try:
        ckpt = torch_load_compat(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    epoch = int(ckpt.get("epoch", 0))
    if "patience_counter" in ckpt:
        return epoch, int(ckpt["patience_counter"])
    best_path = path.parent / "best.pt"
    if best_path.is_file():
        try:
            bck = torch_load_compat(best_path, map_location="cpu", weights_only=False)
            best_epoch = int(bck.get("epoch", 0))
            if best_epoch > 0:
                return epoch, max(0, epoch - best_epoch)
        except Exception:
            pass
    return epoch, 0


def run_decision(out: Path, target_epochs: int, patience: int) -> tuple[str, Path | None]:
    """Classify a Stage-B/C output dir.

    Returns (decision, resume_path) where decision is one of
    ``completed`` (already finished/early-stopped -> skip), ``resume`` (pass
    ``--resume resume_path``) or ``fresh`` (train from scratch).
    """
    best = out / "best.pt"
    last = out / "last.pt"
    if (out / ".completed").is_file():
        return "completed", None
    state = _metrics_state(out)
    if state is not None:
        last_epoch, pat = state
        if last_epoch >= target_epochs or pat >= patience:
            return "completed", None
        if last.is_file():
            return "resume", last
        return "fresh", None
    if last.is_file():
        ck = _ckpt_state(last)
        if ck is not None:
            epoch, cpat = ck
            if epoch >= target_epochs or cpat >= patience:
                return "completed", None
        return "resume", last
    if best.is_file():
        return "completed", None
    return "fresh", None


def _mark_completed(out: Path) -> None:
    """Lazily drop a .completed marker for an already-finished (legacy) run."""
    marker = out / ".completed"
    if marker.is_file():
        return
    try:
        marker.write_text(json.dumps({"marked_by": "run_decision"}), encoding="utf-8")
    except Exception:
        pass


def fold_dir(fold: int) -> Path:
    return KFOLD_ROOT / f"fold{fold}"


def stage_b_dir(fold: int, variant: str = "adapted") -> Path:
    """Stage-B teacher dir for a variant: pure_dir/fold{fold}/stage_b_{raw|adapted}."""
    return fold_dir(fold) / f"stage_b_{variant}"


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
    completed = STAGE_A_RUN_DIR / ".completed"
    if completed.is_file():
        print(f"[stage-a] skip (completed): {STAGE_A_RUN_DIR}", flush=True)
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
        "--ibot-hidden-dim", "2048", "--ibot-out-dim", "2048", "--ibot-dropout", "0.0",
        "--lambda-dino", "1.0", "--lambda-ibot", "1.0",
        "--ibot-student-temp", "0.1", "--ibot-teacher-temp", "0.04", "--ibot-center-momentum", "0.9",
        "--mask-ratio", "0.3", "--mask-min-num-patches", "8",
        "--ibot-diag-every", "100",
        "--num-global-crops", "2", "--num-mid-crops", "2", "--num-local-crops", "4",
        "--elongated-ratio-threshold", "2.5", "--include-ann-prob", "0.85", "--max-crop-aspect", "1.6",
        "--global-crop-size", "1024", "--mid-crop-size", "640", "--local-crop-size", "320",
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
    nproc = int(os.environ.get("STAGE_A_NPROC", "1"))
    if nproc > 1:
        cmd = ["torchrun", "--nproc_per_node", str(nproc), str(TRAIN_STAGE_A)] + cmd[2:]
        print(f"[stage-a] torchrun --nproc_per_node={nproc}", flush=True)
    if STAGE_A_CKPT.is_file():
        cmd += ["--resume", "auto"]
        print(f"[stage-a] resume from {STAGE_A_CKPT}", flush=True)
    run_cmd(cmd)


def run_stage_b(fold: int, variant: str) -> Path:
    out = stage_b_dir(fold, variant)
    best = out / "best.pt"
    last = out / "last.pt"
    decision, resume_from = run_decision(out, MAX_EPOCHS, EARLY_STOP_PATIENCE)
    if decision == "completed":
        _mark_completed(out)
        print(f"[stage-b] fold={fold} variant={variant} skip (completed)", flush=True)
        return best if best.is_file() else (last if last.is_file() else out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(TRAIN_STAGE_B),
        "--labelme-dir", str(LABELME_ALL_DIR),
        "--images-dir", str(LABELME_ALL_DIR),
        "--teacher-weights", str(DINO_WEIGHTS),
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
    if variant == "raw":
        cmd += ["--no-adapters"]
    else:
        cmd += ["--stage-a-ckpt", str(STAGE_A_CKPT)]
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
        print(f"[stage-b] fold={fold} variant={variant} resume from {resume_from}", flush=True)
    run_cmd(cmd)
    if not best.is_file() and not last.is_file():
        raise FileNotFoundError(f"Stage-B training finished but no checkpoint under: {out}")
    return best if best.is_file() else last


def run_stage_c(fold: int, exp: Experiment, stage_b_ckpt: Path | None) -> Path:
    out = exp_dir(fold, exp.paper_id)
    best = out / "best.pt"
    last = out / "last.pt"
    decision, resume_from = run_decision(out, MAX_EPOCHS, EARLY_STOP_PATIENCE)
    if decision == "completed":
        _mark_completed(out)
        print(f"[stage-c] {exp.paper_id} fold={fold} skip (completed)", flush=True)
        return best if best.is_file() else (last if last.is_file() else out)
    out.mkdir(parents=True, exist_ok=True)

    use_teacher = exp.lambda_feat > 0.0 or exp.lambda_attn > 0.0
    cmd = [
        PYTHON, str(TRAIN_STAGE_C),
        "--curated-labelme-dir", str(LABELME_ALL_DIR),
        "--images-dir", str(LABELME_ALL_DIR),
        "--student-weights", str(YOLO_WEIGHTS),
        "--student-arch", exp.student_arch,
        *(["--student-scratch"] if exp.student_scratch else []),
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
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
        print(f"[stage-c] {exp.paper_id} fold={fold} resume from {resume_from}", flush=True)
    run_cmd(cmd)
    if not best.is_file() and not last.is_file():
        raise FileNotFoundError(f"Stage-C training finished but no checkpoint under: {out}")
    return best if best.is_file() else last


def run_eval(fold: int) -> Path | None:
    models = available_models(fold)
    out = fold_dir(fold) / "eval"
    metrics_path = out / "metrics.json"
    if not models:
        print(f"[eval] fold={fold} skip (no checkpoints found)", flush=True)
        return metrics_path if metrics_path.is_file() else None

    if metrics_path.is_file():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            results = existing.get("models") or existing.get("results") or []
            present = {r["model"] for r in results}
        except Exception:
            present = set()
        if set(models) <= present:
            print(f"[eval] fold={fold} skip (metrics.json covers {sorted(models)})", flush=True)
            return metrics_path

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVAL),
        "--labelme-dir", str(LABELME_ALL_DIR),
        "--output-dir", str(out),
        "--fold-split", str(KFOLD_SPLIT), "--fold", str(fold),
        "--stride", str(VAL_STRIDE),
        "--device", DEVICE,
    ]
    teacher_names = [m for m in models if m.startswith("stage_b_")]
    if teacher_names:
        for tn in teacher_names:
            variant = tn[len("stage_b_") :]
            cmd += ["--stage-b-ckpt", f"{tn}={stage_b_dir(fold, variant) / 'best.pt'}"]
        cmd += ["--teacher-weights", str(DINO_WEIGHTS)]
    for model in models:
        if model.startswith("stage_b_"):
            continue
        cmd += ["--student-ckpt", f"{model}={exp_dir(fold, model) / 'best.pt'}"]
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


def discover_report_models() -> list[str]:
    """Models present in every fold's eval metrics, in canonical order."""
    per_fold: list[set[str]] = []
    for fold in range(K):
        metrics_path = fold_dir(fold) / "eval" / "metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = data.get("models") or data.get("results") or []
        per_fold.append({r["model"] for r in results})
    if not per_fold:
        return []
    common = set.intersection(*per_fold)
    ordered = [e.paper_id for e in EXPERIMENTS] + ["stage_b_raw", "stage_b_adapted"]
    return [m for m in ordered if m in common]


def aggregate() -> None:
    model_names = discover_report_models()
    if not model_names:
        print("[aggregate] no eval metrics found for all folds; nothing to aggregate", flush=True)
        return
    print(f"[aggregate] models={model_names}", flush=True)
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


def selected_folds() -> list[int]:
    """Folds to run this invocation, from FOLD_FILTER (default: all 0..K-1)."""
    spec = os.environ.get("FOLD_FILTER", "").strip()
    if not spec:
        return list(range(K))
    folds = [int(x.strip()) for x in spec.replace(" ", ",").split(",") if x.strip()]
    for f in folds:
        if not (0 <= f < K):
            raise ValueError(f"FOLD_FILTER fold {f} out of range for K={K}")
    return folds


def main() -> int:
    KFOLD_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = selected_experiments()
    need_stage_a, stage_b_variants = teacher_stage_requirements(experiments)
    folds = selected_folds()
    print(
        f"[kfold] experiments={[e.paper_id for e in experiments]} folds={folds} "
        f"need_stage_a={need_stage_a} stage_b_variants={sorted(stage_b_variants)}",
        flush=True,
    )

    build_folds()
    if need_stage_a:
        run_stage_a()
    else:
        print("[stage-a] skip (no selected experiment uses an adapted teacher)", flush=True)
    if not stage_b_variants:
        print("[stage-b] skip (no selected experiment uses a Stage-B teacher)", flush=True)

    for fold in folds:
        stage_b_ckpts = (
            {v: run_stage_b(fold, v) for v in sorted(stage_b_variants)} if stage_b_variants else {}
        )
        for exp in experiments:
            ckpt = stage_b_ckpts.get(exp.teacher_ckpt) if exp.teacher_mode == "stage_b" else None
            run_stage_c(fold, exp, ckpt)
        run_eval(fold)

    if os.environ.get("SKIP_AGGREGATE", "").strip().lower() in ("1", "true", "yes"):
        print("[kfold] skip aggregate (SKIP_AGGREGATE set)", flush=True)
    else:
        aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
