#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"

STAGE_A_ROOT="${STAGE_A_ROOT:-$DATA_ROOT/stage_a}"
STAGE_A_EXTERNAL_DIR="${STAGE_A_EXTERNAL_DIR:-$DATA_ROOT/stage_a_external}"
LABELME_ALL_DIR="${LABELME_ALL_DIR:-$DATA_ROOT/labelme/all}"

KFOLD_K="${KFOLD_K:-5}"
KFOLD_SEED="${KFOLD_SEED:-42}"
KFOLD_ROOT="${KFOLD_ROOT:-$RUNS_ROOT/kfold}"
KFOLD_SPLIT="${KFOLD_SPLIT:-$KFOLD_ROOT/folds.json}"

DINO_WEIGHTS="${DINO_WEIGHTS:-$WEIGHTS_ROOT/dinov3-vitb16-pretrain-lvd1689m}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$WEIGHTS_ROOT/yolo11m-seg.pt}"

STAGE_A_RUN_DIR="${STAGE_A_RUN_DIR:-$RUNS_ROOT/stage_a}"
STAGE_B_FULL_RUN_DIR="${STAGE_B_FULL_RUN_DIR:-$RUNS_ROOT/stage_b_full}"
STAGE_B_GENERIC_RUN_DIR="${STAGE_B_GENERIC_RUN_DIR:-$RUNS_ROOT/stage_b_generic}"

STAGE_A_CKPT="${STAGE_A_CKPT:-$STAGE_A_RUN_DIR/stage_a_last.pt}"
STAGE_B_FULL_CKPT="${STAGE_B_FULL_CKPT:-$STAGE_B_FULL_RUN_DIR/best.pt}"
STAGE_B_GENERIC_CKPT="${STAGE_B_GENERIC_CKPT:-$STAGE_B_GENERIC_RUN_DIR/best.pt}"


mkdir -p "$RUNS_ROOT"
need_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[missing-dir] $1" >&2
    exit 1
  fi
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[missing-file] $1" >&2
    exit 1
  fi
}

python_run() {
  "$PYTHON_BIN" "$@"
}
