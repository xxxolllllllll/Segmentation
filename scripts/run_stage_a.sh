#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

need_dir "$STAGE_A_ROOT"
need_dir "$DINO_WEIGHTS"
mkdir -p "$STAGE_A_RUN_DIR"

python_run "$ROOT/solution/train_teacher_stage_a.py" \
  --input-roots "$STAGE_A_ROOT" \
  --output-dir "$STAGE_A_RUN_DIR" \
  --teacher-weights "$DINO_WEIGHTS" \
  --adapter-indices 3,4,7,8,11,12 \
  --adapter-bottleneck 64 \
  --adapter-dropout 0.1 \
  --proj-hidden-dim 2048 \
  --proj-out-dim 1024 \
  --proj-dropout 0.0 \
  --num-global-crops 2 \
  --num-mid-crops 2 \
  --num-local-crops 4 \
  --elongated-ratio-threshold 2.5 \
  --include-ann-prob 0.85 \
  --max-crop-aspect 1.6 \
  --global-crop-size 448 \
  --mid-crop-size 320 \
  --local-crop-size 160 \
  --global-normal-side-frac 0.45,0.80 \
  --mid-normal-side-frac 0.25,0.50 \
  --local-normal-side-frac 0.10,0.25 \
  --global-short-side-frac 0.70,1.00 \
  --global-long-side-frac 0.20,0.45 \
  --mid-short-side-frac 0.40,0.80 \
  --mid-long-side-frac 0.10,0.25 \
  --local-short-side-frac 0.20,0.50 \
  --local-long-side-frac 0.05,0.12 \
  --epochs "$STAGE_A_EPOCHS" \
  --batch-size "$STAGE_A_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --persistent-workers \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --ema-momentum 0.996 \
  --student-temp 0.1 \
  --teacher-temp 0.04 \
  --center-momentum 0.9 \
  --seed 42 \
  --device cuda
