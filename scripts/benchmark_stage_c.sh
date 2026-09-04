#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

need_dir "$LABELME_TRAINVAL_DIR"
need_dir "$DINO_WEIGHTS"
need_file "$YOLO_WEIGHTS"
need_file "$STAGE_B_FULL_CKPT"

MAX_STEPS="${MAX_STEPS:-200}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"
BENCH_NAME="${BENCH_NAME:-wsl_stagec}"
BENCH_OUT="$RUNS_ROOT/benchmarks/$BENCH_NAME"
mkdir -p "$BENCH_OUT"

python_run "$ROOT/solution/train_seg_stage_c_mixed.py" \
  --curated-labelme-dir "$LABELME_TRAINVAL_DIR" \
  --student-weights "$YOLO_WEIGHTS" \
  --teacher-mode stage_b \
  --teacher-stage-b-ckpt "$STAGE_B_FULL_CKPT" \
  --teacher-weights "$DINO_WEIGHTS" \
  --output-dir "$BENCH_OUT" \
  --imgsz "$IMGSZ" \
  --window-stride "$WINDOW_STRIDE" \
  --epochs 1 \
  --batch-size-curated 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --val-ratio 0.05 \
  --seed 42 \
  --num-workers "$NUM_WORKERS" \
  --persistent-workers \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --device cuda \
  --lambda-feat-curated 0.5 \
  --lambda-attn-curated 0.0 \
  --copy-paste-prob 0.0 \
  --decoder-channels 256,192,128,64 \
  --log-every 50 \
  --max-steps "$MAX_STEPS" \
  --max-val-batches "$MAX_VAL_BATCHES"
