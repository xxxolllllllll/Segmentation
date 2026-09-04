#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi

need_dir "$LABELME_TRAINVAL_DIR"
need_dir "$DINO_WEIGHTS"

EXTRA_ARGS=()
OUT_DIR="$STAGE_B_FULL_RUN_DIR"

case "$MODE" in
  full)
    need_file "$STAGE_A_CKPT"
    EXTRA_ARGS+=(--stage-a-ckpt "$STAGE_A_CKPT")
    OUT_DIR="$STAGE_B_FULL_RUN_DIR"
    ;;
  generic)
    OUT_DIR="$STAGE_B_GENERIC_RUN_DIR"
    ;;
  *)
    echo "Usage: bash scripts/run_stage_b.sh [full|generic]" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

python_run "$ROOT/solution/train_teacher_stage_b.py" \
  --labelme-dir "$LABELME_TRAINVAL_DIR" \
  --images-dir "$LABELME_TRAINVAL_DIR" \
  --teacher-weights "$DINO_WEIGHTS" \
  --output-dir "$OUT_DIR" \
  --num-classes 2 \
  --imgsz "$IMGSZ" \
  --epochs "$STAGE_B_EPOCHS" \
  --batch-size "$STAGE_B_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --persistent-workers \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --val-ratio 0.05 \
  --seed 42 \
  --device cuda \
  --adapter-bottleneck 64 \
  --adapter-dropout 0.1 \
  --lambda-ce 1.0 \
  --lambda-dice 1.0 \
  --ignore-index 255 \
  --crack-labels crack \
  --ignore-labels ignore \
  --component-labels component,wood \
  --ncp-label ncp \
  --enable-copy-paste \
  --copy-paste-prob 0.7 \
  --min-pastes 1 \
  --max-pastes 3 \
  --cp-search-radius 400 \
  --cp-max-tries 150 \
  --cp-min-crack-area 30 \
  --cp-bbox-padding 16 \
  --cp-max-rotate-deg 10.0 \
  --cp-scale-min 0.9 \
  --cp-scale-max 1.1 \
  --log-every 10 \
  --labelme-sliding-window \
  --window-stride "$WINDOW_STRIDE" \
  --positive-patch-ratio 0.6 \
  "${EXTRA_ARGS[@]}" \
  "$@"
