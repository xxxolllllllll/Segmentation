#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP="${1:-}"
if [[ -z "$EXP" ]]; then
  echo "Usage: bash scripts/run_stage_c_suite.sh [S0|S1|S2|S3|S4]" >&2
  exit 1
fi
shift

need_dir "$LABELME_TRAINVAL_DIR"
need_dir "$DINO_WEIGHTS"
need_file "$YOLO_WEIGHTS"

LABELME_DIR="$LABELME_TRAINVAL_DIR"
TEACHER_MODE="stage_b"
TEACHER_ARGS=()
LAMBDA_FEAT="0.5"
LAMBDA_ATTN="0.0"
COPY_PASTE="0.0"
BATCH_SIZE="2"
EPOCHS="25"
OUT_DIR="$PAPER_RUN_DIR/$EXP"

case "$EXP" in
  S0)
    need_dir "$LABELME_CURATED_DIR"
    LABELME_DIR="$LABELME_CURATED_DIR"
    LAMBDA_FEAT="0.0"
    LAMBDA_ATTN="0.0"
    COPY_PASTE="0.0"
    BATCH_SIZE="$S0_BATCH_SIZE"
    EPOCHS="$S0_EPOCHS"
    TEACHER_MODE="raw_vit"
    ;;
  S1)
    LABELME_DIR="$LABELME_TRAINVAL_DIR"
    TEACHER_MODE="raw_vit"
    TEACHER_ARGS+=(--teacher-weights "$DINO_WEIGHTS")
    BATCH_SIZE="$S1_BATCH_SIZE"
    EPOCHS="$S1_EPOCHS"
    ;;
  S2)
    LABELME_DIR="$LABELME_TRAINVAL_DIR"
    need_file "$STAGE_B_FULL_CKPT"
    TEACHER_ARGS+=(--teacher-stage-b-ckpt "$STAGE_B_FULL_CKPT" --teacher-weights "$DINO_WEIGHTS")
    BATCH_SIZE="$S2_BATCH_SIZE"
    EPOCHS="$S2_EPOCHS"
    ;;
  S3)
    need_dir "$LABELME_CURATED_DIR"
    LABELME_DIR="$LABELME_CURATED_DIR"
    need_file "$STAGE_B_FULL_CKPT"
    TEACHER_ARGS+=(--teacher-stage-b-ckpt "$STAGE_B_FULL_CKPT" --teacher-weights "$DINO_WEIGHTS")
    LAMBDA_ATTN="0.2"
    BATCH_SIZE="$S3_BATCH_SIZE"
    EPOCHS="$S3_EPOCHS"
    ;;
  S4)
    need_dir "$LABELME_CURATED_DIR"
    LABELME_DIR="$LABELME_CURATED_DIR"
    need_file "$STAGE_B_FULL_CKPT"
    TEACHER_ARGS+=(--teacher-stage-b-ckpt "$STAGE_B_FULL_CKPT" --teacher-weights "$DINO_WEIGHTS")
    LAMBDA_ATTN="0.2"
    COPY_PASTE="$S4_COPY_PASTE_PROB"
    BATCH_SIZE="$S4_BATCH_SIZE"
    EPOCHS="$S4_EPOCHS"
    ;;
  *)
    echo "Unsupported experiment: $EXP" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

python_run "$ROOT/solution/train_seg_stage_c_mixed.py" \
  --curated-labelme-dir "$LABELME_DIR" \
  --student-weights "$YOLO_WEIGHTS" \
  --teacher-mode "$TEACHER_MODE" \
  --output-dir "$OUT_DIR" \
  --num-classes 2 \
  --imgsz "$IMGSZ" \
  --teacher-img-size "$IMGSZ" \
  --window-stride "$WINDOW_STRIDE" \
  --batch-size-curated "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  --persistent-workers \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --val-ratio 0.05 \
  --seed 42 \
  --device cuda \
  --lambda-ce 1.0 \
  --lambda-dice 1.0 \
  --lambda-feat-curated "$LAMBDA_FEAT" \
  --lambda-attn-curated "$LAMBDA_ATTN" \
  --copy-paste-prob "$COPY_PASTE" \
  --attn-crack-gamma 3.0 \
  --decoder-channels 256,192,128,64 \
  --log-every 10 \
  "${TEACHER_ARGS[@]}" \
  "$@"
