#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

need_dir "$LABELME_TEST_DIR"
need_dir "$DINO_WEIGHTS"
need_file "$STAGE_B_FULL_CKPT"

OUT_DIR="${EVAL_OUT_DIR:-$RUNS_ROOT/eval/test}"
mkdir -p "$OUT_DIR"

ARGS=(
  --labelme-dir "$LABELME_TEST_DIR"
  --output-dir "$OUT_DIR"
  --stage-b-ckpt "$STAGE_B_FULL_CKPT"
  --teacher-weights "$DINO_WEIGHTS"
)

for exp in S0 S1 S2 S3 S4; do
  ckpt="$PAPER_RUN_DIR/$exp/best.pt"
  if [[ -f "$ckpt" ]]; then
    ARGS+=(--student-ckpt "$exp=$ckpt")
  fi
done

python_run "$ROOT/solution/scripts/eval_labelme_crack_test.py" "${ARGS[@]}"
