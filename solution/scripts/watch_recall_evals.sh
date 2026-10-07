#!/usr/bin/env bash
# Wait for the recall experiments to finish, then run calibrated whole-image
# evaluation (val-tuned threshold -> test) and summarize into one table.
#
# Usage: bash solution/scripts/watch_recall_evals.sh [EVAL_GPU]
set -u
cd /home/zgt/Segmentation || exit 1
EVAL_GPU="${1:-6}"
RUNS="recallB1_cw recallB2_attn recallB4_shallow1"
ROOT="runs/kfold/fold0"
LOG="$ROOT/reports/watch_recall_evals.log"
mkdir -p "$ROOT/reports"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_run() {
  local r="$1" d="$ROOT/$1"
  while true; do
    if [ -f "$d/.completed" ]; then log "$r completed"; return 0; fi
    if ! pgrep -f "output-dir $d" >/dev/null 2>&1; then
      sleep 30
      if ! pgrep -f "output-dir $d" >/dev/null 2>&1; then
        log "$r no longer running and no .completed (crashed/stopped?)"
        return 1
      fi
    fi
    sleep 300
  done
}

log "watching: $RUNS"
for r in $RUNS; do wait_run "$r" || true; done
log "all runs settled; starting evaluations on GPU $EVAL_GPU"

INPUTS=()
for r in $RUNS; do
  d="$ROOT/$r"
  if [ ! -f "$d/best.pt" ]; then log "skip $r (no best.pt)"; continue
  fi
  log "eval $r ..."
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" \
    --folds runs/kfold/folds.json --fold 0 --val-ratio 0.1 --seed 42 \
    --val-stride 512 --device cuda --out "$d/calibrated_eval.json" \
    >> "$d/eval.log" 2>&1
  if [ -f "$d/calibrated_eval.json" ]; then INPUTS+=("$d/calibrated_eval.json"); fi
done

# Reference rows (already computed earlier)
for ref in abl_fpn_inst_l1 stage_b_adapted SMGNet_paper640; do
  f="$ROOT/$ref/calibrated_eval.json"
  if [ -f "$f" ]; then INPUTS+=("$f"); fi
done

log "summarizing ${#INPUTS[@]} results"
python solution/scripts/summarize_calibrated_eval.py \
  --inputs "${INPUTS[@]}" \
  --labels "B1_weightedCE" "B2_attn0.5" "B4_shallow_only" "baseline_abl_fpn_inst_l1" "teacher_stage_b" "SMGNet_paper640" \
  --out "$ROOT/reports/calibrated_table_recall.md" >> "$LOG" 2>&1
log "done -> $ROOT/reports/calibrated_table_recall.md"
