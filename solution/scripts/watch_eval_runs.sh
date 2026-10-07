#!/usr/bin/env bash
# Generic: wait for named Stage-C runs to finish, run calibrated_eval on each,
# then summarize into one markdown table (with reference rows).
#
# Usage: bash solution/scripts/watch_eval_runs.sh <EVAL_GPU> <report_name> <run1> [run2 ...]
set -u
cd /home/zgt/Segmentation || exit 1
EVAL_GPU="${1:-6}"; REPORT="${2:-calibrated_table_generic.md}"; shift 2
RUNS="$*"
ROOT="runs/kfold/fold0"
mkdir -p "$ROOT/reports"
LOG="$ROOT/reports/${REPORT%.md}.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_run() {
  local r="$1" d="$ROOT/$1"
  while true; do
    if [ -f "$d/.completed" ]; then log "$r completed"; return 0; fi
    if ! pgrep -f "output-dir $d" >/dev/null 2>&1; then
      sleep 30
      if ! pgrep -f "output-dir $d" >/dev/null 2>&1; then log "$r stopped without .completed"; return 1; fi
    fi
    sleep 300
  done
}

log "watching: $RUNS (eval on GPU $EVAL_GPU)"
for r in $RUNS; do wait_run "$r" || true; done
log "runs settled; evaluating"

INPUTS=()
for r in $RUNS; do
  d="$ROOT/$r"
  [ -f "$d/best.pt" ] || { log "skip $r (no best.pt)"; continue; }
  log "eval $r ..."
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" \
    --folds runs/kfold/folds.json --fold 0 --val-ratio 0.1 --seed 42 \
    --val-stride 512 --device cuda --out "$d/calibrated_eval.json" >> "$d/eval.log" 2>&1
  [ -f "$d/calibrated_eval.json" ] && INPUTS+=("$d/calibrated_eval.json")
done
for ref in abl_fpn_inst_l1 stage_b_adapted SMGNet_paper640; do
  f="$ROOT/$ref/calibrated_eval.json"; [ -f "$f" ] && INPUTS+=("$f")
done

log "summarizing ${#INPUTS[@]} results"
python solution/scripts/summarize_calibrated_eval.py --inputs "${INPUTS[@]}" \
  --out "$ROOT/reports/$REPORT" >> "$LOG" 2>&1
log "done -> $ROOT/reports/$REPORT"
