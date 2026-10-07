#!/usr/bin/env bash
# Evaluate the non-distilled supervised student S0 (runs_bk/kfold/fold*/S0) with
# the unified calibrated whole-image protocol, one fold at a time.
# Usage: bash scripts/eval_S0_kfold.sh <GPU>
set -u
cd /home/zgt/Segmentation || exit 1
GPU="${1:-7}"
for f in 0 1 2 3 4; do
  d="runs_bk/kfold/fold$f/S0"
  [ -f "$d/best.pt" ] || { echo "[eval-S0] fold$f no ckpt"; continue; }
  echo "[eval-S0] fold$f on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
done
python solution/scripts/summarize_distill_ablation.py \
  --out runs/kfold/reports/distill_ablation_S0_vs_KD.md || true
echo "[eval-S0] finished"
