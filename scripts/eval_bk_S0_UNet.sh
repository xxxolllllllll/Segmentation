#!/usr/bin/env bash
# Uniformly re-evaluate runs_bk S0 (no-distill) and UNet (supervised) with the
# calibrated whole-image protocol, then aggregate S0 / UNet / KD-baseline.
# Usage: bash scripts/eval_bk_S0_UNet.sh <GPU>
set -u
cd /home/zgt/Segmentation || exit 1
GPU="${1:-7}"

eval_run() {  # fold run_dir_label run_dir
  local f="$1" label="$2" d="$3"
  [ -f "$d/best.pt" ] || { echo "[bk-eval] $label fold$f no ckpt"; return; }
  echo "[bk-eval] $label fold$f on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
}

for f in 0 1 2 3 4; do
  eval_run "$f" "S0"   "runs_bk/kfold/fold$f/S0"
  eval_run "$f" "UNet" "runs_bk/kfold/fold$f/UNet"
done

python solution/scripts/summarize_bk_S0_UNet_KD.py \
  --out runs/kfold/reports/bk_S0_UNet_KD.md || true
echo "[bk-eval] finished"
