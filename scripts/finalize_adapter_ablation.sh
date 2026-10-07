#!/usr/bin/env bash
# Wait until all raw/adapted Stage-B teachers exist and no teacher training is
# running, then evaluate raw vs adapted across folds and write the report.
#
# Usage: bash scripts/finalize_adapter_ablation.sh <GPU>
set -u
cd /home/zgt/Segmentation || exit 1
GPU="${1:-6}"

while true; do
  ok=1
  for f in 0 1 2 3 4; do
    for v in raw adapted; do
      [ -f "runs/kfold/fold$f/stage_b_$v/best.pt" ] || ok=0
    done
  done
  if [ "$ok" = "1" ] && ! pgrep -f "train_teacher_stage_b.py" >/dev/null 2>&1; then
    break
  fi
  sleep 300
done
echo "[finalize-adapter] all teachers ready; evaluating on GPU $GPU"
for f in 0 1 2 3 4; do
  for v in raw adapted; do
    ck="runs/kfold/fold$f/stage_b_$v/best.pt"
    [ -f "$ck" ] || continue
    echo "[finalize-adapter] eval fold$f $v"
    CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
      --model teacher_stage_b --ckpt "$ck" --folds runs/kfold/folds.json --fold "$f" \
      --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
      --out "runs/kfold/fold$f/stage_b_$v/calibrated_eval.json" \
      > "runs/kfold/fold$f/stage_b_$v/eval.log" 2>&1
  done
done
python solution/scripts/summarize_adapter_ablation.py \
  --out runs/kfold/reports/adapter_ablation.md
echo "[finalize-adapter] done"
