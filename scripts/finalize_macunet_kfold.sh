#!/usr/bin/env bash
# Finalize MACUNet K=5: wait for folds 1,2,4 to finish, evaluate any fold whose
# calibrated_eval.json is missing, then aggregate mean+-std.
set -u
cd /home/zgt/Segmentation || exit 1
GPU="${1:-7}"

while true; do
  ok=1
  for f in 1 2 4; do
    [ -f "runs/kfold/fold$f/MACUNet_base/.completed" ] || ok=0
  done
  [ "$ok" = "1" ] && break
  sleep 180
done

echo "[finalize-macunet] all folds done; evaluating on GPU $GPU"
for f in 0 1 2 3 4; do
  d="runs/kfold/fold$f/MACUNet_base"
  [ -f "$d/best.pt" ] || continue
  [ -f "$d/calibrated_eval.json" ] && { echo "[finalize-macunet] fold$f already evaluated"; continue; }
  echo "[finalize-macunet] eval fold$f"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
done
python solution/scripts/summarize_macunet_kfold.py --out runs/kfold/reports/macunet_kfold.md || true
python solution/scripts/master_table.py --out runs/kfold/fold0/reports/master_table_all.md || true
echo "[finalize-macunet] finished"
