#!/usr/bin/env bash
# Finalize the fold0 repo baselines: wait for all three to complete, then run
# calibrated_eval (shared GPU 6 is fine; evals are light) and refresh the
# master table. Safe process check avoids the earlier pgrep self-match bug.
set -u
cd /home/zgt/Segmentation || exit 1
GPU="${1:-6}"

echo "[finalize-repo] waiting for fold0 AttUNet/CMUNet/MACUNet to complete ..."
while true; do
  ok=1
  for n in AttUNet_base CMUNet_base MACUNet_base; do
    [ -f "runs/kfold/fold0/$n/.completed" ] || ok=0
  done
  [ "$ok" = "1" ] && break
  sleep 120
done
echo "[finalize-repo] all done; evaluating on GPU $GPU"

for n in AttUNet_base CMUNet_base MACUNet_base; do
  ck="runs/kfold/fold0/$n/best.pt"
  [ -f "$ck" ] || { echo "[finalize-repo] skip $n (no ckpt)"; continue; }
  echo "[finalize-repo] eval $n"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$ck" --folds runs/kfold/folds.json --fold 0 \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "runs/kfold/fold0/$n/calibrated_eval.json" > "runs/kfold/fold0/$n/eval.log" 2>&1
done

python solution/scripts/master_table.py --out runs/kfold/fold0/reports/master_table_all.md || true
python solution/scripts/summarize_macunet_kfold.py --out runs/kfold/reports/macunet_kfold.md || true
echo "[finalize-repo] finished"
