#!/usr/bin/env bash
# MACUNet supervised baseline across folds 1..4.
# GPUs 6 and 7 start immediately (fold2, fold3); GPU 5 waits for the fold0
# AttUNet run to finish, then does fold1 and fold4.
set -u
cd /home/zgt/Segmentation || exit 1
LB=/mnt/volume3/home/zgt/data/labelme/all

run_fold() {
  local f="$1" gpu="$2" d="runs/kfold/fold$1/MACUNet_base"
  if [ -f "$d/.completed" ]; then echo "[macunet] fold$f exists"; return; fi
  mkdir -p "$d"
  echo "[macunet] train fold$f on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python solution/train_seg_stage_c_mixed.py \
    --curated-labelme-dir "$LB" --images-dir "$LB" \
    --student-scratch --student-arch macunet \
    --fold-split runs/kfold/folds.json --fold "$f" \
    --num-classes 2 --imgsz 1024 --teacher-img-size 1024 --window-stride 800 \
    --batch-size-curated 2 --epochs 100 --early-stop-patience 10 \
    --num-workers 4 --persistent-workers --prefetch-factor 4 \
    --val-stride 512 --val-ratio 0.1 --seed 42 \
    --lambda-ce 1.0 --lambda-dice 1.0 --lambda-feat-curated 0.0 --lambda-attn-curated 0.0 \
    --lr 1e-4 --weight-decay 1e-4 --no-amp \
    --crack-labels crack --ignore-labels ignore --component-labels component,wood --ncp-labels ncp \
    --device cuda --log-every 40 --output-dir "$d" > "$d/stdout.log" 2>&1
  echo "[macunet] eval fold$f"
  CUDA_VISIBLE_DEVICES="$gpu" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
  echo "[macunet] fold$f done"
}

( run_fold 2 6 ) &
( run_fold 3 7 ) &
( while [ ! -f runs/kfold/fold0/AttUNet_base/.completed ]; do sleep 120; done
  run_fold 1 5; run_fold 4 5 ) &
wait
echo "[macunet] folds 1-4 done"
python solution/scripts/summarize_macunet_kfold.py --out runs/kfold/reports/macunet_kfold.md || true
echo "[macunet] finished"
