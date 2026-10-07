#!/usr/bin/env bash
# MACUNet + feature-KD across folds (K=5). fold0 is already running; this driver
# covers folds 1-4 with three parallel lanes, then evaluates every fold and
# aggregates.
#   GPU7: fold2 -> fold3
#   GPU5: wait fold1/MACUNet_base -> fold1
#   GPU6: wait fold0/MACUNet_KD   -> fold4
set -u
cd /home/zgt/Segmentation || exit 1
LB=/mnt/volume3/home/zgt/data/labelme/all

run_fold() {
  local f="$1" gpu="$2" d="runs/kfold/fold$1/MACUNet_KD"
  if [ -f "$d/.completed" ]; then echo "[macunet-kd] fold$f exists"; fi
  if [ ! -f "$d/.completed" ]; then
    mkdir -p "$d"
    echo "[macunet-kd] train fold$f on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python solution/train_seg_stage_c_mixed.py \
      --curated-labelme-dir "$LB" --images-dir "$LB" \
      --student-arch macunet_kd --student-scratch \
      --teacher-mode stage_b --teacher-stage-b-ckpt "runs/kfold/fold$f/stage_b_adapted/best.pt" \
      --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
      --fold-split runs/kfold/folds.json --fold "$f" \
      --imgsz 1024 --teacher-img-size 1024 --window-stride 800 \
      --batch-size-curated 2 --epochs 100 --early-stop-patience 10 \
      --num-workers 4 --persistent-workers --prefetch-factor 4 \
      --val-stride 512 --val-ratio 0.1 --seed 42 \
      --align-dim 256 --feat-norm inst --decoder-channels 256,192,128,64 \
      --lambda-feat-curated 1.0 --lambda-attn-curated 0.2 \
      --lambda-l1 0.5 --lambda-l2 0.3 --lambda-l3 0.2 --attn-crack-gamma 3.0 --no-amp \
      --lr 1e-4 --weight-decay 1e-4 \
      --crack-labels crack --ignore-labels ignore --component-labels component,wood --ncp-labels ncp \
      --device cuda --log-every 40 --output-dir "$d" > "$d/stdout.log" 2>&1
  fi
  if [ ! -f "$d/calibrated_eval.json" ]; then
    echo "[macunet-kd] eval fold$f"
    CUDA_VISIBLE_DEVICES="$gpu" python solution/scripts/calibrated_eval.py \
      --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
      --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
      --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
  fi
  echo "[macunet-kd] fold$f done"
}

( run_fold 2 7; run_fold 3 7 ) &
( while [ ! -f runs/kfold/fold1/MACUNet_base/.completed ]; do sleep 120; done; run_fold 1 5 ) &
( while [ ! -f runs/kfold/fold0/MACUNet_KD/.completed ]; do sleep 120; done; run_fold 4 6 ) &
wait
echo "[macunet-kd] folds 1-4 done; evaluating missing folds + aggregate"
for f in 0 1 2 3 4; do
  d="runs/kfold/fold$f/MACUNet_KD"
  [ -f "$d/best.pt" ] || continue
  [ -f "$d/calibrated_eval.json" ] && continue
  CUDA_VISIBLE_DEVICES=7 python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
done
python solution/scripts/summarize_macunet_kfold.py --run MACUNet_KD \
  --out runs/kfold/reports/macunet_kd_kfold.md || true
python solution/scripts/master_table.py --out runs/kfold/fold0/reports/master_table_all.md || true
echo "[macunet-kd] finished"
