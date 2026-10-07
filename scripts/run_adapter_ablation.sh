#!/usr/bin/env bash
# Adapter ablation: train the Stage-B teacher WITH Stage-A adapters (adapted) vs
# WITHOUT (raw) on each fold, then evaluate both with the unified calibrated
# whole-image protocol, so raw-vs-adapted can be compared across folds.
#
# Usage: bash scripts/run_adapter_ablation.sh <GPU> [fold ...]   (default: 1 2 3 4)
set -u
cd /home/zgt/Segmentation || exit 1
GPU="$1"; shift
FOLDS="${*:-1 2 3 4}"
LB=/mnt/volume3/home/zgt/data/labelme/all
TWEIGHTS=weights/dinov3-vitb16-pretrain-lvd1689m
SA=runs/stage_a/stage_a_last.pt

train_raw() {
  local f="$1" d="runs/kfold/fold$1/stage_b_raw"
  if [ -f "$d/best.pt" ]; then echo "[adapter] raw teacher fold$f exists"; return; fi
  mkdir -p "$d"
  echo "[adapter] training RAW teacher fold$f on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/train_teacher_stage_b.py \
    --labelme-dir "$LB" --images-dir "$LB" --teacher-weights "$TWEIGHTS" \
    --output-dir "$d" --num-classes 2 --imgsz 1024 --epochs 100 --batch-size 2 \
    --num-workers 4 --persistent-workers --prefetch-factor 4 \
    --lr 1e-4 --weight-decay 1e-4 --val-ratio 0.1 --seed 42 \
    --fold-split runs/kfold/folds.json --fold "$f" \
    --early-stop-patience 10 --val-stride 512 --device cuda \
    --adapter-bottleneck 64 --adapter-dropout 0.1 \
    --lambda-ce 1.0 --lambda-dice 1.0 --ignore-index 255 \
    --crack-labels crack --ignore-labels ignore --component-labels component,wood --ncp-label ncp \
    --labelme-sliding-window --window-stride 800 --positive-patch-ratio 0.6 --log-every 10 \
    --no-adapters > "$d/train.log" 2>&1
}

eval_teacher() {
  local f="$1" v="$2" ck="runs/kfold/fold$1/stage_b_$2/best.pt"
  [ -f "$ck" ] || { echo "[adapter] no ckpt $ck"; return; }
  echo "[adapter] eval fold$f $v on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model teacher_stage_b --ckpt "$ck" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "runs/kfold/fold$f/stage_b_$v/calibrated_eval.json" > "runs/kfold/fold$f/stage_b_$v/eval.log" 2>&1
}

for f in $FOLDS; do
  train_raw "$f"
done

echo "[adapter] evaluating raw vs adapted across folds 0..4"
for f in 0 1 2 3 4; do
  eval_teacher "$f" raw
  eval_teacher "$f" adapted
done
echo "[adapter] done"
