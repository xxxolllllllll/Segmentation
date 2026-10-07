#!/usr/bin/env bash
# Generalization validation: per fold, train the Stage-B adapted teacher on that
# fold's train split, then baseline vs E1a (scale 0.6-1.25) students, then
# calibrated whole-image evaluation.
#
# Usage: bash scripts/run_generalization_folds.sh <GPU> <fold1> [fold2 ...]
set -u
cd /home/zgt/Segmentation || exit 1
GPU="$1"; shift
FOLDS="$*"
LB=/mnt/volume3/home/zgt/data/labelme/all
TWEIGHTS=weights/dinov3-vitb16-pretrain-lvd1689m
SA=runs/stage_a/stage_a_last.pt

train_teacher() {
  local f="$1" d="runs/kfold/fold$1/stage_b_adapted"
  if [ -f "$d/best.pt" ]; then echo "[gen] teacher fold$f exists"; return; fi
  mkdir -p "$d"
  echo "[gen] training teacher fold$f on GPU $GPU"
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
    --stage-a-ckpt "$SA" > "$d/train.log" 2>&1
}

train_student() {
  local f="$1" name="$2" extra="$3" d="runs/kfold/fold$1/$2"
  if [ -f "$d/.completed" ]; then echo "[gen] $name fold$f exists"; return; fi
  mkdir -p "$d"
  echo "[gen] training $name fold$f on GPU $GPU"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python solution/train_seg_stage_c_mixed.py \
    --curated-labelme-dir "$LB" --images-dir "$LB" \
    --student-weights weights/yolo11m-seg.pt --student-arch yolo_unet --student-scratch \
    --student-feat-source neck \
    --teacher-mode stage_b --teacher-stage-b-ckpt "runs/kfold/fold$f/stage_b_adapted/best.pt" \
    --teacher-weights "$TWEIGHTS" \
    --fold-split runs/kfold/folds.json --fold "$f" \
    --imgsz 1024 --teacher-img-size 1024 --window-stride 800 \
    --batch-size-curated 2 --epochs 100 --early-stop-patience 10 \
    --num-workers 4 --persistent-workers --prefetch-factor 4 \
    --val-stride 512 --val-ratio 0.1 --seed 42 \
    --align-dim 256 --feat-norm inst --decoder-channels 256,192,128,64 \
    --lambda-feat-curated 1.0 --lambda-attn-curated 0.2 \
    --lambda-l1 0.5 --lambda-l2 0.3 --lambda-l3 0.2 --attn-crack-gamma 3.0 \
    --lr 1e-4 --weight-decay 1e-4 \
    --crack-labels crack --ignore-labels ignore --component-labels component,wood --ncp-labels ncp \
    --device cuda --log-every 40 $extra --output-dir "$d" > "$d/stdout.log" 2>&1
}

eval_model() {
  local f="$1" name="$2" ckpt="runs/kfold/fold$1/$2/best.pt"
  [ -f "$ckpt" ] || { echo "[gen] no ckpt $ckpt"; return; }
  echo "[gen] eval $name fold$f on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$ckpt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "runs/kfold/fold$f/$name/calibrated_eval.json" >> "runs/kfold/fold$f/$name/eval.log" 2>&1
}

for f in $FOLDS; do
  echo "[gen] ===== fold $f (GPU $GPU) ====="
  train_teacher "$f"
  train_student "$f" baseline ""
  train_student "$f" scaleAug_06_125 "--scale-aug-min 0.6 --scale-aug-max 1.25"
  eval_model "$f" baseline
  eval_model "$f" scaleAug_06_125
  echo "[gen] fold $f complete"
done
echo "[gen] all requested folds done on GPU $GPU: $FOLDS"
