#!/usr/bin/env bash
# Resume (continue) our baseline feature-KD student on the given folds, then
# re-run calibrated_eval for each. Same args as the original baseline runs.
# Usage: bash scripts/resume_baseline_folds.sh <GPU> <fold...>
set -u
cd /home/zgt/Segmentation || exit 1
GPU="$1"; shift
FOLDS="$*"
LB=/mnt/volume3/home/zgt/data/labelme/all

for f in $FOLDS; do
  d="runs/kfold/fold$f/baseline"
  [ -f "$d/last.pt" ] || { echo "[resume-baseline] no last.pt for fold$f"; continue; }
  echo "[resume-baseline] fold$f on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/train_seg_stage_c_mixed.py \
    --curated-labelme-dir "$LB" --images-dir "$LB" \
    --student-weights weights/yolo11m-seg.pt --student-arch yolo_unet --student-scratch \
    --student-feat-source neck \
    --teacher-mode stage_b --teacher-stage-b-ckpt "runs/kfold/fold$f/stage_b_adapted/best.pt" \
    --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
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
    --device cuda --log-every 40 --output-dir "$d" --resume "$d/last.pt" --reset-patience \
    > "$d/stdout_resume.log" 2>&1
  echo "[resume-baseline] eval fold$f"
  CUDA_VISIBLE_DEVICES="$GPU" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval_resume.log" 2>&1
  echo "[resume-baseline] fold$f done"
done
python solution/scripts/summarize_kfold_variants.py --out runs/kfold/reports/generalization_baseline_vs_e1a.md || true
echo "[resume-baseline] finished"
