#!/usr/bin/env bash
# Wait for the three fold0 repo baselines (AttUNet/CMUNet/MACUNet) to finish +
# be evaluated, then train the supervised MACUNet baseline on folds 1..4 across
# GPUs 5/6/7 and run calibrated_eval for each fold.
set -u
cd /home/zgt/Segmentation || exit 1

LB=/mnt/volume3/home/zgt/data/labelme/all

no_py_jobs() {
  python - <<'PY'
import os, sys
me = os.getpid(); busy = False
for p in os.listdir('/proc'):
    if not p.isdigit():
        continue
    pid = int(p)
    if pid == me:
        continue
    try:
        parts = open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore').split('\x00')
    except Exception:
        continue
    if not parts or 'python' not in os.path.basename(parts[0]):
        continue
    cl = ' '.join(parts)
    if ('train_seg_stage_c_mixed.py' in cl) or ('calibrated_eval.py' in cl):
        busy = True
        break
sys.exit(1 if busy else 0)
PY
}

echo "[macunet-kfold] waiting for fold0 repo baselines + evaluation to finish ..."
while true; do
  ok=1
  for n in AttUNet_base CMUNet_base MACUNet_base; do
    [ -f "runs/kfold/fold0/$n/.completed" ] || ok=0
  done
  if [ "$ok" = "1" ] && no_py_jobs; then break; fi
  sleep 300
done
echo "[macunet-kfold] fold0 done; launching MACUNet folds 1-4"

run_fold() {
  local f="$1" gpu="$2" d="runs/kfold/fold$1/MACUNet_base"
  if [ -f "$d/.completed" ]; then echo "[macunet-kfold] fold$f exists"; return; fi
  mkdir -p "$d"
  echo "[macunet-kfold] train fold$f on GPU $gpu"
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
  echo "[macunet-kfold] eval fold$f"
  CUDA_VISIBLE_DEVICES="$gpu" python solution/scripts/calibrated_eval.py \
    --model student --ckpt "$d/best.pt" --folds runs/kfold/folds.json --fold "$f" \
    --val-ratio 0.1 --seed 42 --val-stride 512 --device cuda \
    --out "$d/calibrated_eval.json" > "$d/eval.log" 2>&1
  echo "[macunet-kfold] fold$f done"
}

# GPU 5: folds 1 then 4 ; GPU 6: fold 2 ; GPU 7: fold 3
( run_fold 1 5; run_fold 4 5 ) > runs/kfold/reports/macunet_gpu5.log 2>&1 &
( run_fold 2 6 )            > runs/kfold/reports/macunet_gpu6.log 2>&1 &
( run_fold 3 7 )            > runs/kfold/reports/macunet_gpu7.log 2>&1 &
wait
echo "[macunet-kfold] all folds done"
python solution/scripts/summarize_macunet_kfold.py --out runs/kfold/reports/macunet_kfold.md || true
echo "[macunet-kfold] finished"
