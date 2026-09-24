#!/usr/bin/env bash
# Parallelize k-fold Stage B/C across GPUs: one fold per GPU (k-fold folds are independent).
#
# Stage A must be COMPLETED first (it is shared across folds). This script then runs the
# orchestrator per fold with FOLD_FILTER, each restricted to one GPU via CUDA_VISIBLE_DEVICES,
# and aggregates the reports once at the end.
#
# Usage:
#   GPUS="0 1 2" FOLDS="0 1 2 3 4" EXPERIMENT_FILTER="S1,S2,S3_attn" bash scripts/run_kfold_parallel.sh
#   (EXPERIMENT_FILTER empty = all experiments)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
GPUS="${GPUS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
export EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
export SKIP_AGGREGATE=1
export STAGE_A_NPROC=1
export FOLD_FILTER=""

if [ ! -f runs/stage_a/.completed ]; then
  echo "[parallel] ERROR: runs/stage_a/.completed missing; finish Stage A first." >&2
  exit 1
fi

# Build folds.json once (single writer) before fanning out.
$PY -c "import sys; sys.path.insert(0,'scripts'); import run_kfold_experiments as r; r.build_folds()" || exit 1

gpu_arr=($GPUS)
ngpu=${#gpu_arr[@]}
i=0
pids=()
fail=0

for f in $FOLDS; do
  gpu=${gpu_arr[$((i % ngpu))]}
  i=$((i + 1))
  echo "[parallel] fold=$f gpu=$gpu experiments='${EXPERIMENT_FILTER:-ALL}'"
  CUDA_VISIBLE_DEVICES="$gpu" FOLD_FILTER="$f" $PY scripts/run_kfold_experiments.py &
  pids+=($!)
  if (( i % ngpu == 0 )); then
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    pids=()
  fi
done
for p in "${pids[@]:-}"; do
  if [ -n "${p:-}" ]; then wait "$p" || fail=1; fi
done

# Aggregate all folds' metrics once.
$PY -c "import sys; sys.path.insert(0,'scripts'); import run_kfold_experiments as r; r.aggregate()" || fail=1

if [ "$fail" -ne 0 ]; then
  echo "[parallel] finished WITH errors" >&2
  exit 1
fi
echo "[parallel] all folds done; reports under runs/kfold/reports"
