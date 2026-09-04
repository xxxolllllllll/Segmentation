#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

"$PYTHON_BIN" -m pip install --user --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --user torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_BIN" -m pip install --user -r "$ROOT/requirements-linux.txt"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
