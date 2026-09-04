#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python_run "$ROOT/scripts/run_stagec_repeat_stability.py"
