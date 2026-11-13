#!/usr/bin/env bash
# One fixed clip by default. Models and their precision stay as evaluated.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $0 SUBSET_ROOT OUT_DIR [timing options]"; exit 2; }
SUBSET=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL="${LHMPP_INSTALL:-${BASELINE_ROOT:-/dev/shm/ubx858_baselines}/lhmpp}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$OUT"
"${LHMPP_PYTHON:-python}" "$HERE/time_lhmpp.py" --root "$SUBSET" --out "$OUT/lhmpp.json" --clips neuman/bike --lhm "${LHMPP_CODE:?set LHMPP_CODE}" --pretrained "${LHMPP_PRETRAINED:?set LHMPP_PRETRAINED}" --warmup 10 --iters 50 "$@" 2>&1 | tee "$OUT/lhmpp_run.log"
