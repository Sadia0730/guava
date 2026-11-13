#!/usr/bin/env bash
# One fixed clip by default. Models and their precision stay as evaluated.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $0 SUBSET_ROOT OUT_DIR [timing options]"; exit 2; }
SUBSET=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL="${IDOL_INSTALL:-${BASELINE_ROOT:-/dev/shm/ubx858_baselines}/idol}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$INSTALL/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export IDOL_CODE="${IDOL_CODE:-$INSTALL/src}"
export U2NET_HOME="$INSTALL/cache"
mkdir -p "$OUT"
"${IDOL_PY:-python}" "$HERE/time_idol.py" --root "$SUBSET" --out "$OUT/idol.json" --clips neuman/bike --warmup 10 --iters 50 "$@" 2>&1 | tee "$OUT/idol_run.log"
