#!/usr/bin/env bash
# One fixed clip by default. Models and their precision stay as evaluated.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $0 SUBSET_ROOT OUT_DIR [timing options]"; exit 2; }
SUBSET=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSTALL="${EXA_ROOT_ORIG:-${BASELINE_ROOT:-/dev/shm/ubx858_baselines}/exavatar}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export EXA_ROOT_ORIG="$INSTALL"
export PATH="$INSTALL/envs/exavatar/bin:$PATH"
mkdir -p "$OUT"
"$INSTALL/envs/exavatar/bin/python" "$HERE/time_exavatar.py" --root "$SUBSET" --out "$OUT" --clip neuman/bike --keep-work --warmup 10 --iters 50 "$@" 2>&1 | tee "$OUT/exavatar_run.log"
