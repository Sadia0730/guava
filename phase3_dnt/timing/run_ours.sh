#!/usr/bin/env bash
# Default: one fixed clip, unchanged model settings, 50 timed frames.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $0 SUBSET_ROOT OUT_DIR [time_ours.py options]"; exit 2; }
SUBSET=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
CKPT_ARGS=()
if [ -n "${RAGA_FB_CKPT:-}" ]; then CKPT_ARGS+=(--fb-ckpt "$RAGA_FB_CKPT"); fi
if [ -n "${POSE_CKPT:-}" ]; then CKPT_ARGS+=(--sprint-ckpt "$POSE_CKPT"); fi
exec "${GUAVA_PY:-python}" "$HERE/time_ours.py" --root "$SUBSET" --out-dir "$OUT" \
  --assets "${RAGA_ASSETS:-$HERE/../../live_assets}" --clips neuman/bike --warmup 10 --iters 50 \
  --recon-warmup 1 --recon-reps 3 --upd-frames 6 --box center "${CKPT_ARGS[@]}" "$@"
