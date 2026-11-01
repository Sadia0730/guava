#!/usr/bin/env bash
# Run GART on one AvatarTest-v1 clip: bash run_clip.sh <dataset> <clip> [extra run_gart.py args, e.g. --sanity 8]
# GPU: CUDA_VISIBLE_DEVICES if set, else the first of GPUs 0-3 (then 4-7) with < 2 GB in use (checked by run_gart.py).
set -euo pipefail
ROOT=/dev/shm/ubx858_baselines/gart
export PIP_CACHE_DIR=$ROOT/cache/pip CONDA_PKGS_DIRS=$ROOT/cache/conda_pkgs TORCH_HOME=$ROOT/cache/torch HF_HOME=$ROOT/cache/hf \
       XDG_CACHE_HOME=$ROOT/cache/xdg TMPDIR=$ROOT/cache/tmp MPLCONFIGDIR=$ROOT/cache/mpl TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_ext \
       PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
HERE=$(cd "$(dirname "$0")" && pwd)
exec $ROOT/env/bin/python -u $HERE/run_gart.py --dataset "$1" --clip "$2" "${@:3}"
