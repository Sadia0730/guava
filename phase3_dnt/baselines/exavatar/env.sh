# Source this before any ExAvatar setup / run step. Everything heavy lives under /dev/shm (RAM-backed).
export EXA_ROOT=/dev/shm/ubx858_baselines/exavatar
export EXA_SRC=$EXA_ROOT/src/ExAvatar_RELEASE
export EXA_ENV=$EXA_ROOT/envs/exavatar          # avatar training + SMPL-X/FLAME fitting + DECA + Depth-Anything-V2
export EXA_ENV_POSE=$EXA_ROOT/envs/exavatar_pose # mmpose / mmdet (whole-body 2D keypoints; needs mmcv 2.1 -> torch 2.1)
export PIP_CACHE_DIR=$EXA_ROOT/cache/pip
export CONDA_PKGS_DIRS=$EXA_ROOT/cache/conda_pkgs
export TORCH_HOME=$EXA_ROOT/cache/torch
export HF_HOME=$EXA_ROOT/cache/hf
export XDG_CACHE_HOME=$EXA_ROOT/cache/xdg
export MPLCONFIGDIR=$EXA_ROOT/cache/mpl
export CUDA_CACHE_PATH=$EXA_ROOT/cache/nv
export TORCH_EXTENSIONS_DIR=$EXA_ROOT/cache/torch_ext
export TMPDIR=$EXA_ROOT/tmp
export PYTHONNOUSERSITE=1
mkdir -p $PIP_CACHE_DIR $CONDA_PKGS_DIRS $TORCH_HOME $HF_HOME $XDG_CACHE_HOME $MPLCONFIGDIR $CUDA_CACHE_PATH $TORCH_EXTENSIONS_DIR $TMPDIR
