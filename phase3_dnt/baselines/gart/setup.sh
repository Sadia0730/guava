#!/usr/bin/env bash
# Rebuild the GART baseline environment from scratch under /dev/shm (AvatarTest-v1 addendum 1).
# Follows GART's install.sh (python 3.9, pytorch 2.0.0 + CUDA 11.8, pytorch3d from the pytorch3d channel,
# requirements.txt, simple-knn, diff_gaussian_rasterization-alphadep, lpips) with these differences:
#   - prefix env (conda create -p) instead of a named env; every cache under $ROOT/cache
#   - CUDA 11.8 nvcc / headers installed into the env (the machine's nvcc is 12.8; torch 2.0.0 is cu118)
#   - MVDream (lib_guidance, Text2GART generation only) is not installed; it is not used for video fitting
#   - neutral SMPL copied into code/data/smpl_model (read-only source in /home/ubx858/ml-hugs)
#   - chumpy (in requirements.txt) built with --no-build-isolation first (its setup.py imports pip)
#   - BITE dog data (lib_gart/smal/smal_data, dogs only) not downloaded; run_gart.py registers a placeholder for the
#     unconditionally imported smal.smal_tpg module instead
# Built and tested 2026-10-31: torch 2.0.0 (CUDA 11.8), pytorch3d 0.7.7, numpy 1.23.1; env ~3 GB + caches.
set -euo pipefail
ROOT=/dev/shm/ubx858_baselines/gart
GART_COMMIT=16c11f8a5bb3ae249a9d04dc9d98c316e10f1126
ENV=$ROOT/env
CODE=$ROOT/code
mkdir -p $ROOT/cache/{pip,conda_pkgs,torch,hf,xdg,tmp,mpl,torch_ext}
export PIP_CACHE_DIR=$ROOT/cache/pip
export CONDA_PKGS_DIRS=$ROOT/cache/conda_pkgs
export TORCH_HOME=$ROOT/cache/torch
export HF_HOME=$ROOT/cache/hf
export XDG_CACHE_HOME=$ROOT/cache/xdg
export TMPDIR=$ROOT/cache/tmp
export MPLCONFIGDIR=$ROOT/cache/mpl
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_ext
export PYTHONNOUSERSITE=1
unset PYTHONPATH
CONDA=/home/ubx858/miniconda3/bin/conda

# 1. official code at the recorded commit
if [ ! -d $CODE/.git ]; then
  git clone https://github.com/JiahuiLei/GART.git $CODE
fi
git -C $CODE checkout -q $GART_COMMIT
echo "GART commit: $(git -C $CODE rev-parse HEAD)"

# 2. conda env (install.sh: python=3.9; pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8; fvcore iopath; pytorch3d)
if [ ! -x $ENV/bin/python ]; then
  $CONDA create -y -p $ENV python=3.9
fi
$CONDA install -y -p $ENV pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$CONDA install -y -p $ENV -c fvcore -c iopath -c conda-forge fvcore iopath
$CONDA install -y -p $ENV pytorch3d -c pytorch3d
# nvcc 11.8 + headers for building the two CUDA extensions against torch 2.0.0+cu118
$CONDA install -y -p $ENV -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev

PY=$ENV/bin/python
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"   # L40S (Ada)

# 3. pip requirements (official requirements.txt, unchanged); chumpy's setup.py imports pip, so it is built without
#    build isolation first (otherwise 'No module named pip' in the isolated build env)
$PY -m pip install --no-build-isolation chumpy
$PY -m pip install -r $CODE/requirements.txt
# 4. CUDA extensions shipped in the repo
$PY -m pip install --no-build-isolation $CODE/lib_render/simple-knn
$PY -m pip install --no-build-isolation $CODE/lib_render/diff_gaussian_rasterization-alphadep
$PY -m pip install lpips

# 5. body model (neutral SMPL v1.1.0, 10 betas); sources are read-only
mkdir -p $CODE/data/smpl_model
cp -n /home/ubx858/ml-hugs/data/smpl/SMPL_NEUTRAL.pkl $CODE/data/smpl_model/SMPL_NEUTRAL.pkl

# 6. smoke test
cd $CODE
$PY -c "import torch, pytorch3d, simple_knn._C, diff_gaussian_rasterization; print('torch', torch.__version__, torch.version.cuda, 'pytorch3d', pytorch3d.__version__, 'cuda ok', torch.cuda.is_available())"
$CONDA list -p $ENV --export > $ROOT/conda_list.txt
$PY -m pip freeze > $ROOT/pip_freeze.txt
echo "setup done"
