#!/usr/bin/env bash
# Rebuild the GaussianAvatar baseline environment from scratch under /dev/shm (RAM disk).
#   bash phase3_dnt/baselines/gaussianavatar/setup.sh
# Nothing is written to /raid or /home (except this repo folder, which is not touched by this script).
# Official code: https://github.com/huliangxiao/GaussianAvatar @ d981c62238ef64e89dcc04719d2ebbb4758b080a
set -euo pipefail

BASE=${GA_BASE:-/dev/shm/ubx858_baselines/gaussianavatar}
GA_COMMIT=d981c62238ef64e89dcc04719d2ebbb4758b080a
# 3DGS rasterizer as used by gaussian-splatting before its Oct-2024 update (returns (color, radii), the
# 2-output API GaussianAvatar's gaussian_renderer expects) and simple-knn as pinned by gaussian-splatting.
DGR_COMMIT=59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
SKNN_COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
SMPL_DIR=/home/ubx858/ml-hugs/data/smpl            # neutral SMPL v1.1.0 python release (read only, copied)
ONEDRIVE=https://hiteducn0-my.sharepoint.com/personal/lx_hu_hit_edu_cn
ONEDRIVE_SHARE="https://hiteducn0-my.sharepoint.com/:f:/g/personal/lx_hu_hit_edu_cn/EsGcL5JGKhVGnaAtJ-rb1sQBR4MwkdJ9EWqJBIdd2mpi2w?e=KnloBM"
ASSETS_SHA256=caec1c551722b8ba5b5ea28a436b479e4fd3b7f581d8e625baa2351bb3bb7588

# ---- all caches under BASE, set before anything is installed ----
mkdir -p "$BASE"/{cache/pip,cache/conda_pkgs,cache/torch,cache/hf,cache/xdg,src,work,tmp,downloads}
export PIP_CACHE_DIR=$BASE/cache/pip CONDA_PKGS_DIRS=$BASE/cache/conda_pkgs TORCH_HOME=$BASE/cache/torch \
       HF_HOME=$BASE/cache/hf XDG_CACHE_HOME=$BASE/cache/xdg TMPDIR=$BASE/tmp PYTHONNOUSERSITE=1
CONDA=/home/ubx858/miniconda3/bin/conda
ENV=$BASE/env

# ---- 1. official code ----
REPO=$BASE/src/GaussianAvatar
if [ ! -d "$REPO/.git" ]; then git clone https://github.com/huliangxiao/GaussianAvatar.git "$REPO"; fi
git -C "$REPO" checkout -q $GA_COMMIT
git -C "$REPO" rev-parse HEAD

# ---- 2. conda env from the repository's environment.yml (python 3.10.13, pytorch 2.0.1, cuda 11.8) ----
# The yml leaves numpy / mkl / opencv unpinned; resolved today that gives numpy 2.x and mkl 2025 (pytorch 2.0.1 then
# fails with "undefined symbol: iJIT_NotifyEvent") and opencv 5 (numpy 2). Pin them to versions of the code's time
# (numpy 1.23.5 is also what chumpy, needed to unpickle the official SMPL .pkl, still supports). Nothing else changes.
PINNED=$BASE/tmp/environment_pinned.yml
sed -e 's/^  - scipy$/  - scipy\n  - numpy=1.23.5\n  - mkl=2023.1.0/' -e 's/^    - opencv-python$/    - opencv-python==4.8.1.78/' \
    "$REPO/environment.yml" > "$PINNED"
grep -q "mkl=2023.1.0" "$PINNED" && grep -q "opencv-python==4.8.1.78" "$PINNED"
if [ ! -x "$ENV/bin/python" ]; then
  $CONDA env create -p "$ENV" -f "$PINNED"
fi
PY=$ENV/bin/python
PIP="$PY -m pip"
# CUDA 11.8 compiler + headers (matching pytorch-cuda=11.8) to build the rasterizer; freeglut for the official
# (GLUT) position-map renderer, run under the system Xvfb.
if [ ! -x "$ENV/bin/nvcc" ]; then
  $CONDA install -y -p "$ENV" -c "nvidia/label/cuda-11.8.0" cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl
fi
if ! ls "$ENV"/lib/libglut.so* >/dev/null 2>&1; then
  $CONDA install -y -p "$ENV" -c conda-forge --no-update-deps freeglut
fi

# ---- 3. python packages imported by the code but not listed in environment.yml ----
$PIP install "numpy==1.23.5" "lpips==0.1.4" "open3d==0.18.0" "trimesh==4.0.10" "torchmetrics==1.2.1" \
     "PyOpenGL==3.1.7" "smplx==0.1.28" "plyfile==1.0.3" "opencv-python==4.8.1.78" "tensorboard" \
     "imageio==2.33.1" "pillow==10.1.0"
$PIP install --no-build-isolation "chumpy==0.70"   # needed to unpickle the official SMPL v1.1.0 .pkl
# pytorch3d (only pytorch3d.transforms is used, in arguments/__init__.py): official prebuilt wheel py310/cu118/pt201
$PIP install "fvcore==0.1.5.post20221221" "iopath==0.1.10"
$PIP install --no-index --no-deps --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html "pytorch3d==0.7.4"

# ---- 4. 3DGS CUDA extensions (README: compile diff-gaussian-rasterization and simple-knn as in 3DGS) ----
export CUDA_HOME=$ENV TORCH_CUDA_ARCH_LIST="8.9" PATH=$ENV/bin:$PATH
mkdir -p "$BASE/src/ext"
if [ ! -d "$BASE/src/ext/diff-gaussian-rasterization" ]; then
  git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$BASE/src/ext/diff-gaussian-rasterization"
fi
git -C "$BASE/src/ext/diff-gaussian-rasterization" checkout -q $DGR_COMMIT
git -C "$BASE/src/ext/diff-gaussian-rasterization" submodule update --init --recursive
if [ ! -d "$BASE/src/ext/simple-knn" ]; then
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git "$BASE/src/ext/simple-knn"
fi
git -C "$BASE/src/ext/simple-knn" checkout -q $SKNN_COMMIT
$PIP install --no-build-isolation "$BASE/src/ext/diff-gaussian-rasterization" "$BASE/src/ext/simple-knn"

# ---- 5. official assets (OneDrive assets.zip, 17 MB) unzipped into the repository ----
AZ=$BASE/downloads/assets_zip/assets.zip
if [ ! -f "$AZ" ]; then
  mkdir -p "$(dirname "$AZ")"
  CJ=$BASE/tmp/onedrive_cookies.txt; rm -f "$CJ"
  curl -sS -c "$CJ" -b "$CJ" -L -o /dev/null "$ONEDRIVE_SHARE"        # anonymous share link -> session cookie
  curl -sS -c "$CJ" -b "$CJ" -L -o "$AZ" \
    "$ONEDRIVE/_api/web/GetFileByServerRelativeUrl('/personal/lx_hu_hit_edu_cn/Documents/GaussianAvatar/assets.zip')/\$value"
  rm -f "$CJ"
fi
echo "$ASSETS_SHA256  $AZ" | sha256sum -c -
$PY -I - "$AZ" "$REPO/assets" <<'EOF'
import sys, zipfile, os
zf, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zf)
for n in z.namelist():
    assert not n.startswith('/') and '..' not in n.split('/'), n
z.extractall(out)
print('assets:', len(z.namelist()), 'entries ->', out)
EOF

# ---- 6. SMPL (copies, unmodified) ----
mkdir -p "$REPO/assets/smpl_files/smpl"
for g in NEUTRAL MALE FEMALE; do
  [ -f "$REPO/assets/smpl_files/smpl/SMPL_$g.pkl" ] || cp "$SMPL_DIR/SMPL_$g.pkl" "$REPO/assets/smpl_files/smpl/SMPL_$g.pkl"
done

# ---- 7. smoke test ----
$PY - <<'EOF'
import torch, diff_gaussian_rasterization, simple_knn, pytorch3d, lpips, open3d, trimesh, smplx, chumpy, OpenGL
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
EOF
echo "setup done: env=$ENV repo=$REPO"
