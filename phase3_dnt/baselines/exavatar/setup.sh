#!/bin/bash
# Rebuild the ExAvatar baseline from scratch under /dev/shm/ubx858_baselines/exavatar (code, 2 conda envs, weights,
# body-model files). Nothing is written to /raid; /home only receives this folder's scripts (already in the repo).
# Requirements on the host: miniconda at /home/ubx858/miniconda3, system CUDA toolkit 12.8 (nvidia-cuda-toolkit, nvcc in
# /usr/bin, headers in /usr/include; used to build pytorch3d / the depth rasterizer / mmcv), gcc 11, internet.
# download.openmmlab.com is NOT reachable from this host (connection reset): mmcv is therefore built from the PyPI sdist
# and the RTMDet person detector is taken from Meta's HF copy (facebook/sapiens-pose-bbox-detector).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
CONDA=/home/ubx858/miniconda3/bin/conda
mkdir -p $EXA_ROOT/src $EXA_ROOT/envs $EXA_ROOT/downloads $EXA_ROOT/logs
L=$EXA_ROOT/logs
export CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST="8.9"

EXA_COMMIT=518f5647c8fa75cb648315c1677fdc105101fe11
PT3D_TAG=V0.7.8                                       # 75ebeeaea0908c5527e7b1e305fbc7681382db47
DGR_COMMIT=03f0b7d00383d6e96c22b37325ac9e5450947bf5   # leo-frank/diff-gaussian-rasterization-depth
CHUMPY_COMMIT=580566eafc9ac68b2614b64d6f7aaa84eebb70da # mattloper/chumpy master (0.71, numpy>=1.24 compatible)
DECA_COMMIT=a11554ae2a2b0f3998cf1fa94dd4db03babb34a2
DA2_COMMIT=a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
MMPOSE_TAG=v1.3.2                                     # 5408bc76f5b848cf925a0d1857899011d8c5b497

clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }

# ---------------- code ----------------
clone https://github.com/mks0601/ExAvatar_RELEASE.git $EXA_SRC $EXA_COMMIT
clone https://github.com/facebookresearch/pytorch3d.git $EXA_ROOT/src/pytorch3d $PT3D_TAG
[ -d $EXA_ROOT/src/diff-gaussian-rasterization-depth ] || git clone -q --recursive https://github.com/leo-frank/diff-gaussian-rasterization-depth.git $EXA_ROOT/src/diff-gaussian-rasterization-depth
git -C $EXA_ROOT/src/diff-gaussian-rasterization-depth checkout -q $DGR_COMMIT
clone https://github.com/mattloper/chumpy.git $EXA_ROOT/src/chumpy $CHUMPY_COMMIT
T=$EXA_SRC/fitting/tools
clone https://github.com/yfeng95/DECA.git $T/DECA $DECA_COMMIT
clone https://github.com/DepthAnything/Depth-Anything-V2.git $T/Depth-Anything-V2 $DA2_COMMIT
clone https://github.com/open-mmlab/mmpose.git $T/mmpose $MMPOSE_TAG
# Hand4Whole_RELEASE and segment-anything are NOT installed (replaced by the GT body and the given masks); COLMAP not used.
# README step "Set dataset in main/config.py" (avatar); the fitting config is already 'Custom'
sed -i "s/^    dataset = 'NeuMan' # Custom, NeuMan/    dataset = 'Custom' # Custom, NeuMan/" $EXA_SRC/avatar/main/config.py
grep -q "dataset = 'Custom'" $EXA_SRC/avatar/main/config.py
mkdir -p $EXA_SRC/fitting/data/Custom/data $EXA_SRC/avatar/data/Custom/data

# ---------------- env 1: exavatar (python 3.10, torch 2.6.0 cu124 as ExAvatar's environment.yml) ----------------
[ -x $EXA_ENV/bin/python ] || $CONDA create -y -p $EXA_ENV -c conda-forge --override-channels python=3.10 pip > $L/conda_create_main.txt 2>&1
$EXA_ENV/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 > $L/pip_torch.txt 2>&1
$EXA_ENV/bin/pip install numpy==1.26.4 opencv-python==4.10.0.84 scipy==1.15.2 smplx==0.1.28 lpips==0.1.4 \
  face-alignment==1.3.4 kornia==0.8.0 scikit-image==0.25.2 yacs==0.1.8 tqdm==4.67.1 trimesh==3.23.5 plyfile==1.1 \
  matplotlib==3.10.1 pillow==10.4.0 einops==0.8.1 gdown==5.2.0 huggingface-hub==0.25.2 imageio==2.37.0 \
  fvcore==0.1.5.post20221221 iopath==0.1.10 pyyaml==6.0.2 ninja numba==0.61.0 setuptools==69.5.1 wheel > $L/pip_main_deps.txt 2>&1
$EXA_ENV/bin/pip install --no-build-isolation $EXA_ROOT/src/chumpy > $L/pip_chumpy.txt 2>&1
FORCE_CUDA=1 MAX_JOBS=48 $EXA_ENV/bin/pip install --no-build-isolation -v $EXA_ROOT/src/pytorch3d > $L/build_pytorch3d.txt 2>&1
MAX_JOBS=16 $EXA_ENV/bin/pip install --no-build-isolation -v $EXA_ROOT/src/diff-gaussian-rasterization-depth > $L/build_dgr_depth.txt 2>&1
$EXA_ENV/bin/python -c "import torch, pytorch3d, diff_gaussian_rasterization_depth, chumpy, smplx, lpips, face_alignment; print('exavatar env ok', torch.__version__)"

# ---------------- env 2: exavatar_pose (mmpose needs mmcv 2.1 -> torch 2.1.2 cu121) ----------------
[ -x $EXA_ENV_POSE/bin/python ] || $CONDA create -y -p $EXA_ENV_POSE -c conda-forge --override-channels python=3.10 pip > $L/conda_create_pose.txt 2>&1
$EXA_ENV_POSE/bin/pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 > $L/pip_pose_torch.txt 2>&1
$EXA_ENV_POSE/bin/pip install numpy==1.26.4 opencv-python==4.10.0.84 mmengine==0.10.7 setuptools==69.5.1 wheel ninja psutil > $L/pip_pose_deps.txt 2>&1
MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=64 $EXA_ENV_POSE/bin/pip install --no-build-isolation -v mmcv==2.1.0 > $L/build_mmcv.txt 2>&1
$EXA_ENV_POSE/bin/pip install --no-build-isolation $EXA_ROOT/src/chumpy > $L/pip_pose_chumpy.txt 2>&1
$EXA_ENV_POSE/bin/pip install mmdet==3.3.0 > $L/pip_pose_mmdet.txt 2>&1
( cd $T/mmpose && $EXA_ENV_POSE/bin/pip install --no-build-isolation -e . ) > $L/pip_pose_mmpose.txt 2>&1
$EXA_ENV_POSE/bin/python -c "import mmcv, mmdet, mmpose; from mmcv.ops import nms; print('pose env ok', mmcv.__version__, mmdet.__version__, mmpose.__version__)"

# ---------------- weights (each download in its own folder) ----------------
D=$EXA_ROOT/downloads; mkdir -p $D/deca $D/dwpose $D/rtmdet $D/depth_anything
[ -f $D/deca/deca_model.tar ] || $EXA_ENV/bin/gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O $D/deca/deca_model.tar   # DECA fetch_data.sh id
[ -f $D/dwpose/dw-ll_ucoco_384.pth ] || curl -sSL -o $D/dwpose/dw-ll_ucoco_384.pth https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth
[ -f $D/rtmdet/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth ] || curl -sSL -o $D/rtmdet/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  https://huggingface.co/facebook/sapiens-pose-bbox-detector/resolve/main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
if [ ! -f $D/depth_anything/depth_anything_v2_vitl.pth ]; then
  VACE=/home/ubx858/VACE/models/VACE-Annotators/depth/depth_anything_v2_vitl.pth   # local copy, same SHA-256 as HF
  if [ -f $VACE ]; then cp $VACE $D/depth_anything/; else curl -sSL -o $D/depth_anything/depth_anything_v2_vitl.pth \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth; fi
fi
sha256sum -c - <<EOF
e714ed293054cba5eea9c96bd3b6b57880074cd84b3fd00d606cbaf0bee7c5c2  $D/deca/deca_model.tar
0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07  $D/dwpose/dw-ll_ucoco_384.pth
b66b27072c6a3cd4f093882df440921987076131fb78a7df7b1cf92d67f41509  $D/rtmdet/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345  $D/depth_anything/depth_anything_v2_vitl.pth
EOF

# ---------------- body models (copies; sources untouched) ----------------
H=$EXA_ROOT/human_model_files; mkdir -p $H/smplx/smplx_uv $H/flame/2019
V=/raid/ubx858/datasets/models/smplx_v1_1/models/smplx          # SMPL-X v1.1 (read only)
G=/home/ubx858/guava/assets
LHM=/raid/ubx858/LHM-plusplus_data/pretrained_models/huggingface/models--3DAIGC--LHMPP-Prior/snapshots/b683c8f68bede4f318b0bb539730b8e6711d30a0/human_model_files
# LHM++ ships ExAvatar's exact human_model_files layout; its SMPL-X v1.1 / FLAME 2020 / vertex-id / UV files are byte-identical
# to our official copies above, so its remaining official files (FLAME 2019, FLAME male/female 2020, FLAME embeddings,
# FLAME_texture.npz, smplx_flip_correspondences.npz) are copied from there (read only).
cp $V/SMPLX_NEUTRAL.npz $V/SMPLX_MALE.npz $V/SMPLX_FEMALE.npz $H/smplx/
cp $G/SMPLX/MANO_SMPLX_vertex_ids.pkl $G/SMPLX/SMPL-X__FLAME_vertex_ids.npy $H/smplx/
cp $G/SMPLX/smplx_uv.obj $H/smplx/smplx_uv/
cp -L $G/FLAME/FLAME2020/generic_model.pkl $H/flame/FLAME_NEUTRAL.pkl
cp -L $LHM/smplx/smplx_flip_correspondences.npz $H/smplx/
cp -L $LHM/flame/FLAME_MALE.pkl $H/flame/FLAME_MALE.pkl
cp -L $LHM/flame/FLAME_FEMALEl.pkl $H/flame/FLAME_FEMALE.pkl
cp -L $LHM/flame/flame_dynamic_embedding.npy $LHM/flame/flame_static_embedding.pkl $LHM/flame/FLAME_texture.npz $H/flame/
cp -L $LHM/flame/2019/generic_model.pkl $H/flame/2019/
for d in fitting avatar; do ln -sfn $H $EXA_SRC/$d/common/utils/human_model_files; done

# ---------------- place weights where ExAvatar's tools expect them; ExAvatar's copy_code.py ----------------
cp $D/deca/deca_model.tar $T/DECA/data/
cp $H/flame/FLAME_NEUTRAL.pkl $T/DECA/data/generic_model.pkl
mkdir -p $T/DECA/data/FLAME2020
cp $H/flame/FLAME_MALE.pkl $T/DECA/data/FLAME2020/male_model.pkl
cp $H/flame/FLAME_FEMALE.pkl $T/DECA/data/FLAME2020/female_model.pkl
cp $D/dwpose/dw-ll_ucoco_384.pth $D/rtmdet/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth $T/mmpose/
mkdir -p $T/Depth-Anything-V2/checkpoints && cp $D/depth_anything/depth_anything_v2_vitl.pth $T/Depth-Anything-V2/checkpoints/
( cd $T && $EXA_ENV/bin/python copy_code.py ) || true   # Hand4Whole / segment-anything copies fail harmlessly (not installed)
# face_alignment (DECA's FAN detector) and lpips' VGG16 download their weights into TORCH_HOME on first use.
echo "setup done"
