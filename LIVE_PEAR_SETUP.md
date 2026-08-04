# Live PEAR to GUAVA Setup

This guide reproduces the live webcam pipeline on another Ubuntu PC. The
pipeline uses one `guava` Conda environment: PEAR estimates motion parameters,
and GUAVA deforms and renders a blue-shirt avatar that is created once before
the stream starts.

## 1. System requirements

- NVIDIA GPU with a recent driver and at least 16 GB VRAM recommended for
  loading PEAR and GUAVA together.
- Ubuntu Linux, Conda, Git, a C++ compiler, and the CUDA toolkit with `nvcc`.
- The tested local stack is Python 3.10, PyTorch 2.2.0+cu121, and CUDA toolkit
  12.2. The driver must support CUDA 12.x.

Check the GPU and compiler before installing:

```bash
nvidia-smi
nvcc --version
gcc --version
```

## 2. Clone the live branch

```bash
git clone --branch feature/live-pear-guava --recursive \
  https://github.com/Sadia0730/guava.git GUAVA
cd GUAVA

mkdir -p third_party
git clone https://github.com/Pixel-Talk/PEAR.git third_party/PEAR
```

## 3. Create the environment

```bash
conda create -n guava python=3.10 -y
conda activate guava

python -m pip install --upgrade pip
python -m pip install ninja wheel setuptools
python -m pip install -r requirements.txt
```

Verify that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## 4. Build PyTorch3D and CUDA extensions

`CUDA_HOME` must point to the CUDA toolkit containing `nvcc`. Change the path
if CUDA is installed elsewhere.

```bash
export CUDA_HOME=/usr/local/cuda

python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-32
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/fused-ssim
```

Use the paths beginning with `./submodules/`. A command such as
`pip install diff-gaussian-rasterization-32` searches PyPI and will fail.

Verify the compiled packages:

```bash
python -c "import pytorch3d, diff_gaussian_rasterization_32, simple_knn, fused_ssim; print('CUDA extensions OK')"
```

## 5. Restore model assets

These files are licensed, large, or generated, so Git does not contain them.
Download them on the new PC or transfer them from the working PC.

GUAVA requires:

```text
assets/GUAVA/config.yaml
assets/GUAVA/checkpoints/best_160000.pt
assets/SMPLX/SMPLX_NEUTRAL_2020.npz
assets/SMPLX/flame_generic_model.pkl
assets/FLAME/FLAME2020/generic_model.pkl
```

- Download the GUAVA checkpoint from the link in `README.md`.
- Download SMPL-X from the official SMPL-X site.
- Download FLAME 2020 from the official FLAME site. Copy its
  `generic_model.pkl` to both FLAME paths shown above.

PEAR requires its asset bundle under `third_party/PEAR/assets`. Follow
`third_party/PEAR/README.md` and confirm at least these files exist:

```text
third_party/PEAR/assets/SMPLX/SMPLX_NEUTRAL_2020.npz
third_party/PEAR/assets/SMPLX/flame_generic_model.pkl
third_party/PEAR/assets/SMPLX/smpl_mean_params.npz
third_party/PEAR/assets/FLAME/FLAME2020/generic_model.pkl
```

The PEAR neural checkpoint is downloaded automatically from Hugging Face on
the first run and then read from the local cache.

## 6. Restore the source avatar tracking

The blue-shirt avatar is generated from this EHM-tracked directory:

```text
assets/example/tracked_image/random_google_pic/blue_shirt/
```

Transfer the complete directory from the working PC. It must include:

```text
optim_tracking_ehm.pkl
id_share_params.pkl
videos_info.json
img_lmdb/data.mdb
img_lmdb/lock.mdb
```

You can instead use any source image already processed by EHM-Tracker and pass
its tracked directory with `--source_data_path`.

## 7. Run the webcam pipeline

```bash
conda activate guava
cd GUAVA

python main/live_pear_guava.py \
  --input 0 \
  --render_size 256
```

Press `q` in the preview window to stop. The script loads and warms PEAR and
GUAVA, creates the blue-shirt Gaussian avatar, and only then opens the webcam.
No per-frame files are saved.

For an RTSP or HTTP stream:

```bash
python main/live_pear_guava.py \
  --input "rtsp://host/path" \
  --render_size 256
```

For a headless timing run:

```bash
python main/live_pear_guava.py \
  --input 0 \
  --render_size 256 \
  --no_display
```

The printed metrics have different scopes:

- `PEAR`: PEAR neural parameter inference only.
- `GUAVA`: avatar deformation and rendering only.
- `pipeline`: PEAR, conversion, GUAVA, and GPU readback.
- `observed_live`: capture, processing, and display loop throughput.
