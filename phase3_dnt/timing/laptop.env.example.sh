# Copy to your own file outside the repository; adjust to existing laptop installs.
# Activate the matching env first. These variables change paths, not model settings.
export GUAVA_PY="${GUAVA_PY:-python}"
export RAGA_ASSETS="${RAGA_ASSETS:-$PWD/live_assets}"
export RAGA_FB_CKPT="$RAGA_ASSETS/checkpoints/guava_fb_B28k.pt"
export POSE_CKPT="$RAGA_ASSETS/checkpoints/sprint_scratch_ema128k.pt"
export ARCFACE_ONNX="$RAGA_ASSETS/arcface/glintr100.onnx"
export TORCH_HOME="$RAGA_ASSETS/torch"
export CUDA_VISIBLE_DEVICES=0
export TIMING_POWER_NOTE="AC power; performance mode; RTX 3080 Laptop 16 GB"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Existing baseline installs; do not re-download or extract all 120 GB for FPS.
export BASELINE_ROOT="${BASELINE_ROOT:-/dev/shm/ubx858_baselines}"
export GART_ROOT="$BASELINE_ROOT/gart"
export GA_BASE="$BASELINE_ROOT/gaussianavatar"
export EXA_ROOT_ORIG="$BASELINE_ROOT/exavatar"
export IDOL_CODE="$BASELINE_ROOT/idol/src"
export LHM_CODE="$BASELINE_ROOT/lhm/src"
export LHM_MODEL="$BASELINE_ROOT/weights/lhm_1b_hf"
export LHMPP_CODE="${LHMPP_CODE:-$HOME/LHM-plusplus}"
export LHMPP_PRETRAINED="${LHMPP_PRETRAINED:-$LHMPP_CODE/pretrained_models}"
# Set these to the existing matching environments. Do not substitute the guava env.
export LHM_PY="${LHM_PY:-$HOME/miniconda3/envs/lhmpp/bin/python}"
export LHMPP_PYTHON="${LHMPP_PYTHON:-$LHM_PY}"
export IDOL_PY="${IDOL_PY:-$LHM_PY}"
