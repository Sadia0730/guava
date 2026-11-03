#!/usr/bin/env bash
# Host entry point for the validation-only, fixed-inference-cost experiments.
set -euo pipefail
cd /home/ubx858/guava
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -z "${EXPERIMENT_GPU:-}" ]; then
  EXPERIMENT_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, 'BEGIN {best=2000} $2 < best && $3 < 5 {best=$2; gpu=$1} END {gsub(/ /,"",gpu); if (gpu!="") print gpu}')
fi
if [ -z "${EXPERIMENT_GPU:-}" ]; then
  echo 'No idle GPU with less than 2 GB resident memory; retry later.' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=$EXPERIMENT_GPU
echo "Experiment GPU: $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
case "${1:-}" in
  inventory|probe|benchmark|verify|pilot|report|matte_benchmark)
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/no_latency_experiments.py "$@" ;;
  eval)
    shift
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/eval_guava_rows.py "$@" ;;
  score)
    shift
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/score.py "$@" ;;
  *) echo 'Expected inventory, probe, benchmark, verify, pilot, report, eval, or score' >&2; exit 2 ;;
esac
