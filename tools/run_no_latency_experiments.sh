#!/usr/bin/env bash
# Host entry point for the validation-only, fixed-inference-cost experiments.
set -euo pipefail
cd /home/ubx858/guava
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${EXPERIMENT_GPU:-2}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
case "${1:-}" in
  inventory|probe|benchmark|teacher|train)
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/no_latency_experiments.py "$@" ;;
  eval)
    shift
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/eval_guava_rows.py "$@" ;;
  score)
    shift
    exec /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/eval_split/score.py "$@" ;;
  *) echo 'Expected inventory, benchmark, teacher, train, eval, or score' >&2; exit 2 ;;
esac
