#!/usr/bin/env bash
# DNT C1 B3-B4 (approved 2026-10-16): wait for the B2 oracle tracking to finish, then render and score every tracked clip
# of the split on GPU $1 (phase3_dnt/c1_live_gap.py, future frames >= 60). Prints C1_DONE or FAILED.
set -uo pipefail
GPU=$1; LOG=/raid/ubx858/outputs/dnt/c1/tracking.log
until grep -q "EHM_TRACKING_DONE" $LOG; do sleep 120; done
echo "$(date '+%F %T') tracking done: $(grep -c 'exit 0' $LOG) clips exit 0, $(grep -c 'exit [1-9]' $LOG) failed"
free=$(df --output=avail -B1G /raid | tail -1); [ "$free" -lt 15 ] && { echo "FAILED: /raid below 15 GB ($free GB)"; exit 1; }
cd /home/ubx858/guava
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU \
  /home/ubx858/miniconda3/envs/guava/bin/python phase3_dnt/c1_live_gap.py --split /raid/ubx858/outputs/dnt/c1/split.json \
  --tracked /raid/ubx858/outputs/dnt/c1/tracked --out /raid/ubx858/outputs/dnt/c1/results --future-from 60 --device cuda:0 \
  > /raid/ubx858/outputs/dnt/c1/results.log 2>&1 || { echo "FAILED: c1_live_gap"; exit 1; }
echo "$(date '+%F %T') C1_DONE"
