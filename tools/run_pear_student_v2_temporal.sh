#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubx858/guava
PEAR="$ROOT/third_party/PEAR"
PY=/home/ubx858/miniconda3/envs/guava/bin/python
DATA=/raid/ubx858/datasets/processed/pear_student
OUT=/raid/ubx858/outputs/pear_student_v2_temporal_refine
TEACHER=/home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt
RESUME=/raid/ubx858/outputs/pear_student_v2_phase2/checkpoints/step_0235000.pt

cd "$PEAR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$PY" train_pear_student_distill.py \
  --student_config configs/student_l70_v2.yaml \
  --teacher_ckpt "$TEACHER" \
  --train_manifest "$DATA/train.jsonl" \
  --val_manifest "$DATA/val.jsonl" \
  --output_dir "$OUT" \
  --resume "$RESUME" \
  --batch_size 24 \
  --clip_length 2 \
  --temporal_stride 1 \
  --clip_step 4 \
  --motion_pool_factor 3 \
  --motion_random_fraction 0.25 \
  --steps 285000 \
  --lr 5e-6 \
  --lr_schedule constant \
  --feature_weight 1.0 \
  --velocity_weight 0.0 \
  --flame_weight 2.0 \
  --body_delta_magnitude_weight 2.0 \
  --body_delta_direction_weight 0.15 \
  --hand_delta_magnitude_weight 2.0 \
  --hand_delta_direction_weight 0.25 \
  --face_delta_magnitude_weight 0.5 \
  --face_delta_direction_weight 0.10 \
  --save_every 2500 \
  --val_every 2500 \
  --health_every 2500 \
  --skeleton_every 2500 \
  --num_workers 8 \
  --precision bf16 \
  --device cuda:0
