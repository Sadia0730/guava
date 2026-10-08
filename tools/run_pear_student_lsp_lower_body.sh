#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubx858/miniconda3/envs/guava/bin/python}"
GPU="${GPU:-0}"
OUTPUT="${OUTPUT:-/raid/ubx858/outputs/pear_student_v2_lsp_lower_body/finetune_from_0235000}"

cd "$ROOT"
exec "$PYTHON" tools/train_pear_student_lsp.py \
  --lsp_train_manifest /raid/ubx858/datasets/processed/pear_student_lsp/train.jsonl \
  --lsp_val_manifest /raid/ubx858/datasets/processed/pear_student_lsp/val.jsonl \
  --original_manifest /raid/ubx858/datasets/processed/pear_student/train.jsonl \
  --resume /raid/ubx858/outputs/pear_student_v2_lsp_lower_body/baseline/step_0235000.pt \
  --output_dir "$OUTPUT" \
  --teacher_ckpt /home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt \
  --device "cuda:$GPU" \
  --lsp_batch_size 4 \
  --original_batch_size 4 \
  --num_workers 8 \
  --steps 5000 \
  --lr 1e-5 \
  --precision bf16 \
  --log_every 20 \
  --save_every 500 \
  --val_every 500 \
  --val_batches 50
