#!/usr/bin/env bash
# Phase 1 Task 1 screening: evaluate each variant's best checkpoint (by 3DPW validation PA-MPJPE)
# on 3DPW test (GT-keypoint and detector crops) and EHF, with the same evaluator commands as the
# PEAR and student-235000 baseline rows. Variant (a) on GPU 0, (b) on 1, (c) on 2, in parallel.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/raid/ubx858/datasets/eval_assets/pylib
PY=/home/ubx858/miniconda3/envs/guava/bin/python
OUT=/raid/ubx858/outputs/phase1_task1
DET=$OUT/eval_3dpw/detections/detections_test.json

run_variant() {
  local v=$1 gpu=$2 ckpt=$OUT/screen/$1/checkpoints/best.pt
  export CUDA_VISIBLE_DEVICES=$gpu
  $PY tools/eval_3dpw_standard.py --model student --checkpoint "$ckpt" --split test --crop gt_keypoints \
    --batch-size 64 --num-workers 12 --save-overlays 24 --output-dir "$OUT/eval_3dpw/screen_${v}_best/test_gt_keypoints"
  $PY tools/eval_3dpw_standard.py --model student --checkpoint "$ckpt" --split test --crop detector --detections "$DET" \
    --batch-size 64 --num-workers 12 --save-overlays 24 --output-dir "$OUT/eval_3dpw/screen_${v}_best/test_detector"
  $PY tools/eval_ehf_standard.py --model student --checkpoint "$ckpt" --output-dir "$OUT/eval_ehf/screen_${v}_best"
}

pids=()
for pair in a:0 b:1 c:2; do
  run_variant "${pair%:*}" "${pair#*:}" > "$OUT/screen/eval_${pair%:*}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
