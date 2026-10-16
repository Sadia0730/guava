#!/usr/bin/env bash
# DNT C1 B2 (driver O): EHM-Tracker on every clip of the C1 split, N clips in parallel (CPU-bound ONNX + one GPU for
# the optimisation), from the A0 shadow directory (fixed assets / pretrained links). Args: GPU, PARALLEL.
# Output: /raid/ubx858/outputs/dnt/c1/tracked/<clip>/ (EHM format consumed by phase3_dnt/c1_live_gap.py).
set -uo pipefail
GPU=$1; PAR=${2:-4}
OUT=/raid/ubx858/outputs/dnt/c1/tracked; mkdir -p $OUT /raid/ubx858/outputs/dnt/c1/videos /raid/ubx858/outputs/dnt/c1/logs
cd /raid/ubx858/outputs/phase1_task3/a0/ehm_run
python3 -c "
import json, shutil, os
s = json.load(open('/raid/ubx858/outputs/dnt/c1/split.json'))
for c in s['clips']:
    dst = f\"/raid/ubx858/outputs/dnt/c1/videos/{c['clip']}.mp4\"
    if not os.path.exists(dst): os.symlink(c['path'], dst)
print(len(s['clips']), 'clips linked')"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/datasets/eval_assets/pylib_task3 CUDA_VISIBLE_DEVICES=$GPU
# CPU: each worker is pinned to its own block of cores (ONNX / torch thread pools otherwise size themselves to the whole
# machine and six workers thrash: measured 15-17 s/frame vs 1.9 s/frame for one worker).
NC=$(( $(nproc) / PAR )); export NC OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC
ls /raid/ubx858/outputs/dnt/c1/videos/*.mp4 | xargs -P $PAR --process-slot-var=SLOT -I{} bash -c '
  n=$(basename {} .mp4); test -f '$OUT'/$n/optim_tracking_ehm.pkl && { echo "skip $n"; exit 0; }
  lo=$(( SLOT * NC )); hi=$(( lo + NC - 1 ))
  s=$(date +%s); taskset -c $lo-$hi /home/ubx858/miniconda3/envs/guava/bin/python tracking_video.py --in_root {} --output_dir '$OUT' --not_check_hand -n 1 -v 0 -p 0,1 > /raid/ubx858/outputs/dnt/c1/logs/$n.log 2>&1
  echo "$n exit $? $(( $(date +%s) - s )) s"'
echo "EHM_TRACKING_DONE $(ls $OUT | wc -l) clips"
