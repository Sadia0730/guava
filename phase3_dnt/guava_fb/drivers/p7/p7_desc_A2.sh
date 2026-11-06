#!/bin/bash
# Phase 7 round 2 descriptive rows (PREREG amendment 2: PS / NeuMan fused rows reported descriptively after the decision). usage: p7_desc_A2.sh GPU
O=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb; PB=$O/eval/publicbench_v1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims:/dev/shm/ubx858_eval/pylib_o2t PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
try3() { for t in 1 2 3; do "$@" && return 0; echo "RETRY $t"; sleep 60; done; return 1; }
P=/home/ubx858/miniconda3/envs/guava/bin/python; cd /home/ubx858/guava/phase3_dnt/eval_split
for ARM in A2 B2; do
  try3 $P fusion_gt_row.py --E $PB --split publicbench_v1.json --name p7_${ARM}_fused --mode mean --prior 2 --thr 0 --cap 0 --fb-ckpt $O/runs/guava_fb_p7_$ARM/checkpoints/latest.pt && echo PB_${ARM}_DONE
done
$P score.py --methods p7_A2_fused,p7_B2_fused --split publicbench_v1.json --E $PB --tag full && echo P7_DESC2_DONE
