#!/bin/bash
# Phase 7 evaluation of one arm (PREREG Phase 7 + amendment 1) on ValBench-v1: single-image row, fused row (mean, prior 2, thr 0, cap 0),
# exposure-shift probe (support frames x1.2 brightness, x0.75 saturation), then scores (--tag vb).  usage: p7_eval.sh ARM GPU [CKPT]
ARM=$1; GPU=$2; O=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb
CK=${3:-$O/runs/guava_fb_p7_$ARM/checkpoints/latest.pt}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims:/dev/shm/ubx858_eval/pylib_o2t PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
try3() { for t in 1 2 3; do "$@" && return 0; echo "RETRY $t"; sleep 60; done; return 1; }
P=/home/ubx858/miniconda3/envs/guava/bin/python; S=/dev/shm/ubx858_eval/valbench_v1; cd /home/ubx858/guava/phase3_dnt/eval_split
$P eval_guava_rows.py --row fb_gt --fb-ckpt $CK --name p7_${ARM}_single --split valbench_v1.json --E $S && echo SINGLE_DONE
try3 $P fusion_gt_row.py --E $S --split valbench_v1.json --name p7_${ARM}_fused --mode mean --prior 2 --thr 0 --cap 0 --fb-ckpt $CK && echo FUSED_DONE
try3 $P fusion_gt_row.py --E $S --split valbench_v1.json --name p7_${ARM}_fused_exp --mode mean --prior 2 --thr 0 --cap 0 --fb-ckpt $CK --support-gain 1.2 --support-sat 0.75 && echo EXP_DONE
$P score.py --methods p7_${ARM}_single,p7_${ARM}_fused,p7_${ARM}_fused_exp --split valbench_v1.json --E $S --tag vb && echo P7_EVAL_${ARM}_DONE
