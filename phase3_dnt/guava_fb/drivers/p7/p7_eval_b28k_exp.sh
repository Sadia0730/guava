#!/bin/bash
# Phase 7 amendment 1: exposure-shift probe reference with B28k (naive fusion, support frames x1.2 brightness / x0.75 saturation). usage: p7_eval_b28k_exp.sh GPU
O=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims:/dev/shm/ubx858_eval/pylib_o2t PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/ubx858/miniconda3/envs/guava/bin/python; S=/dev/shm/ubx858_eval/valbench_v1; cd /home/ubx858/guava/phase3_dnt/eval_split
$P fusion_gt_row.py --E $S --split valbench_v1.json --name p7_b28k_fused_exp --mode mean --prior 2 --thr 0 --cap 0 --support-gain 1.2 --support-sat 0.75 && echo EXP_DONE
$P score.py --methods p7_b28k_fused_exp --split valbench_v1.json --E $S --tag vb && echo P7_EVAL_B28K_EXP_DONE
