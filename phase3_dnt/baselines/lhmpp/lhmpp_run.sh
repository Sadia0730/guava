#!/bin/bash
# LHM++ baseline (PREREG "FinalBench addendum: LHM++"): render one row on all FinalBench splits + BEDLAM2, then score.
# usage: lhmpp_run.sh NAME VIEWS GPU
NAME=$1; VIEWS=$2; GPU=$3; O=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval; ES=/home/ubx858/guava/phase3_dnt/eval_split
C=/dev/shm/ubx858_baselines/lhmpp/cache; export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 HF_HOME=$C TORCH_HOME=$C XDG_CACHE_HOME=$C OMP_NUM_THREADS=8
PL=/home/ubx858/miniconda3/envs/lhmpp/bin/python; PG=/home/ubx858/miniconda3/envs/guava/bin/python; R=/home/ubx858/guava/phase3_dnt/baselines/lhmpp/run_lhmpp.py
cd /dev/shm/ubx858_baselines/lhmpp
for S in "avatar_test_v1 avatar_test_v1.json" "publicbench_v1 publicbench_v1.json" "publicbench_v2 publicbench_v2_xhumans.json"; do
  set -- $S
  $PL $R --E $O/$1 --split $ES/$2 --name $NAME --views $VIEWS 2>&1 | grep -v -i -E "flash attention|warn" && echo "RENDER_DONE $1"
  (cd $ES && PYTHONNOUSERSITE=1 $PG score.py --methods $NAME --split $ES/$2 --E $O/$1 --tag full) && echo "SCORE_DONE $1"
done
echo LHMPP_RUN_${NAME}_DONE
