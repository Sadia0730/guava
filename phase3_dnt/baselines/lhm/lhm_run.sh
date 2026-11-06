#!/bin/bash
# LHM baseline (PREREG "FinalBench addendum 2: LHM"): render one shard of a split, or score all three splits.
# usage: lhm_run.sh render GPU SPLITNAME SPLITJSON [CLIPS_COMMA_LIST]   |   lhm_run.sh score
O=/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval; ES=/home/ubx858/guava/phase3_dnt/eval_split; NAME=${LHM_NAME:-lhm}; FLAGS=${LHM_FLAGS:-}
C=/dev/shm/ubx858_baselines/lhm/cache; export PYTHONDONTWRITEBYTECODE=1 HF_HOME=$C TORCH_HOME=$C XDG_CACHE_HOME=$C OMP_NUM_THREADS=8
PL=/home/ubx858/miniconda3/envs/lhmpp/bin/python; PG=/home/ubx858/miniconda3/envs/guava/bin/python; R=/home/ubx858/guava/phase3_dnt/baselines/lhm/run_lhm.py
if [ "$1" = render ]; then
  export CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/dev/shm/ubx858_baselines/lhm/pydeps
  cd /dev/shm/ubx858_baselines/lhm
  $PL $R --E $O/$3 --split $ES/$4 --name $NAME $FLAGS --clips "${5:-}" 2>&1 | grep -v -i -E "flash attention|warn|return forward_call|F.conv2d" && echo "RENDER_DONE $3 $2"
else
  for S in "avatar_test_v1 avatar_test_v1.json" "publicbench_v1 publicbench_v1.json" "publicbench_v2 publicbench_v2_xhumans.json"; do
    set -- $S
    (cd $ES && PYTHONNOUSERSITE=1 $PG score.py --methods $NAME --split $ES/$2 --E $O/$1 --tag full) && echo "SCORE_DONE $1"
  done
fi
