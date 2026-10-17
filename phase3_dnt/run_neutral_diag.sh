#!/usr/bin/env bash
set -uo pipefail
GPU=$1; O=/raid/ubx858/outputs/dnt/trackA/eval; P=/home/ubx858/miniconda3/envs/guava/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU
cd /home/ubx858/guava
step() { echo "$(date '+%F %T') $*"; }; fail() { step "FAILED: $*"; exit 1; }
$P $O/make_neutral.py /raid/ubx858/outputs/phase1_task1/screen/L_real/checkpoints/best.pt $O/ckpt/Lreal_neutral.pt || fail neutral1
$P $O/make_neutral.py /raid/ubx858/outputs/dnt/trackA/run/checkpoints/best.pt $O/ckpt/A_best_live_neutral.pt || fail neutral2
for N in Lreal_neutral A_best_live_neutral; do CK=$O/ckpt/$N.pt
  for DS in COCO-VAL LSP-EXTENDED 3DPW-TEST; do
    $P tools/eval_author_protocol.py --dataset $DS --model student_vitpose --checkpoint $CK --output-dir $O/diag_$N/author/$DS --device cuda:0 > $O/diag_$N.author.$DS.log 2>&1 || fail "$N $DS"
  done
  $P tools/eval_ehf_standard.py --model student_vitpose --checkpoint $CK --output-dir $O/diag_$N/ehf > $O/diag_$N.ehf.log 2>&1 || fail "$N ehf"
  step "$N done"
done
step "NEUTRAL_DIAG_DONE"
