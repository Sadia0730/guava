#!/usr/bin/env bash
# DNT A2 / A2-ctrl evaluation: best.pt EMA (+ live), author's protocol (COCO, LSP-Ext, 3DPW test) and EHF, as Track A.
# Args: NAME RUN_DIR GPU. Output /raid/ubx858/outputs/dnt/trackA2/eval/<NAME>_{ema,live}/
set -uo pipefail
NAME=$1; RUN=$2; GPU=$3; O=/raid/ubx858/outputs/dnt/trackA2/eval; P=/home/ubx858/miniconda3/envs/guava/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU
cd /home/ubx858/guava; mkdir -p $O/ckpt
step() { echo "$(date '+%F %T') $*"; }; fail() { step "FAILED: $*"; exit 1; }
$P - "$RUN/checkpoints/best.pt" "$O/ckpt/${NAME}_best_ema.pt" <<'PY' || fail "ema extraction"
import sys, torch
s = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
torch.save({'student': s['student_ema'], 'step': s['step'], 'source': sys.argv[1] + ':student_ema'}, sys.argv[2]); print('step', s['step'])
PY
for spec in "${NAME}_ema|$O/ckpt/${NAME}_best_ema.pt" "${NAME}_live|$RUN/checkpoints/best.pt"; do
  IFS='|' read -r N CK <<< "$spec"
  for DS in COCO-VAL LSP-EXTENDED 3DPW-TEST; do
    $P tools/eval_author_protocol.py --dataset $DS --model student_vitpose --checkpoint $CK --output-dir $O/$N/author/$DS --device cuda:0 > $O/$N.author.$DS.log 2>&1 || fail "$N $DS"
  done
  $P tools/eval_ehf_standard.py --model student_vitpose --checkpoint $CK --output-dir $O/$N/ehf > $O/$N.ehf.log 2>&1 || fail "$N ehf"
  step "$N done"
done
step "A2_EVAL_DONE $NAME"
