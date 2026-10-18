#!/usr/bin/env bash
# PEAR-free programme evaluation (PREREG "RUN 1 / RUN 2 / RUN 3 / FINAL"): the selected weights of a run's best.pt
# (WHICH = ema | live) under the author's protocol (COCO, LSP-Ext, 3DPW test), EHF, COCO-WholeBody val face / hand,
# the 24-frame face check (next to PEAR / L-real / A2-ctrl / Track A) and face / hand overlays (vs A2-ctrl).
# Args: NAME RUN_DIR WHICH GPU. Output /raid/ubx858/outputs/dnt/eval_programme/<NAME>/
set -uo pipefail
NAME=$1; RUN=$2; WHICH=$3; GPU=$4; O=/raid/ubx858/outputs/dnt/eval_programme/$NAME; P=/home/ubx858/miniconda3/envs/guava/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $O; cd /home/ubx858/guava
step() { echo "$(date '+%F %T') $*"; }; fail() { step "FAILED: $*"; exit 1; }
CK=$RUN/checkpoints/best.pt
if [ "$WHICH" = ema ]; then
  CK=$O/best_ema.pt
  $P - "$RUN/checkpoints/best.pt" "$CK" <<'PY' || fail "ema extraction"
import sys, torch
s = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
torch.save({'student': s['student_ema'], 'step': s['step'], 'source': sys.argv[1] + ':student_ema'}, sys.argv[2]); print('step', s['step'])
PY
fi
for DS in COCO-VAL LSP-EXTENDED 3DPW-TEST; do
  $P tools/eval_author_protocol.py --dataset $DS --model student_vitpose --checkpoint $CK --output-dir $O/author/$DS --device cuda:0 > $O/author.$DS.log 2>&1 || fail "author $DS"
done
$P tools/eval_ehf_standard.py --model student_vitpose --checkpoint $CK --output-dir $O/ehf > $O/ehf.log 2>&1 || fail ehf
$P tools/eval_wholebody_val.py --model student_vitpose --checkpoint $CK --output-dir $O/wholebody > $O/wholebody.log 2>&1 || fail wholebody
$P phase3_dnt/qual_face.py --extra "$NAME=$CK" --out $O/qual_face > $O/qual_face.log 2>&1 || fail qual_face
(cd third_party/PEAR && $P tools/fa_overlays.py --ckpt /raid/ubx858/outputs/dnt/trackA2/eval/ckpt/A2ctrl_best_ema.pt --ckpt $CK --names "A2-ctrl" "$NAME" --out $O/overlays > $O/overlays.log 2>&1) || fail overlays
step "FULL_EVAL_DONE $NAME"
