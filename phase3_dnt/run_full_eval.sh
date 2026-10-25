#!/usr/bin/env bash
# PEAR-free programme evaluation (PREREG "RUN 1 / RUN 2 / RUN 3 / FINAL"): the selected weights of a run's best.pt
# (WHICH = ema | live) under the author's protocol (COCO, LSP-Ext, 3DPW test), EHF, COCO-WholeBody val face / hand,
# the 24-frame face check (next to PEAR / L-real / A2-ctrl / Track A) and face / hand overlays (vs A2-ctrl).
# Args: NAME RUN_DIR WHICH GPU [HEAD] [ORIENT]. Output /raid/ubx858/outputs/dnt/eval_programme/<NAME>/
#   WHICH = ema | live | ckpt (ckpt: RUN_DIR is the checkpoint file itself, used as is)
#   HEAD  = ehm (default: SMPL-X body + FLAME head, unchanged commands) | smplx (plain SMPL-X 2020, no FLAME, head / hand
#           scale 1; tools/eval_3dpw_standard.py HEADS). The FA overlays (third_party/PEAR/tools/fa_overlays.py) are EHM-only
#           and are skipped for HEAD=smplx.
#   ORIENT = 0 (default) | 1: also tools/eval_orientation_hard.py (LSP hard subset / flips / torso direction) -> <NAME>/orientation
#   CAMERA = 0 (default, unchanged commands) | 2: --camera --camera-width-only (focal = width for every dataset) | 1: pass --camera (SPRINT camera input cond={"cam": ...}; ignored by non-SPRINT models)
#   MODEL  = student_vitpose (default) | sprint | sprint_raw: the evaluators' --model (qual_face: --extra-model)
set -uo pipefail
NAME=$1; RUN=$2; WHICH=$3; GPU=$4; HEAD=${5:-ehm}; ORIENT=${6:-0}; CAMERA=${7:-0}; MODEL=${8:-student_vitpose}; O=/raid/ubx858/outputs/dnt/eval_programme/$NAME; P=/home/ubx858/miniconda3/envs/guava/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $O; cd /home/ubx858/guava
step() { echo "$(date '+%F %T') $*"; }; fail() { step "FAILED: $*"; exit 1; }
CK=$RUN/checkpoints/best.pt
HA=(); [ "$HEAD" = ehm ] || HA=(--head "$HEAD")
[ "$CAMERA" = 1 ] && HA+=(--camera)
[ "$CAMERA" = 2 ] && HA+=(--camera --camera-width-only)   # no extra information: focal = image width everywhere
QA=(); [ "$MODEL" = student_vitpose ] || QA=(--extra-model "$MODEL")
[ "$WHICH" = ckpt ] && CK=$RUN
if [ "$WHICH" = ema ]; then
  CK=$O/best_ema.pt
  $P - "$RUN/checkpoints/best.pt" "$CK" <<'PY' || fail "ema extraction"
import sys, torch
s = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
d = {'student': s['student_ema'], 'step': s['step'], 'source': sys.argv[1] + ':student_ema'}
if s.get('config') is not None: d['config'] = s['config']      # SPRINT loader reads config.sprint (n_dense)
torch.save(d, sys.argv[2]); print('step', s['step'])
PY
fi
for DS in COCO-VAL LSP-EXTENDED 3DPW-TEST; do
  $P tools/eval_author_protocol.py --dataset $DS --model $MODEL --checkpoint $CK --output-dir $O/author/$DS --device cuda:0 "${HA[@]}" > $O/author.$DS.log 2>&1 || fail "author $DS"
done
$P tools/eval_ehf_standard.py --model $MODEL --checkpoint $CK --output-dir $O/ehf "${HA[@]}" > $O/ehf.log 2>&1 || fail ehf
$P tools/eval_wholebody_val.py --model $MODEL --checkpoint $CK --output-dir $O/wholebody "${HA[@]}" > $O/wholebody.log 2>&1 || fail wholebody
$P phase3_dnt/qual_face.py --extra "$NAME=$CK" --out $O/qual_face "${HA[@]}" "${QA[@]}" > $O/qual_face.log 2>&1 || fail qual_face
if [ "$HEAD" != ehm ]; then step "overlays skipped (EHM head only) for HEAD=$HEAD"; else
(cd third_party/PEAR && $P tools/fa_overlays.py --ckpt /raid/ubx858/outputs/dnt/trackA2/eval/ckpt/A2ctrl_best_ema.pt --ckpt $CK --names "A2-ctrl" "$NAME" --out $O/overlays > $O/overlays.log 2>&1) || fail overlays
fi
if [ "$ORIENT" = 1 ]; then
  $P tools/eval_orientation_hard.py --model $MODEL --checkpoint $CK --output-dir $O/orientation "${HA[@]}" > $O/orientation.log 2>&1 || fail orientation
fi
step "FULL_EVAL_DONE $NAME"
