#!/usr/bin/env bash
# DNT Track A final evaluation (requested 2026-10-16): PEAR-free student best.pt (live = PREREG-selected; EMA at the same
# step) + latest.pt EMA (descriptive), author's protocol (COCO, LSP-Extended, 3DPW test), standard 3DPW test, EHF.
# L-real re-run under the current evaluator SHA for the gated comparison. Args: GPU.
set -uo pipefail
GPU=$1; O=/raid/ubx858/outputs/dnt/trackA/eval; P=/home/ubx858/miniconda3/envs/guava/bin/python
RUN=/raid/ubx858/outputs/dnt/trackA/run/checkpoints; LREAL=/raid/ubx858/outputs/phase1_task1/screen/L_real/checkpoints/best.pt
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=$GPU
cd /home/ubx858/guava
step() { echo "$(date '+%F %T') $*"; }
fail() { step "FAILED: $*"; exit 1; }
mkdir -p $O/ckpt
# EMA weights as {'student': ema} so the evaluators' loader is unchanged (read-only on the run directory)
$P - <<'PY' || fail "ema extraction"
import torch, hashlib
R = '/raid/ubx858/outputs/dnt/trackA/run/checkpoints'; O = '/raid/ubx858/outputs/dnt/trackA/eval/ckpt'
for src, dst in (('best.pt', 'best_ema.pt'), ('latest.pt', 'latest_ema.pt')):
    s = torch.load(f'{R}/{src}', map_location='cpu', weights_only=False)
    assert set(s['student_ema']) == set(s['student'])
    torch.save({'student': s['student_ema'], 'step': s['step'], 'source': f'{R}/{src}:student_ema'}, f'{O}/{dst}')
    print(src, 'step', s['step'], 'val', s.get('metrics', {}).get('pa_mpjpe_mm'), '->', dst)
PY
step "ema extracted"
for spec in "Lreal_current_eval|$LREAL|COCO-VAL 3DPW-TEST|0" "A_best_live|$RUN/best.pt|COCO-VAL LSP-EXTENDED 3DPW-TEST|1" \
            "A_best_ema|$O/ckpt/best_ema.pt|COCO-VAL LSP-EXTENDED 3DPW-TEST|1" "A_latest_ema|$O/ckpt/latest_ema.pt|COCO-VAL LSP-EXTENDED 3DPW-TEST|1"; do
  IFS='|' read -r N CK DSS FULL <<< "$spec"
  for DS in $DSS; do
    $P tools/eval_author_protocol.py --dataset $DS --model student_vitpose --checkpoint $CK --output-dir $O/$N/author/$DS --device cuda:0 > $O/$N.author.$DS.log 2>&1 || fail "$N author $DS"
  done
  if [ "$FULL" = 1 ]; then
    $P tools/eval_3dpw_standard.py --model student_vitpose --checkpoint $CK --split test --crop gt_keypoints --precision fp32 --batch-size 64 --num-workers 12 --output-dir $O/$N/3dpw_std > $O/$N.3dpw.log 2>&1 || fail "$N 3dpw std"
    $P tools/eval_ehf_standard.py --model student_vitpose --checkpoint $CK --output-dir $O/$N/ehf > $O/$N.ehf.log 2>&1 || fail "$N ehf"
  fi
  step "$N done"
done
step "TRACKA_EVAL_DONE"
