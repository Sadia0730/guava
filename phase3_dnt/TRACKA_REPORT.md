# Track A — PEAR-free student: final evaluation and the PREREG check (2026-10-16, measured)

**Run.** `configs/dnt/student_pearfree.yaml` (PEAR `5dc6b98`, clean), GPU 1, seed 20261009, 128k steps, 18.9 h; PEAR not
loaded (`pear_loaded: false`). Output `/raid/ubx858/outputs/dnt/trackA/run/` (best.pt = step 82,000, latest.pt = 128,000).
3DPW-validation: best live PA-MPJPE 54.65 mm (step 82k); best EMA 54.58 (step 98k, not saved: best.pt follows the live
weights); final 54.77 live / 54.86 EMA. L-real's best validation was 49.7.

**Checkpoint selection (PREREG, validation only).** EMA at the best-live step validates at 55.23 mm, 0.58 mm worse than
the live weights (> 0.3 mm), so **the live weights of best.pt are the reported Track A checkpoint**. The EMA weights
(of best.pt and of latest.pt) are evaluated as well.

**Evaluation.** `run_trackA_eval.sh` (GPU 2): author's protocol (`tools/eval_author_protocol.py`, SHA `9838287b`),
standard 3DPW test (GT-keypoint crop, fp32), EHF (fp32). L-real was re-run under the same evaluator SHA for the gated
comparison and is identical to the stored Table-3 row (74.51 / 48.75, COCO 0.7724). Table built by
`trackA_report.py` from the summary files (`trackA_table.md`, `trackA_stats.json`).

| Model | COCO PCK @0.05 / @0.1 | LSP-Ext PCK @0.05 / @0.1 | 3DPW test MPJPE / PA (author) | 3DPW test MPJPE / PA (standard) | EHF PA-PVE all / PVE all / hands / face / body PA-MPJPE |
|---|---|---|---|---|---|
| PEAR (released ViT-H) | 0.790 / 0.919 | 0.566 / 0.817 | 73.78 / 46.73 | 73.56 / 45.65 | 47.9 / 66.6 / 11.0 / 8.7 / 49.2 |
| L-real (PEAR-distilled) | 0.772 / 0.921 | 0.484 / 0.755 | 74.51 / 48.75 | 75.83 / 48.46 | 50.6 / 83.4 / 14.8 / 6.4 / 60.0 |
| L-real + refiner v2b | 0.759 / 0.919 | 0.476 / 0.757 | 74.24 / 48.24 | — | 45.2 / 81.8 / 14.8 / 6.3 / 50.2 |
| **Track A best.pt, live (PREREG-selected)** | 0.580 / 0.846 | 0.355 / 0.649 | 86.66 / 59.61 | 81.96 / 53.74 | 84.8 / 119.5 / 13.9 / 64.5 / 58.4 |
| Track A best.pt, EMA (step 82k) | 0.587 / 0.851 | 0.373 / 0.663 | 86.51 / 60.12 | 81.65 / 54.20 | 83.7 / 116.1 / 13.8 / 66.1 / 57.4 |
| Track A latest.pt, EMA (step 128k; descriptive) | 0.607 / 0.864 | 0.388 / 0.677 | 86.50 / 60.00 | 82.14 / 54.26 | 78.3 / 108.3 / 13.7 / 70.5 / 55.9 |
| *diag: L-real, unsupervised outputs neutral* | 0.771 / 0.921 | 0.483 / 0.755 | 74.47 / 48.73 | — | 50.3 / 83.5 / 14.8 / 5.4 / 59.8 |
| *diag: Track A best.pt live, unsupervised outputs neutral* | 0.743 / 0.915 | 0.391 / 0.695 | 80.92 / 50.66 | — | 47.5 / 93.7 / 13.9 / 5.5 / 51.6 |

L-real re-evaluated with the current evaluator (SHA 9838287b): 3DPW test 74.51 / 48.75, COCO @0.05 0.7724; identical to the stored row: True.

| Weights | ΔPA-MPJPE (≤ +1.0) | ΔMPJPE (≤ +2.0) | ΔCOCO PCK@0.05 (≥ −0.01) | Circularity answered |
|---|---|---|---|---|
| **Track A best.pt, live (PREREG-selected)** | +10.87 ✗ | +12.15 ✗ | -0.1920 ✗ | **FAIL** |
| Track A best.pt, EMA (step 82k) | +11.38 ✗ | +12.00 ✗ | -0.1858 ✗ | **FAIL** |
| Track A latest.pt, EMA (step 128k; descriptive) | +11.25 ✗ | +11.99 ✗ | -0.1652 ✗ | **FAIL** |
| *diag (not gated): neutral vs neutral* | +1.92 | +6.45 | -0.0282 | *would still fail* |

## Verdict: circularity NOT answered (FAIL on all three conditions, for every set of weights)

## Why — diagnosed after the verdict (descriptive, does not change it)

1. **Every output that only PEAR supervised has drifted to degenerate values.** Track A's losses (BEDLAM2 GT, 2D
   keypoints, UBody pseudo-GT) all go through the kinematic SMPL-X mesh, so nothing constrains EHM's `head_scale`,
   `hand_scale`, the body-side `exp`, or any FLAME parameter. In L-real these came from the face / body-parameter
   distillation terms. On 50 EHF crops: head_scale 0.39 (L-real 1.05), hand_scale 0.26 (0.91), body `exp` mean |x| 9.7,
   max 46 (0.06), FLAME jaw / eye / eyelid 0.31 / 0.40 / 0.42 (0.02 / 0.01 / 0.06). The evaluated (EHM) mesh therefore has
   a head shrunk to ~40 % and a distorted face: EHF face PA-PVE 64.5 mm (L-real 6.4), and the COCO face keypoints
   collapse (ears 0.91 -> 0.14, eyes 0.96 -> 0.41, nose 0.95 -> 0.59 PCK@0.05). The H36M head / neck joints of 3DPW come
   from the same head vertices.
2. **Neutral-output diagnostic** (`make_neutral.py`, `run_neutral_diag.sh`): those decoders set to their neutral constants
   (scales 1; `exp` and all FLAME parameters 0), no free parameter, applied identically to L-real and Track A. L-real
   barely moves (74.47 / 48.73, COCO 0.771), so the neutral face is a fair common reference. Track A then reaches
   80.92 / 50.66 and COCO 0.743: **about 9 of the 10.9 mm PA-MPJPE gap is the unsupervised head / face / scale outputs,
   but even then Track A is +1.9 mm PA-MPJPE, +6.5 mm MPJPE and −0.028 COCO PCK behind L-real, and would still fail all
   three conditions.**
3. **What remains is limbs and depth, not the torso.** COCO per keypoint (neutral vs neutral): shoulders and hips are
   better than L-real (+0.02 to +0.10, the effect of the detector-fitted regressor), wrists −0.07, knees −0.05/−0.06,
   ankles −0.12/−0.15. LSP-Extended (limb-dominated) −0.09 PCK@0.05. MPJPE (+6.5) falls behind far more than PA-MPJPE
   (+1.9), and EHF PVE all is worse (93.7 vs 83.5), i.e. a global scale / depth error. EHF body PA-MPJPE is
   *better* than L-real's (51.6 vs 59.8), close to L-real + refiner v2b (50.2).

**Interpretation (registered: "Fail -> PEAR distillation contributes beyond the labels").** It does, in two parts:
(a) PEAR is the only source of the EHM-specific outputs (head and hand scale, face); without them the mesh is
degenerate. That is a gap in the label set, not evidence about body-pose learning. (b) Removing (a), a real body gain
remains: extremities (wrists, ankles) and 3D scale / depth. No further tuning of the PEAR-free run is done in this phase,
as registered.
