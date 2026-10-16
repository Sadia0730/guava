# Driven, Not Tracked — pre-registered criteria (committed before each experiment runs)

Both repositories: branch `driven_not_tracked`. All new outputs: `/raid/ubx858/outputs/dnt/`. Test splits are never used
for tuning or checkpoint selection; checkpoints are selected on 3DPW validation as before.

## Track A — PEAR-free student (fairness control)

**Registered 2026-10-15, before the smoke test and the full run.**

- **Model and schedule:** the L-real recipe (ViTPose-S backbone, PEAR head architecture, 128,000 steps, batch 64 = 38
  BEDLAM2 + 13 UBody + 13 COCO/MPII, lr 3e-4 / backbone x0.1, warm-up 1,500, cosine to 5%, bf16, seed 20261009, the same
  augmentation, 3DPW-validation every 2,000 steps, `best.pt` by validation PA-MPJPE), with **no PEAR anywhere**: no
  teacher weights loaded, no distillation or feature terms (`configs/dnt/student_pearfree.yaml`, `teacher_checkpoint: none`).
- **Supervision:** BEDLAM2 SMPL-X GT (unchanged terms); COCO/MPII 2D keypoint loss through the detector-fitted vertex
  regressor with hips re-enabled; UBody SMPL-X pseudo-GT (root, body, valid hands; lower body only where `full_body`)
  + UBody 2D keypoints; no shape / camera terms on UBody rows; **no face supervision** (expression, jaw, eyes) in this run.
- **EMA:** decay 0.9995 per step; the EMA weights are validated at every validation and saved in every checkpoint. The
  reported Track A checkpoint is the EMA weights at the step of the best (live) validation PA-MPJPE, unless the EMA
  validation at that step is worse by more than 0.3 mm PA-MPJPE, in which case the live weights are reported — this
  choice is made on validation only and stated in the report.

**Keypoint regressor (pre-step).** Fitted on BEDLAM2 training renders (ViTPose+ small, COCO expert) against the GT
meshes; evaluated on BEDLAM2 validation renders against the same detector.
- **Pass (original, 2026-10-15):** mean detector distance (torso units) of the hips and of the shoulders down by
  **>= 50%** against the current SMPL-X keypoint mapping. If it fails, the 2D loss keeps the current mapping with hip
  weight 0 (as L-real) and the failure is reported; the run still proceeds.
- **Measured (2026-10-15, BEDLAM2 val, 5,000 rows):** r = 0.12 m / 8 vertices: hips -48%, shoulders -12% (fail);
  r = 0.20 m / 16 vertices: hips -57% (0.21 -> 0.09 torso units), hip bias ("down") +0.18 -> +0.01, shoulders -14%
  (0.074 -> 0.063), shoulder bias +0.026 -> +0.005.
- **AMENDMENT (2026-10-16, approved by the user, written before the smoke test and the full run):** the criterion is
  replaced by **"hip distance down >= 50% AND hip and shoulder bias (mean signed torso-frame offset) < 0.02 torso
  units"**. Reason: the shoulders were already at the detector's noise floor before fitting (0.07 torso units ~ 4 px,
  the same as nose / eyes / wrists), so a 50% cut in their distance is unreachable by any keypoint definition; what a
  definition can and should remove is the systematic bias, which the regressor removes (+0.026 -> +0.005). The r = 0.20 m,
  16-vertex regressor meets the amended criterion and is used, with hips re-enabled (hip weight 1.0):
  `/raid/ubx858/outputs/dnt/regressor_r020/kp_regressor_coco17.npz`.
- **Checkpoints (2026-10-16, disk rule):** every new run writes only `latest.pt` and `best.pt`
  (`logging.periodic_checkpoints: false`).

**Primary criterion — "circularity answered":** the PEAR-free student is within
- **1.0 mm PA-MPJPE** and **2.0 mm MPJPE** of L-real on **3DPW test** (author's protocol, `tools/eval_author_protocol.py`,
  the same evaluator SHA as every Table-3 row), and
- **0.01 COCO PCK@0.05** of L-real (author's protocol, COCO val 4,648 persons).

Reference values (L-real, measured): 3DPW test 74.51 / 48.75 mm; COCO PCK@0.05 0.772. All three conditions must hold.
Secondary (reported, not gated): LSP-Extended PCK, EHF (PA-PVE all / hands / face, body PA-MPJPE), standard 3DPW test,
and the same numbers for the live (non-EMA) weights.

**Interpretation registered in advance.** Pass -> the student's gains over earlier checkpoints are not an artefact of
distilling PEAR (PEAR is not needed to reach them). Fail -> PEAR distillation contributes beyond the labels; the gap
is reported per metric and no further tuning of the PEAR-free run is done in this phase.

## Track B — C1 live-driving gap (G1)

**Registered 2026-10-16, before B4 (rendering and metrics) runs.** Code: `phase3_dnt/c1_live_gap.py` (guava
`driven_not_tracked`); split: `/raid/ubx858/outputs/dnt/c1/split.json` (SHA-256 `d760fb86…`, 36 UBody intra-scene test
identities, one clip each).

- **Setup.** One GUAVA avatar per identity, built from the first tracked frame of its clip (EHM-Tracker source record).
  Every variant renders the same avatar with the same identity parameters (EHM per-clip shape, FLAME shape, joints
  offset, hand scale) and the same StyleUNet noise seed; only the per-frame driving parameters differ:
  O = EHM-Tracker (offline, oracle), S = L-real `best.pt`, S+R = L-real + refiner v2 (`refiner_v2_Lreal/checkpoints/latest.pt`,
  the only refiner used), P = PEAR released ViT-H; per-part swaps S+O:{body, hands, face, cam} and leave-one-out
  O+S:{body, hands, face, cam}. Drivers see the same 256 px union-box crop.
- **Frames.** Future frames only: tracked frames with index >= 60 (>= 2 s at 30 fps) after the source.
- **Primary metrics.** Masked PSNR and masked LPIPS-VGG (GT upper-body mask = EHM's StyleMatte alpha; both images
  multiplied by the mask; PSNR over mask pixels), RAW (no alignment), per-identity mean over its future frames.
- **Statistic.** Paired difference per identity, d_PSNR = PSNR(O) − PSNR(S) and d_LPIPS = LPIPS(S) − LPIPS(O)
  (positive = O better); mean over identities with a 95% bootstrap CI over identities (1,000 resamples, seed 0).
- **G1 decision.** **GO** if d_PSNR >= 0.5 dB with its CI excluding 0, **or** d_LPIPS >= 0.01 with its CI excluding 0.
  Otherwise **NO-GO**.
- **Interpretation registered in advance.** GO -> the live driver (S) loses a meaningful amount of rendering quality
  against the offline tracker, so closing the driving gap is worth pursuing; the per-part swaps attribute the gap
  (descriptive, no threshold). NO-GO -> S already drives GUAVA about as well as the offline tracker on this split, and
  the remaining quality gap is in the avatar / renderer, not the driver.
- **Validity conditions.** At least 30 of the 36 identities must have a successful EHM track with >= 60 future frames;
  otherwise the result is reported as inconclusive and I ask before changing the split. Failed tracks are listed.
- **Secondary (reported, not gated).** The same statistics after per-frame 2D similarity alignment; SSIM; face-crop and
  hand-crop LPIPS (EHM head / hand boxes); keypoint distance (ViTPose+ small, COCO expert, render vs real frame,
  real-frame score >= 0.5, torso units); temporal LPIPS and flicker; S+R and P against O; driver time per frame.
- **ADDENDUM (2026-10-16 09:50, after 15 of 36 clips were scored, before the remaining 21; not part of the G1 rule):**
  one clip (Interview_S18_Trim3) showed the oracle far below every learned driver (O 13.9 dB vs S 17.5 dB), i.e. a
  failed EHM track. The primary G1 analysis stays exactly as registered (all identities). A **sensitivity analysis**
  is added: identities whose oracle fit is poor by its OWN measure — O's mean keypoint distance (ViTPose+, render vs
  real frame) **> 0.10 torso lengths** — are excluded and the G1 statistic is recomputed. The threshold is fixed here
  from the smoke clip (O 0.029) without looking at the other clips' O keypoint distances; flagged clips are listed.
  EHM frames that GUAVA's loader cannot read (NaN crop boxes) are dropped per clip and counted.
- **Caveats stated in advance.** O is offline-smoothed and its identity is fitted over the whole clip (non-causal);
  GUAVA's weights were trained on undisclosed UBody clips, so the split is held out w.r.t. our students, not provably
  w.r.t. GUAVA; masked PSNR over upper-body framings is dominated by the torso area.
