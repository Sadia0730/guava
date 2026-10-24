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

## Track A2 — PEAR-free fine-tune with more SMPL-X GT, plus its control (A2-ctrl)

**Registered 2026-10-17, before either full run.** Code: PEAR `4c49d99` (`train_student_gt.py`, `student_gt_losses.py`),
guava `b20f17e1` (`tools/eval_3dpw_inloop.py`). Disclosure: one 200-step A2-ctrl smoke test (code check; RUNLOG run 15)
ran before this entry; its 3DPW-**validation** values were seen (live PA-MPJPE 50.84 at step 200). No test split has
been evaluated for A2 or A2-ctrl.

- **Both runs.** Initialised from Track A `best.pt` (live weights, step 82k). 32k steps, lr 1e-4 (backbone x0.1), warmup
  500, cosine to 0.05 x lr; seed 20261009; Track A augmentation, losses and keypoint regressor (r = 0.20 m, 16 vertices);
  EMA 0.9995; 3DPW-val every 2k steps; only `latest.pt` / `best.pt`. Still **no PEAR anywhere**.
- **Output regularisers (both runs).** `reg_scale` = mean over {hand_scale, head_scale} of mean((x − 1)^2), weight 0.1;
  `reg_face` = mean over {body exp, FLAME eye_pose, pose, jaw, eyelid, expression, shape} of mean(x^2), weight 0.1
  (each group averaged over its own elements). Their means are logged on 3DPW-val at every validation.
- **A2-ctrl** (`configs/dnt/student_pearfree_A2ctrl.yaml`): batch 38 BEDLAM2 + 13 UBody + 13 COCO/MPII (as Track A).
- **A2** (`configs/dnt/student_pearfree_A2.yaml`): batch 20 BEDLAM2 + 10 BEDLAM v1 + 8 AGORA + 13 UBody + 13 COCO/MPII.
  AGORA: training split only (validation held out), persons with box height >= 100 px and occlusion / truncation
  <= 50 %, kids excluded, converted to the neutral locked-head SMPL-X (16 betas) by a vertex fit, rows with fit error
  > 15 mm dropped. BEDLAM v1: labels in the BEDLAM2 convention (BEDLAM's locked-head 16-beta processing), 6 fps frames,
  official training scenes only. GT-mesh overlays (10 crops per dataset) are checked before A2 starts. If the data
  build changes any of this, the change is added here before A2 runs.
- **Checkpoint selection (validation only).** As Track A: EMA weights at the best-live-validation step unless their
  validation PA-MPJPE is worse by > 0.3 mm, else the live weights.
- **Evaluation.** Table 3 rows under the author's protocol (`tools/eval_author_protocol.py`, SHA `9838287b`: COCO,
  LSP-Extended, 3DPW test) and EHF (fp32), next to L-real and Track A (`phase3_dnt/trackA_report.py` extended).

**Primary criterion (user-specified), for A2:** A2 closes **>= 50 % of Track A's 3DPW-test PA-MPJPE gap to L-real**
(Track A 59.61, L-real 48.75 mm -> A2 <= 54.18 mm, author's protocol) **AND** COCO PCK@0.05 **>= 0.74**.

**Attribution (registered secondary, also gating the claim "the new data helps"):** A2-ctrl is evaluated identically.
The neutral-output diagnostic (Track A report) and the smoke test suggest the regularisers alone may meet the primary
criterion, so the new data is credited only if **A2 beats A2-ctrl by >= 1.0 mm PA-MPJPE on 3DPW test** (author's
protocol) **and is not worse on COCO PCK@0.05 by more than 0.005**. Outcomes: primary pass + attribution pass -> the new
GT data helps; primary pass + attribution fail -> the gain is the regularisers; primary fail -> reported per metric.

## C2 — baselines for closing the live-driving gap (registered 2026-10-17, before the full baseline run)

Code: `phase3_dnt/c1_live_gap.py --c2` (additive; the C1 path is unchanged), C1's split, avatars, frames (>= 60),
masks, metrics and cached driver outputs (`--drivers-cache` = C1 results, read-only). Driver S = L-real best.pt.
- **B1, render-space alignment (S:B1):** the S render warped by a per-frame 2D similarity (Umeyama) from its ViTPose+
  keypoints to the ViTPose+ keypoints of the **unmasked live frame** (both scores >= 0.5, >= 4 points; else no warp).
  (C1's "aligned" numbers used the GT-masked frame; B1 is the deployable version.)
- **B2, driver-side camera re-fit (S:B2):** per frame, S's camera translation re-solved (linear least squares, 3
  unknowns) so the COCO-17 keypoints of S's EHM mesh (vertex regressor fitted to ViTPose+ on BEDLAM2, r = 0.20 m) project
  onto the live frame's ViTPose+ keypoints (score >= 0.5, >= 4 points, else S's translation); rotation, pose, face, hands
  unchanged.
- **B2 + One-Euro (S:B2+OE):** B2's translation through a causal One-Euro filter with the live demo's defaults
  (min_cutoff 2.0 Hz, beta 0.3, d_cutoff 1.0, 30 fps), applied in (x/z, y/z, log z) so the parameters act on
  radian-like units as in the demo. No parameter is tuned on these clips.
- **Primary (descriptive, per baseline):** raw masked PSNR and LPIPS vs S and vs O (raw); gap closed =
  (B − S) / (O − S) on the identity-mean level, 95 % bootstrap CI over identities (1,000 resamples, seed 0). Secondary:
  keypoint distance, temporal LPIPS, flicker, face / hand crop LPIPS, the same after C1's alignment, added latency.
- **Reading rule (registered):** a baseline is "useful" if its PSNR gap-closed CI excludes 0; "strong" if the point
  estimate is >= 50 %. A baseline that makes things worse is reported as such.
- **Disclosed before registration:** B1's gap closure computed from C1's stored per-frame data (masked-frame keypoints):
  47 % [36, 66] PSNR, 43 % LPIPS; and a 1-clip / 20-frame smoke test of all three (RUNLOG 18), which showed B2+OE
  lagging B2 by ~2 frames. These definitions were fixed after that smoke test and are not changed after the full run.

## PEAR-free student programme: RUN 1 / RUN 2 / RUN 3 / FINAL (registered 2026-10-18, before RUN 1 starts)

Goal (user): a PEAR-free student that beats released PEAR in real time. **No 3DPW training data in any run; no PEAR
outputs or features anywhere.** Every run: fine-tune from **A2-ctrl best.pt EMA**, 32k steps, lr 1e-4 (backbone x0.1),
warmup 500, cosine, seed 20261009, same augmentation, 3DPW-val every 2k; runs differ ONLY in labels / data. Checkpoint
selection as Track A (validation only: EMA at the best-live step unless > 0.3 mm worse). Only latest.pt / best.pt.

**Evaluation (every run, next to PEAR, L-real, A2-ctrl):** author's protocol (COCO, LSP-Extended, 3DPW test;
`tools/eval_author_protocol.py`, SHA `9838287b`), EHF fp32 (PA-PVE all / hands / face, body PA-MPJPE),
COCO-WholeBody val face / hand PCK@0.1 / 0.2 (`tools/eval_wholebody_val.py`), jaw / expression magnitude on the 24
face-check frames (`phase3_dnt/qual_face.py`), face / hand overlay grids (`tools/fa_overlays.py`).

**RUN 1 "face/hand keypoints"** (`configs/dnt/student_pearfree_Fa.yaml`, PEAR `c50688e`): COCO-WholeBody face (68) + hand
(2 x 21) 2D through the EHM mesh (weights 1.0 / 1.0); UBody jaw + expression (FLAME jaw 1.0 + expression 0.5, x2.0;
body exp 0.5, x1.0; SMPL-X -> FLAME map: 0.59 mm mean / 37 % relative displacement error, RUNLOG 21); head crops
(UBody 2.5-3.5 x head-keypoint box, p 0.55; COCO face 1.4-2.0 x, p 0.6); **hand crops 3-4 x the hand box** (p 0.6);
~15 % zoomed rows; regularisers only on rows without face / hand labels. Batch 38 BEDLAM2 + 13 UBody + 13 COCO/MPII.
Disclosed: one 200-step smoke test with 1.6-2.4 x hand crops (RUNLOG 23).
- *Body guard vs A2-ctrl:* 3DPW-test PA-MPJPE <= 50.97 + 0.5 = 51.47 mm and COCO PCK@0.05 >= 0.758 − 0.005 = 0.753.
- *Credit vs A2-ctrl:* EHF hand or face PA-PVE better by >= 0.5 mm, or COCO-WholeBody hand or face PCK@0.1 higher by >= 0.03.

**RUN 2 "+ SMPL-X face/hand data"** and **RUN 3 "+ expert labels"**: Run 1 + their data / labels (each defined in its
own PREREG amendment, with the data audit, before it starts). Criteria **vs RUN 1**: body guard 3DPW-test PA-MPJPE not
worse by > 0.5 mm and COCO PCK@0.05 not lower by > 0.005; credit if EHF hand or face PA-PVE improves by >= 0.5 mm, or
COCO-WholeBody hand or face PCK@0.1 by >= 0.03. A run "beats RUN 1" = guard passed AND credit earned.

**FINAL:** RUN 1 + every run that beat RUN 1 (both combined if both did) + AGORA / BEDLAM v1 data if A2 (with AGORA)
beat A2-ctrl under the A2 PREREG; one 32k fine-tune from A2-ctrl EMA; then the refiner retrained with the v2b recipe on
the FINAL student (PEAR-free). Reported next to PEAR with the same evaluation, plus laptop latency (fp16, CUDA graph,
batch 1; target <= 10 ms student + refiner, measured by the user).

## GUAVA variants for the two pipelines (registered 2026-10-19, before the full fine-tunes)

User decision: **pipeline A ("pure SMPL-X") = Student A + GUAVA (a); pipeline B ("FLAME face") = Student B + GUAVA (b).**
Code: `phase3_dnt/guava_finetune.py`, conversion `phase3_dnt/guava_a_conversion.py`. No SMPL-X-based GUAVA checkpoint
exists (only the released EHM / FLAME `best_160000.pt`), so both variants are fine-tuned from it.
- **GUAVA (b):** EHM (SMPL-X body + FLAME head) with the EHM-only identity extras off: head_scale = hand_scale = 1, joint
  offsets = 0, for every training / test frame.
- **GUAVA (a):** no FLAME; locked-head SMPL-X head (template + 300 shape + 50 expression blend shapes replace the SMPL-X
  2020 rows of EHM's buffers; teeth rows kept), driven by SMPL-X expression + jaw + eye rotations; no head / hand scale,
  no joint offsets. Training-data conversion (measured, `guava_a_conversion_report.json`): tracker FLAME expression ->
  locked-head expression by a least-squares map (basis residual 53 %; on 2,000 tracked frames the face displacement error
  is 1.78 mm mean vs 2.60 mm displacement, i.e. 62 % relative: the SMPL-X head reproduces ~38 % of the FLAME face motion;
  100 components would give 47 % basis residual, not used); per-clip identity betas fitted to the tracker identity
  (0.84 mm mean vertex error). Eyelids have no SMPL-X counterpart.
- **Data:** the 269 EHM-tracked UBody TRAINING clips (C2-T); 10 clips held out (seed 0) for validation / checkpoint
  selection; the C1 test identities are never used for training or selection.
- **Schedule (both identical):** GUAVA's loss and optimiser groups, Adam (0, 0.99), lr 5e-5 linearly to 0.1 x over 20k
  iterations, batch 6, iteration counter offset 160k (late-training loss schedule), seed 0; validation every 2k
  (frame 0 source, every 4th frame target, GUAVA metrics); best.pt by validation SSIM (as GUAVA's trainer) + latest.pt.
- **Smoke tests (disclosed):** 60 iterations each on 12 clips (b: 0.93 it/s, a: 0.84 it/s, 23.8 GB).
- **Evaluation (descriptive, the user's table):** C1 test split (33 tracked identities), GUAVA's own protocol (main/test.py
  semantics: frame 0 = source, every later frame = target, GT x mask on black; main/metrics.py metrics PSNR / L1 / SSIM /
  LPIPS-VGG on the full image), FPS (per-frame animation + rendering) and reconstruction time (avatar creation) with
  `phase3_dnt/guava_timing.py`; extra: face / hand crop LPIPS, keypoint distance, temporal LPIPS. Rows: released GUAVA,
  GUAVA (a), GUAVA (b), each driven by the EHM tracker (upper bound) and by its pipeline's student (A -> a, B -> b);
  cross cells (A -> b, B -> a) optional. No IPS (user).

## RUN A ("dataset labels") vs RUN B ("expert labels") — registered 2026-10-19, before either full run

User design: two runs that differ ONLY in the source of the face and hand labels; Run 1 is the reference. No PEAR, no 3DPW
training data. Configs `configs/dnt/student_pearfree_Run{A,B}.yaml` (PEAR `724656e`): identical except the label
paths. Pipelines (user): **A = Student A -> GUAVA (a) "pure SMPL-X"; B = Student B -> GUAVA (b) "FLAME face"**.
- **Training (identical):** from A2-ctrl best.pt EMA; 32k steps, lr 1e-4 (backbone x0.1), warmup 500, cosine, seed 20261009,
  Run 1 augmentation; batch 64 = BEDLAM2 20 (fixed 400k-row subset, seed 0, shared by both runs and the experts) + AGORA 6
  (62,569 rows) + ARCTIC 6 (27,364 rows) + Motion-X 12 (366,300 rows) + UBody 8 + COCO/MPII 12; COCO-WholeBody face / hand
  2D, head crops and hand crops (3-4 x box), ~8 % zoomed rows; regularisers only on rows without face / hand labels;
  **the UBody hand-mean fix in both** (UBody hand poses are relative to SMPL-X's mean hand; earlier runs used them as
  absolute). Selection on 3DPW-val as before (EMA at the best-live step unless > 0.3 mm worse).
- **Run A labels:** hands = each dataset's own (BEDLAM2, AGORA, ARCTIC, Motion-X, UBody); face = SMPL-X expression (10, v1.1 /
  locked-head basis) + jaw of UBody, Motion-X and AGORA (body-side expression directly; FLAME via the SMPL-X -> FLAME map:
  0.59 mm / 37 % relative error); BEDLAM2 / ARCTIC / COCO: no 3D face / hand labels beyond the hands above.
- **Run B labels:** the datasets' own face / hand labels are not used. Hands = HaMeR + WiLoR on every hand crop of the same
  rows (GT-mesh boxes on the BEDLAM2-format roots, annotated keypoints on UBody / COCO), kept where they agree (< 15 mm
  root-aligned 3D AND < 8 px 2D), label = their rotation mean, left hands mirrored back; disagreeing hands masked. Face =
  SMIRK FLAME expression / jaw / eyelids on every face crop, kept if MediaPipe finds the face and reprojection < 2 x the
  dataset median; no body-side expression. Body labels unchanged.
- **Disclosed:** Run A 200-step smoke test (3DPW-val PA-MPJPE 47.06 at step 200).
- **Evaluation (Run 1, A, B, A2-ctrl, L-real, PEAR):** author's protocol (COCO, LSP, 3DPW test), EHF (all / body / hands /
  face), COCO-WholeBody face / hand PCK@0.1 / 0.2, jaw / expression magnitude on the 24 face-check frames, overlay grids.
- **Choice rule:** body guard = 3DPW-test PA-MPJPE within 0.5 mm of the better of A / B AND COCO PCK@0.05 within 0.005 of the
  better; among runs passing it, the higher **S = WB_hand_PCK@0.1 + WB_face_PCK@0.1 − EHF_hand_PA-PVE[mm] / 20** wins
  (1 mm of EHF hand error = 0.05 PCK). Then the refiner (v2b recipe) is retrained on the chosen student.
- **Avatar table (descriptive):** C1 test split with GUAVA's protocol; rows released GUAVA / GUAVA (a) / GUAVA (b), driven by the
  tracker and by their pipeline's student (see "GUAVA variants" above).

## RUN A vs RUN B — decision (2026-10-21, user)
Both runs failed the registered body guard (A: 3DPW-test PA-MPJPE 48.95 > 48.81; B: COCO PCK@0.05 0.7469 < 0.7496), a case
the rule did not define (AB_REPORT.md). **User decision: RUN A** (S −0.010 vs −0.065; dataset labels). Pipeline A (Student A
+ GUAVA (a)) is the student line taken forward; the next steps (refiner retrain, pose / orientation / shape improvements) start
from RUN A `best.pt` EMA (step 32k).

## SPRINT Phase 2 pilots (registered 2026-10-23, before the runs; validation only, no test set)
Code PEAR ca826b1, configs `configs/sprint/sprint_pilot{,_ctrl,_b38}.yaml`. All: 16k steps from Run A best.pt EMA, lr 1e-4
(backbone x0.1), warmup 500, cosine, bf16, EMA 0.9995, seed 20261023, best/latest only; pure SMPL-X head (SMPLX_NEUTRAL_2020,
200 shape, 50 expression; face labels converted to SMPL-X 2020, `sprint/data/face_conversion_report.json`); Phase 0 data
(AGORA v2, ARCTIC fixed, BEDLAM2 full + tilt repeats, NeuralAnnot COCO / MPII, UBody-EHM with hand weight 0.5, rotation_mix
[[0.4,30],[0.15,90],[0.25,180]], shape_fix). MOYO is not in the pilots (download in progress); COCO + MPII take its 2 rows.
- **P1:** batch BEDLAM2 30 / AGORA 3 / ARCTIC 3 / Motion-X 7 / UBody-EHM 4 / UBody 5 / COCO+MPII 12; ingredients ON: shape token
  (p 0.6, noise 0.05), camera input + full-frame 2D loss (1.0), TALS (tau 0.01), orientation bins (CE 0.1), face-visibility aux (0.1),
  dense-keypoint aux (1.0, BEDLAM2 + AGORA).
- **P2 (control):** P1's data and batch, ingredients OFF.
- **P3 (mix control):** P1's ingredients, batch BEDLAM2 38 / AGORA 2 / ARCTIC 1 / Motion-X 4 / UBody-EHM 3 / UBody 4 / COCO+MPII 12.
- **Validation (every 2k, EMA):** 3DPW-val PA-MPJPE / MPJPE; held-out sources never trained on (`sprint/data/val_holdout/holdout.json`):
  BEDLAM2 val jobs, ARCTIC subject s10, 8 of 75 UBody-EHM identities: PA-MPJPE (22 body joints), upper-body PA, MPJPE, root and
  hand rotation error. Overfitting flag: a source whose held-out error rises by > 5 % from its best while 3DPW-val does not.
- **Decision rule (reported to the user before Phase 3):** the ingredient package is kept if P1 is not worse than P2 on 3DPW-val
  PA-MPJPE (best EMA, tolerance 0.3 mm) and not worse on the held-out sources' PA-MPJPE (tolerance 5 %). Ingredients that can be
  switched at inference without retraining (orientation bins: pred vs raw; shape token: given vs none on holdouts) are also
  reported separately; single-ingredient ablation runs are not part of this pilot (offered to the user). The batch mix is chosen
  between P1 and P3 by the same rule.

## SPRINT round-2 pilots (registered 2026-10-24, before the runs; user decisions after round 1; validation only)
Round 1 result: ingredient package rejected (P1 vs P2: 3DPW-val +1.7 mm, ARCTIC held out +5.2 %); orientation bins dropped
(argmax snapping +6 mm MPJPE). Labels: SMPL-X only (no FLAME anywhere; UBody-EHM removed from training and validation; face labels
from UBody / Motion-X / AGORA / NeuralAnnot converted to SMPL-X 2020). Batch: BEDLAM2 30 / COCO+MPII 10 / MOYO 2 / UBody 9 /
Motion-X 8 / AGORA v2 3 / ARCTIC 2. All runs: 16k from Run A best.pt EMA, seed 20261023, lr 1e-4 (backbone x0.1), warmup 500,
cosine, bf16, EMA 0.9995, only best.pt kept at the end. Configs PEAR 4d33978 configs/sprint/pilot2_*.yaml.
- base: no ingredients, rotation_mix [[0.4,30],[0.15,90],[0.25,180]] (12.6 % of samples rotated > 90 deg).
- (a) base + shape token (SMPL-X shape, p 0.6, noise 0.05); (b) base + camera input + full-frame 2D loss (the 3DPW in-loop
  validator now passes camera input from the 3DPW intrinsics); (c) base + TALS (tau 0.01) + training-only face-visibility (0.1) /
  dense-keypoint (1.0) heads; (d) base with rotation_mix [[0.5,30],[0.2,90],[0.08,180]] (4.1 % > 90 deg).
- Validation (EMA, every 2k): 3DPW-val (pure SMPL-X mesh) and the **hard-pose validation set** = BEDLAM2-val rows tilted > 90 deg
  (4,489 rows, in-loop) + MOYO val split (2 supported-headstand takes; built after launch, evaluated on best.pt); BEDLAM2-val and
  ARCTIC s10 held out also reported.
- Rule: an ingredient (a, b, c) is kept only if, versus base, it is not worse on 3DPW-val PA-MPJPE (tolerance 0.3 mm, best EMA)
  AND not worse on the hard-pose set (BEDLAM2-val-tilt90 and MOYO-val PA-MPJPE, tolerance 3 %). (d) replaces the base mix if it is
  better on 3DPW-val and not worse than the tolerance on the hard-pose set. ARCTIC s10 is watched (overfitting flag: > 5 % rise).

## GUAVA (a20) (registered 2026-10-24, before the full fine-tune; user decision: re-fine-tune GUAVA (a) on the SMPL-X 2020 head)
"Pure SMPL-X" on the ONE model shared by SPRINT, GUAVA and the evaluators (SMPLX_NEUTRAL_2020.npz): EHM's own SMPL-X 2020 head
(no FLAME; EHM's FLAME-template insertion undone; smplx_only_forward), driven by the tracker's smplx_shape (200, no refit), 50
expression + jaw (per-row LBS refit of the tracker's FLAME expression / jaw / eyelids = the SPRINT UBody-EHM face labels; 101
truncated frames refit the same way) and copied eye rotations (eye joints agree to 0.71 mm); no eyelids, head / hand scale 1, no
joint offsets. From the released best_160000.pt. Code phase3_dnt/guava_finetune.py --variant a20, guava_a20_conversion.py,
guava_a20_check.py. Measured vs EHM's FLAME head (300 frames): face 2.61 mm (identity 2.55, expression + jaw 0.58 vs 3.78 motion),
body 0.03 mm. Data: the 269 tracked UBody training clips minus the 56 dropped by the UBody-EHM label filter (training AND validation):
203 training clips (28,420 frames); validation = (a)/(b)'s seed-0 draw minus 2 dropped clips, replaced by the same seeded RNG
(SignLanguage_S62, LiveVlog_S6_Trim19_scene003): 10 clips, 347 frames. Schedule identical to (a)/(b) (20k iterations, lr 5e-5 ->
0.1x, batch 6, best.pt by validation SSIM; latest.pt deleted after a successful run). Disclosed: 200-iteration smoke (val PSNR /
SSIM / LPIPS 23.08 / 0.809 / 0.165 -> 24.32 / 0.820 / 0.158); (a) / (b) unchanged by the edit (identical step-1 losses). Evaluation:
C1 test split, GUAVA protocol, driven by the tracker (converted the same way) and by SPRINT.

### Round-2 pilot results and decision (2026-10-24, applying the registered rule)
EMA weights; 3DPW-val = best EMA over the run (registered criterion); hard-pose set and held-out sources on best.pt (best live
3DPW-val step; MOYO val scored post hoc). PA-MPJPE mm.
| run | 3DPW-val best EMA | 3DPW-val EMA at best.pt | BEDLAM2-val tilt>90 | MOYO val | BEDLAM2-val | ARCTIC s10 |
|---|---|---|---|---|---|---|
| Run A (reference) | 47.6-47.8 | — | 47.8 | 51.4 | 29.2 | 19.1 |
| base | 47.87 | 49.28 | 48.5 | 32.3 | 29.8 | 22.8 |
| (a) shape token | 47.81 | 49.29 | 48.7 | 32.0 | 30.0 | 23.7 |
| (b) camera input + full-frame 2D | **46.50** | **46.61** | 48.3 | **30.9** | 29.9 | 24.6 |
| (c) TALS + aux heads | 47.85 | 49.01 | 48.4 | 32.6 | 29.8 | 23.0 |
| (d) gentler rotation | 47.89 | 48.44 | 47.7 | 35.9 | 29.3 | 21.6 |
Decision: (a) KEEP (not worse on either set); (b) KEEP (better on both; ARCTIC s10 +7.9 % vs base: overfitting flag);
(c) KEEP (not worse within tolerance; small effect); (d) NOT adopted (3DPW-val not better than base; MOYO val +11 %).
Caveats: MOYO val is the same subject and studio as MOYO train (in-domain); BEDLAM2-val tilt > 90 does not improve over Run A in
any run; all runs except (b) lose 3DPW-val after ~2k steps (best EMA at 2k), (b) keeps improving. Ingredients (a)+(b)+(c) were not
yet tested together (round 1's combined run also had orientation bins and no camera input in its 3DPW validator).

## SPRINT final run (registered 2026-10-24 at launch; user decision "start the final SPRINT run now")
PEAR 073af06 configs/sprint/sprint_final.yaml, GPU 3: 128k steps from scratch (ViTPose+-S HF backbone, decoder random, no PEAR weights),
shape token + camera input + full-frame 2D loss + TALS + training-only heads, no orientation bins, pure SMPL-X 2020 head; batch
BEDLAM2 30 / COCO+MPII 10 / UBody 9 / Motion-X 9 / AGORA v2 3 / ARCTIC 1 / MOYO 2; L-real's schedule (AdamW 3e-4, backbone x0.1,
warmup 1500, cosine, bf16, EMA 0.9995). Validation every 2k: 3DPW-val (camera input, pure SMPL-X mesh), hard-pose set (BEDLAM2-val
tilt > 90, MOYO val), BEDLAM2-val, ARCTIC s10; best.pt by live 3DPW-val; reported weights: EMA at the best live step unless worse by
> 0.3 mm (as before); only best.pt kept. Benchmarks afterwards WITHOUT a reference shape (fair vs PEAR) as registered in the SPRINT
brief (author protocol COCO / LSP, 3DPW test, EHF, COCO-WholeBody, orientation evaluation), next to PEAR, L-real, Run A.

## SPRINT-ft insurance run and PEAR-arch baseline (registered 2026-10-24, before launch; user decisions)
- **SPRINT-ft** (PEAR 7e9917b configs/sprint/sprint_ft.yaml, GPU 2): the exact final SPRINT recipe (data, labels, ingredients, validation,
  seed) fine-tuned from Run A best.pt EMA for 48k steps (lr 1e-4, backbone x0.1, warmup 500, cosine, EMA 0.9995); only best.pt kept.
- **Final-model rule:** the final SPRINT model is whichever of SPRINT (from scratch, 128k) and SPRINT-ft is better on 3DPW-val
  (PA-MPJPE, the reported weights per the usual EMA / live rule) AND not worse on the hard-pose set (BEDLAM2-val tilt > 90 and MOYO
  val PA-MPJPE, tolerance 3 %). Both are reported. If neither satisfies both conditions over the other, both are reported and the
  user decides.
- **PEAR-arch, "PEAR architecture trained with our recipe"** (PEAR 7e9917b configs/sprint/pear_arch_final.yaml, GPU 1): PEAR's network
  architecture (ViT-H/16, 631 M, backbone from the original public ViTPose+-H weights usyd-community/vitpose-plus-huge @ 9f36d7ae,
  COCO expert folded into the plain ViT; decoder = PEAR's decoder + SPRINT tokens / heads, random init; no PEAR weights; drop-path
  0.55 as PEAR's released config), trained with exactly SPRINT's data, labels, ingredients, schedule (128k from scratch) and
  evaluation; no pixel-level renderer loss. The like-for-like comparison with PEAR-arch always uses SPRINT from scratch.
- Monitoring: the final run is reported every ~16k steps and flagged if 3DPW-val EMA is not below ~55 mm by 48k steps. GPU 0 is
  left free (shared).
