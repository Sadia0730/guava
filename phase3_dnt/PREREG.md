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

### Final-model decision (2026-10-26, applying the registered rule)
SPRINT from scratch: best live step 128k (live 48.97, EMA 49.04 -> EMA reported); BEDLAM2-val tilt>90 48.22, MOYO val 35.68.
SPRINT-ft: step 48k (live 45.89, EMA 45.88 -> EMA); BEDLAM2-val tilt>90 42.46, MOYO val 31.15. SPRINT-ft is better on 3DPW-val
AND on the hard-pose set -> **the final SPRINT model is SPRINT-ft (EMA, step 48k)**. Both are reported; the like-for-like comparison
with PEAR-arch uses SPRINT from scratch.

## Refiner (v2b) on SPRINT-ft (registered 2026-10-26, before training; user-approved step)
- **Student (frozen):** SPRINT-ft EMA step 48k (`eval_programme/sprint_ft/best_ema.pt`, sprint_raw, pure SMPL-X 2020 head), always
  run with the camera input cond={"cam": cam_features} exactly as in its training (BEDLAM2: real intrinsics, real 1; COCO / MPII:
  focal = image width, real 0).
- **Error bank:** rebuilt from SPRINT-ft (the L-real refiner is unusable on another student's features): the same 20,000 evenly
  spaced BEDLAM2-val crops (bedlam2_gt val, `--build-error-bank 20000`) as L-real's bank, with the camera input.
- **Recipe:** refiner v2b unchanged (configs/phase1_task3/refiner_v2b_Lreal.yaml): no-render RenderRefiner (4.18 M), seed 20261012,
  20,000 steps, batch 51 BEDLAM2 (bedlam2_gt train, same augmentation) + 13 COCO/MPII real rows, start 50/50 student / GT + bank
  error, AdamW 3e-4 wd 0.05, warmup 1000, cosine to 0.05, clip 1.0, GT losses root / body rot, 3D joints, camera, 2D (1.0 each),
  real-image 2D loss 1.0 with the v2b dataset-specific hip regressor (hip weight 1.0). Only changes: the student class / camera input
  (new config keys; defaults reproduce the L-real path). Checkpoint = the final step (as v2b; validation only monitors).
- **Evaluation:** the full suite (`phase3_dnt/run_full_eval.sh`, sprint_raw, smplx head, orientation) with the refiner, in setting (i)
  known camera (CAMERA=1) and (ii) no extra information (CAMERA=2, focal = image width everywhere), next to SPRINT-ft without the
  refiner and PEAR; plus 3DPW-val (in-loop validator, camera on) with / without the refiner; latency L40S batch 1 eager.
- **Question:** does the refiner close the 3DPW-test PA-MPJPE gap to PEAR (46.73 mm) in setting (ii)?
- **Decision rule:** the refiner is kept iff it improves 3DPW-test PA-MPJPE in setting (ii) AND COCO PCK@0.05 (setting ii) is not
  lower by > 0.005. No tuning on any test set; one training run.

### Final-model decision, user override (2026-10-26)
User: SPRINT-ft cannot be used because its initialisation lineage (Track A -> A2-ctrl -> Run A) was trained with a FLAME head
(Run A: face labels mapped to FLAME, COCO-WholeBody 2D through the EHM mesh, FLAME regularisers). **The final SPRINT model is SPRINT
from scratch** (configs/sprint/sprint_final.yaml, EMA at step 128k; no FLAME anywhere in its training). SPRINT-ft is reported only
as an ablation. All downstream steps (refiner, head-to-head with released PEAR, avatar table, latency) use SPRINT from scratch.

### Correction (2026-10-26, user decision, before any SPRINT-ft refiner result was evaluated)
The refiner target changes from SPRINT-ft to **SPRINT from scratch** (final SPRINT model; SPRINT-ft is not used because its Run A
ancestry was trained with a FLAME head): `/raid/ubx858/outputs/dnt/sprint/final/checkpoints/best.pt` step 128k, EMA weights
(`eval_programme/sprint_scratch/best_ema.pt`), sprint_raw, camera input as in its training. The SPRINT-ft refiner run was stopped at
step ~6k (outputs left in `/raid/ubx858/outputs/dnt/refiner_sprint_ft/`, not evaluated, not used). Everything else in the entry above
applies unchanged to SPRINT from scratch: its own error bank (same 20,000 BEDLAM2-val crops, camera on), the identical v2b recipe
(PEAR configs/sprint/refiner_v2b_sprint_scratch.yaml), the full suite in settings (i) and (ii) next to the no-refiner baselines
`eval_programme/sprint_scratch` (i) and `sprint_scratch_camwidth` (ii) and PEAR, 3DPW-val in-loop with / without, latency, the same
question (gap to PEAR 46.73 in setting ii) and the same decision rule (keep iff 3DPW-test PA-MPJPE (ii) improves and COCO PCK@0.05
(ii) is not lower by > 0.005).

## GUAVA-FB: GUAVA extended to the full body (registered 2026-10-28, before any conversion run or training; user brief)
GUAVA-FB is GUAVA (Zhang et al., "GUAVA: Generalizable Upper Body 3D Gaussian Avatar") extended to the full body; all credit for
the architecture, losses and training recipe goes to GUAVA. Pure SMPL-X 2020 head (SMPLX_NEUTRAL_2020, 200 shape, 50 expression),
no FLAME, initialised from GUAVA (a20) `guava_a20/run/checkpoints/best.pt` (iteration 2k), same UV layout and visibility outputs.
**Storage (user decision 2026-10-28, /raid at 26 GB free):** all GUAVA-FB data and outputs go to
`/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/` (/raid is only read). **GPUs (user):** 0, 1, 2.

### Camera patch and its regression check
Per-sample perspective camera: perspective crop (virtual camera rotated onto the crop centre ray, homography warp, principal point
at the crop centre) + per-sample `invtanfov` = 2 f / L, used in all 4 places GUAVA hard-codes `invtanfov` (AVATAR_B_PLAN §1 #3) and
an empty-box guard in `cal_box_loss`. Applied as a monkeypatch (GUAVA files untouched).
- **Regression check (must pass before any training):** GUAVA (a20) best.pt on its own 10 UBody validation clips (347 frames)
  with the patch active and every sample's invtanfov = 24 must reproduce the unpatched validation: |dPSNR| <= 0.01 dB,
  |dSSIM| <= 1e-4, |dLPIPS| <= 1e-4.
- **Correctness check:** re-render the converted SMPL-X through GUAVA's own mesh rasteriser / camera on 50 rows per exact-label
  source (BEDLAM2, 4D-DRESS): mean vertex reprojection error vs the source camera's projection < 1 px at 512.

### Data (each source: size, licence, SMPL-X availability, views, 10 overlays, reported before training)
| source | share | labels | pairs | split |
|---|---|---|---|---|
| BEDLAM2 videos | 30 % | exact SMPL-X (locked head -> SMPLX_NEUTRAL_2020: shape16 -> betas200 map used by SPRINT), exact cameras | source / target = different frames of a clip | AVATAR_B_PLAN strict split: 40 test subjects (not used now), 10 val clips, train pool (pilot rule, <= 2 clips / subject, cap 1,500 clips) |
| Motion-X full-body | 20 % | dataset SMPL-X kept if the clip passes the alignment filter, else SPRINT (from scratch) + refinement refit, filter applied again | different frames | 20 val clips held out (seed 0, after filtering) |
| 4D-DRESS | 20 % | dataset SMPL-X (v1.1, gendered, 10 betas, PCA hands) refitted to SMPLX_NEUTRAL_2020 (one shape per subject, per-frame poses); calibrated cameras (4 views) | **source and target from DIFFERENT cameras** of the same subject (any take / outfit of that subject) | 4 subjects held out for validation (2 female, 2 male; SHA-1 of the id, lowest), never in training |
| UBC-Fashion | 15 % | SPRINT (from scratch) + refinement fit, ONE shared shape per video, temporal smoothing | different frames | official TRAIN videos only; all 100 official test videos (incl. GART's 6) excluded; train videos whose identity matches a test video (ArcFace, below) also excluded |
| UBody (EHM-tracked, a20 data) | 15 % | a20 conversion (56 bad clips excluded) | as a20 | a20's own 10 validation clips |

- **Fitting model:** SPRINT from scratch (EMA, step 128k; `eval_programme/sprint_scratch/best_ema.pt`), camera input focal = image
  width (setting ii), `sprint_avatar_fit.py` refinement (DWPose keypoints + StyleMatte mask). SPRINT-ft is not used.
- **Masks:** StyleMatte (EHM-Tracker `stylematte_synth.pt`, GUAVA's matting) on every frame of every source (4D-DRESS's rendered
  masks only as a check of StyleMatte).
- **Alignment filter (fitted / dataset labels of real video: Motion-X, UBC):** per frame (a) projected mesh outside the matte < 5 %
  of mesh pixels and (b) max per-limb DWPose-to-joint error < 3 % of body height (keypoints with score > 0.5, >= 15 of them);
  a clip is kept iff >= 80 % of its frames pass. Thresholds fixed now; no tuning on any evaluation set.
- **4D-DRESS SMPL-X refit check:** mean vertex distance converted vs original mesh < 5 mm per subject (median over frames); a
  subject failing it is dropped and reported.
- **NEVER in training:** People-Snapshot, NeuMan, X-Humans, UBC-Fashion test videos, held-out 4D-DRESS subjects (also any 3DPW).
  **Identity overlap check** (reported): ArcFace (antelopev2 glintr100, `onnx_arcface_torch.py`) max cosine between face crops of
  each training identity and each evaluation identity; a pair with cosine > 0.5 is a match; a matching training clip / video is
  removed from training before it starts. Evaluation sets not yet on disk are checked when they arrive; if a match is then found,
  the affected evaluation subject is reported as contaminated.

### Identity retention and losses
- Shape lock: one SMPL-X 2020 shape per clip / subject (`id_share_params.smplx_shape`), never per frame.
- `lambda_head_crop` 0.25 -> **0.5** (faces are about 60 px in a full-body 512 render); other GUAVA weights unchanged.
- ArcFace identity loss on face crops (research use only; insightface models are licensed non-commercial research only):
  1 - cos(ArcFace(render face crop), ArcFace(GT face crop)), aligned by the projected SMPL-X eye / nose / mouth landmarks, weight
  **0.05**, only for targets whose GT face box is >= 48 px and whose face faces the camera (head forward axis within 60 deg of the
  camera axis).
- Avatar built once per source and reused for all its targets: batch = 2 sources x 3 targets (6 renders, as GUAVA's batch 6).

### Training
- Smoke test: 200 iterations + 10 renders. Pass iff every loss is finite, the 10 renders overlay the GT silhouette (visual check),
  peak memory < 44 GB and the regression check above passed. If the smoke test and data checks pass, training starts without
  asking (user instruction).
- Run: 30k iterations, Adam (GUAVA's param groups, betas (0, 0.99)), lr 5e-5 -> 0.1x linear, iteration offset 160k, seed 0.
- Validation every 2k iterations on: held-out 4D-DRESS subjects (frame 0 of camera A as source, other cameras as targets),
  10 held-out BEDLAM2 clips, 20 held-out Motion-X clips, a20's 10 UBody val clips; GUAVA metrics (full 512 image, GT x mask on
  black) plus face-crop LPIPS.
- **best.pt rule:** a checkpoint is eligible iff on UBody val SSIM >= a20 SSIM - 0.002 AND face-crop LPIPS <= a20 + 0.002 (a20
  measured by the same validator at iteration 0). Among eligible checkpoints, best = highest mean SSIM over the three full-body
  tracks (4D-DRESS, BEDLAM2, Motion-X). Only best.pt is kept at the end. If no trained checkpoint is eligible, this is reported as a
  failed criterion.

### Evaluation (after training; each with its published protocol; not used for any choice)
People-Snapshot (InstantAvatar protocol, 4 subjects; SMPL -> SMPL-X conversion stated), NeuMan (bike, citron, jogging, seattle;
official split; masked metrics), X-Humans (00028, 00034, 00087; given SMPL-X, no test-time fitting), UBC-Fashion (GART's 6 test videos
and frame rule). Source image = the first training frame (declared). Metrics PSNR, SSIM, LPIPS, L1, CSIM (ArcFace identity over
frames), tLPIPS; with and without test-time pose fitting. Baseline numbers copied from the papers where available (GART,
GaussianAvatar, ExAvatar) and run where stated. Every table states that GUAVA-FB uses ONE image while the baselines train on video.

### GUAVA-FB addendum (registered 2026-10-28, after the data checks, before the smoke test and training)
Data checks are complete (reports in `O/reports/<source>/`). The changes and precisions below were fixed without any training run
and without looking at an evaluation set.
- **Correction:** GART's 6 UBC videos (91+bCFG1jOS, 91Iegdp9HFS, 91Ile3zLhMS, 91QFEra7jDS, 91WvLcNpdzS, 91fxYsir49S) are in UBC's
  official TRAIN list, not in the test list. All 6 are excluded from training, and their identities count as evaluation
  identities for the ArcFace overlap check (100 test + GART 6). This removed 362 of 494 checked train videos.
- **Alignment filter (b), how it is read:** for each limb, average the DWPose-to-joint errors over the limb's keypoints; a frame
  passes if the largest of these limb means is < 3 % of body height. Implementation: `fit_ubc.frame_filter` + `silhouettes`, the
  same for UBC and Motion-X. Reading it as the largest single-keypoint error would pass almost nothing (Motion-X: 0 % with
  dataset labels, 0.07 % with refits).
- **Known defect, not fixed for this run:** the shared DWPose-134 -> SMPL-X regressor maps the heels to joints 10 / 11 (ball of the
  foot) instead of heel vertices (8846 / 8635). Effects:
  - The filter is stricter than intended. On Motion-X, fixing it would raise clip pass rates from 1.96 % to 3.84 % with dataset
    labels and from 22.9 % to 30.4 % with refits.
  - The heel term of the UBC / Motion-X refits is slightly biased toward the toes.
  
  This is kept as-is because UBC and Motion-X were filtered identically and every kept frame passed visual overlay checks. It is
  reported as a limitation; a later version may refit with the fix.
- **Dropped frames** (`O/data/<source>/drop_frames.json`, applied by the trainer):
  - UBC: 38 frames where the fitted body faces the wrong way (front / back flip; fitted facing disagrees with the DWPose
    shoulder order);
  - 4D-DRESS: rows whose body is cut off by the image edge by more than 5 % (`trunc_frac` > 0.05; 521 train and 100 val rows);
  - Motion-X: none (kept clips contain only passing, non-flipped frames).
  
  BEDLAM2 rows with any mesh vertex closer than 0.2 m to the camera were not converted.
- **4D-DRESS:** 5 takes of 00135 / Outer are missing from the extracted download and are not used.
- **Motion-X:**
  - sampling and labels:
    - 1,500 candidate clips, equal quota over 7 subsets, at most 1 clip per source video (source video = subject = one shape);
    - frames every ~0.5 s, at most 40 per clip;
    - dataset SMPL-X v1.1 labels convert exactly to 2020: zero-padded betas, with one shape per clip = the mean of the clip's betas;
  - kept clips:
    - 310 in total: 27 with dataset labels and 283 with SPRINT refits;
    - 20 val clips (seed 0, never HuMMan, because its clips share performers) and 290 train clips.
- **UBC:** shape is per video. An `identity_group` field ties videos of the same person together; the group never crosses the
  train / evaluation boundary.
- **Correctness check result** (registered sample = the first 10 clips × 5 rows in seed-0 order): pass on both sources.
  - BEDLAM2: 0.37 px mean.
  - 4D-DRESS: 0.55 px mean.
  - Over all clips (the script did not stop at 50 rows), BEDLAM2 is 0.37 px over 1,510 clips with none above 1 px; 4D-DRESS is
    0.57 px over 516 clips, with 6 clips between 1.0 and 1.64 px. All clips are kept.
- **Validation sets (fixed for compute):**
  - targets per group: 4D-DRESS at most 60 per held-out subject (evenly strided over its other-camera rows); BEDLAM2 and Motion-X
    every 4th frame from the 2nd, at most 20 / 10 per clip;
  - UBody: exactly a20's rule (all 347 frames);
  - `torch.manual_seed(0)` before every validation forward / render, so validation is deterministic (StyleUNet noise);
  - a20's reference values for the best.pt rule are measured by this same validator at iteration 0.
- **Staging:** at launch, training data is copied from the vault to RAM disk (`/dev/shm/ubx858_guava_fb/data`; the vault reads
  14 reads/s single-threaded). The vault keeps the master copy, checkpoints and results.

### GUAVA-FB amendment 2 (registered 2026-10-28 ~23:00, before v1b): 4D-DRESS pairs by subject AND outfit
- **Found from validation renders** (iterations 4k / 6k; validation data only, no evaluation set): grouping 4D-DRESS cross-view
  pairs by subject alone mixed each subject's Inner and Outer outfits. 71 of 141 logged 4D-DRESS training targets (50 %) wore a
  different outfit from their source, and so did part of the 4D-DRESS validation targets. Such a target cannot be reached from the
  source image.
- **Change:** a cross-view pairing group is now subject + outfit (index fields `subject`, `outfit`), for both training and
  validation. Source and target still come from different cameras, and may come from any take of the same outfit.
  - 4D-DRESS validation: for each held-out subject-outfit group, the source is the first frame of the lowest-numbered camera of
    the group's first take (by name), and the targets are the other cameras' rows of that group, evenly strided and capped at
    30 per group (so still at most 60 per held-out subject).
  - Option `--cross-view-group` (v1 = `subject`, v1b = `subject+outfit`).
- **v1:** stopped at iteration ~8.5k (validation at 2k, 4k, 6k and 8k; no checkpoint eligible; latest.pt at step 8k kept, nothing
  deleted). It is superseded and is used for no choice beyond this fix.
- **v1b:** the v1 recipe with this change only (same seed, data, losses, schedule, best.pt rule). The a20 reference is measured
  again at iteration 0 of v1b by the same validator, because the 4D-DRESS validation set changed.
- **Validation renders** (`val_render.py`, separate process, visual only; it computes no metric and changes nothing in
  training): for each new validation, a fixed subset (first 8 4D-DRESS groups, otherwise first 4 groups; 2 targets each) is
  rendered from that step's checkpoint into `<run>/val_renders/iter_XXXXXX/`.

### GUAVA-FB amendment 3 (registered 2026-10-29 ~01:40, before any pilot run): clothing-detail pilots A and B, 8k comparison
User request (2026-10-29): add the reviewed changes, train 8k iterations and compare with v1b. These are training-only changes:
the model, its weights' shapes and inference are unchanged. Not in these pilots: adversarial loss, pose-dependent offsets,
1024 px colour input (separate tests later).
- **Pilot A** = v1b recipe plus:
  - (1) **source frame as a target:** for non-4D-DRESS sources, with prob. 0.75 one of the 3 targets is the source frame
    (≈ 25 % of their targets). Cross-frame targets stay.
  - (2) **same moment, other camera:** for 4D-DRESS, each target is, with prob. 0.5, the same frame of the same take from
    another camera. This keeps the user's different-camera rule and removes cloth-state differences.
  - (3) **true 2× zoomed targets:** for 4D-DRESS and UBC, each target with prob. 0.33 is a zoomed target. The camera is rotated
    about its centre onto a chosen ray with invtanfov × 2, and the 1024 crop is warped by the exact homography to 512.
    - Centre: the head with prob. 0.25, otherwise a random matte pixel of the clothing band (head bottom to 70 % of the
      person's height).
    - Not used if the window holds < 15 % person pixels.
    - Checked before training (`zoom_check.py`, 48 targets): projection vs homography ≤ 0.0005 px; mesh-in-matte precision
      0.982 zoomed vs 0.974 unzoomed.
  - (4) **UBody share 25 %** (bedlam2 26.47, motionx 17.65, dress4d 17.65, ubc 13.24, ubody 25).
  - Options: `--self-target-p 0.75 --same-frame-p 0.5 --zoom-sources dress4d,ubc --zoom-p 0.33 --zoom-head-p 0.25`, `--mix` as
    above.
- **Pilot B** = pilot A plus:
  - clothing-box L1 + LPIPS, weight 0.5, like GUAVA's head crop; the box covers the projected torso / hip / thigh / upper-arm
    vertices (2-98 % quantiles);
  - Sobel-gradient L1, weight 0.5, on source-frame targets only (exactly aligned);
  - silhouette L1 between the Gaussians' coverage (render with constant features) and the matte, weight 0.5.
  - Options: `--cloth-crop 0.5 --edge-weight 0.5 --sil-weight 0.5`.
- **Both pilots:**
  - 8k iterations on the 30k schedule (`--stop-iter 8000`, lr 5e-5 → 0.1× over 30k, as v1b), seed 0, validation every 2k
    with the registered validator and best.pt rule (unchanged).
  - latest.pt at 8k is kept for the comparison; the winning pilot can later be resumed to 30k (user decision).
  - Smoke test first.
- **Comparison** (validation data only, never an evaluation set):
  - (1) registered validation at 8k vs v1b at 8k (per track PSNR / SSIM / LPIPS / face LPIPS; UBody eligibility);
  - (2) clothing-box LPIPS / SSIM / L1 on the 4D-DRESS, BEDLAM2 and Motion-X validation targets (512, same box as the loss);
  - (3) zoomed clothing LPIPS / SSIM / PSNR on the 4D-DRESS validation targets: 2× zoom at the median matte pixel of the
    clothing band, against the 1024 crop;
  - (4) Gaussian coverage vs matte (missed / spilled fraction);
  - (5) flicker: 4 held-out 4D-DRESS groups (one Outer group per subject), first non-source camera, the take's frames with 3
    slerp in-betweens; mean LPIPS between consecutive renders; side-by-side mp4;
  - (6) cost: Gaussians after pruning, per-frame render time (deform + rasterise + refine; median of 200) and avatar build
    time (median of 20), same GPU, after warm-up.
  - Checkpoint-based measures (2-6) use v1b's step-10k checkpoint (a copy taken before 12k overwrote it), which has 2k more
    iterations than the pilots: a conservative baseline.
- **Decision** ("pilot improves clothing"): all of the following must hold.
  - (2) 4D-DRESS clothing-box LPIPS is lower than v1b's;
  - (3) zoomed clothing LPIPS is lower than v1b's;
  - (1) mean full-body SSIM is not lower than v1b's by more than 0.002;
  - (5) flicker is not higher than v1b's by more than 10 %;
  - (6) per-frame render time is not slower than v1b's by more than 2 %.
  
  Everything is reported whether it passes or not.

### GUAVA-FB amendment 4 (registered 2026-10-29 ~12:35, before any pilot C run): pilot C, normal / depth-normal supervision
User request (2026-10-29): another 8k pilot with normal supervision decoupled to depth, as in VCR-GauS (Chen et al., NeurIPS
2024, "View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction"), on top of pilot B (which passed
amendment 3's rule). Training-only changes: the model, its weights' shapes and inference are unchanged.
- **Pseudo normals:** Sapiens-normal-1B (Meta, CC-BY-NC 4.0, non-commercial research), frozen, bf16, run on each training
  target image (square crop resized to 768, zero-padded to 1024 × 768, no aspect change). Its axes are mapped to GUAVA's
  OpenCV camera frame by (x, −y, −z). This was checked before training by `normal_check.py` on 18 validation targets:
  - mean cosine vs SMPL-X normals 0.91;
  - D-Normal of the SMPL-X surface vs its rendered normals 0.97.
- **Pilot C** = pilot B (same options, seed 0) plus:
  - (1) **flattening** (VCR-GauS Eq. 6): L1 of each Gaussian's smallest world scale, weight 1.0 (paper value);
  - (2) **rendered normal loss** (Eq. 8): L1 + (1 − cos) between the alpha-normalised rendered normal map and Sapiens.
    The Gaussian normal is the shortest scale axis, oriented to the camera. Weight 0.1;
  - (3) **D-Normal loss** (Eqs. 11, 14): normal from the finite differences of the back-projected rendered depth map vs
    Sapiens, times the confidence w = exp((N̂·N − 1)/γ) (detached). Weight 0.15.
  - Losses (2) and (3) apply inside the matte eroded by 2 px ∩ the rendered coverage (alpha > 0.5) eroded by 2 px.
  - Deviations from the paper:
    - depth is the alpha-blended Gaussian **centre** depth, not the ray-plane intersection depth (that needs a custom CUDA
      kernel);
    - the normal and D-Normal weights are the paper's 0.01 / 0.015 × 10, because GUAVA's photometric objective (L1 + LPIPS +
      head / hand / clothing crops, ≈ 0.29) is ≈ 7× the L1 term that corresponds to the paper's RGB loss;
    - γ = 0.1 instead of 0.005: the paper's per-scene setting would give w ≈ 0 everywhere at the start (pilot B's rendered
      normals agree with Sapiens at mean cosine 0.74).
  - Options: pilot B's plus `--flat-weight 1.0 --normal-weight 0.1 --dnormal-weight 0.15 --dn-gamma 0.1`.
  - 8k iterations on the 30k schedule (`--stop-iter 8000`), validation every 2k with the registered validator and best.pt
    rule (unchanged); latest.pt at 8k is kept for the comparison. Smoke test first.
- **Comparison:** amendment 3's measures (1)-(6), for pilot C against pilot B at 8k (and v1b_10k / pilot A for reference),
  plus:
  - (7) **geometry:** mean cosine of the rendered normal and of the D-Normal vs Sapiens on the validation targets of
    measure (2), inside matte ∩ coverage (each eroded by 2 px);
  - normal-map visuals.
- **Decision** ("pilot C improves on B"): all of the following must hold, against pilot B.
  - (2) 4D-DRESS clothing-box LPIPS is lower;
  - (3) zoomed clothing LPIPS is lower;
  - (7) the D-Normal cosine is higher;
  - (1) mean full-body SSIM is not lower by more than 0.002;
  - (5) flicker is not higher by more than 10 %;
  - (6) per-frame render time is not slower by more than 2 %.
  
  Everything is reported whether it passes or not.

### GUAVA-FB amendment 5 (registered 2026-10-29 ~18:50, before any pilot D run): pilot D, real scan geometry + texture losses
User request (2026-10-29): "start pilot D, I need visual and quantitative improvements", restricted (user's earlier choice) to
changes that add **no inference latency**: training losses and evaluation only; the model, its weights' shapes, the Gaussian
count and both inference stages (avatar build, per-frame render) are unchanged. Motivation (pilot C review): normals follow
the body, not the garment; loose garments (vest back) are not covered; folds / patterns / texture are lost.
- **Scan geometry (4D-DRESS, train rows only; val rows for evaluation only):** the dataset's 80k-face textured scans and
  per-vertex garment labels (non-commercial research licence), cached by `dress4d_scans.py` and mapped into each row's
  GUAVA world (X = R_v R_c X_world + R_v t_c - t'; convert_dress4d.py step 4). Check before training (`scan_check.py`,
  16 clips): scan silhouette vs matte IoU 0.976 (zoomed targets 0.981); labels 0 skin / 1 hair / 2 shoes / 3 upper /
  4 lower / 5 outer confirmed by the nearest SMPL-X joint group; scan-to-body distance p50 / p90 / p99: outer 2.4 / 7.0 /
  12.1 cm = 1.32 / 3.45 / 6.02 local face units (GUAVA's offset threshold 3.0 is below the outer p90).
- **Pilot D** = pilot C's flattening and Sapiens losses (weights / gamma unchanged) **restricted to bedlam2, motionx, ubc
  targets** (not ubody, to protect upper body / face; not dress4d, which gets the scan), plus on 4D-DRESS targets (normal,
  same-moment and zoomed), with the scan rendered in the target camera (pytorch3d, interpolated vertex normals):
  - (1) rendered normal vs scan normal, L1 + (1 − cos), weight 0.1; (2) D-Normal vs scan normal, L1 + (1 − cos), no
    confidence weight, weight 0.15; (3) |rendered depth − scan depth| (m), weight 0.5 — all inside (scan silhouette ∩ matte)
    eroded 2 px ∩ coverage eroded 2 px;
  - (4) truncated two-sided chamfer (m, truncation 0.10) between 8,192 random scan vertices and the UV Gaussian centres
    (scan → Gaussians mean; Gaussians → scan weighted by detached opacity), weight 1.0;
  - (5) offset threshold (GUAVA threshold_local_xyz) 3.0 → **6.0** (= outer-garment p99 above), all sources;
  - (6) hinge patch-GAN (spectral-norm PatchGAN, 256-px clothing-box crops via roi_align, Adam 2e-4, betas (0, 0.99); D trains
    from iteration 0, the generator term from iteration 1,000), weight 0.02;
  - (7) VGG Gram texture loss on the same crops (LPIPS' VGG16 layers, L1 of Gram matrices), weight set so that its mean over
    the smoke test's logged iterations is 0.01 (≈ 1/5 of the clothing-box term); the value is recorded in RUNLOG before
    the run.
  - Same seed / schedule / data / mix as pilots B and C; 8k on the 30k schedule (`--stop-iter 8000`), registered validator
    and best.pt rule unchanged; latest.pt at 8k kept. Smoke test first (finite, memory, speed).
- **Comparison** (`compare_fb.py`, pilot D vs pilot B at 8k; pilot C and v1b_10k for reference; validation data only):
  amendment 3's (1)-(6), amendment 4's (7), and new
  - (8) **scan geometry** on the 239 4D-DRESS validation targets: mean cosine of the rendered normal vs the scan normal
    (8a) and **fold detail** (8b): correlation of the high-pass normal components (normal minus its masked Gaussian blur,
    σ = 3 px) of the rendered normal and the scan normal, inside (scan ∩ matte) eroded 2 px ∩ coverage eroded 2 px, eroded
    2 px more; also D-Normal versions, depth MAE, fold detail vs Sapiens on all tracks (reported);
  - (9) **per-garment coverage** on the same targets: fraction of matte pixels of the scan's outer-garment label not covered
    by the Gaussians (alpha < 0.5); upper / lower / whole scan reported;
  - visuals: the showcase figures (largest per-target gains AND the median-gain target for each measure), D vs B (and C).
- **Decision** ("pilot D improves on B"): all of the following must hold, against pilot B.
  - (2) 4D-DRESS clothing-box LPIPS is lower; (3) zoomed clothing LPIPS is lower;
  - (7) the D-Normal cosine vs Sapiens (all targets) is higher;
  - (8a) the scan rendered-normal cosine is higher and (8b) the scan fold detail is higher;
  - (9) the outer-garment miss fraction is lower;
  - (1) mean full-body SSIM is not lower by more than 0.002; (5) flicker is not higher by more than 10 %;
    (6) per-frame render time is not slower by more than 2 %.

  UBody SSIM / face LPIPS (registered validator) are reported. Everything is reported whether it passes or not.

### GUAVA-FB amendment 6 (registered 2026-10-29 ~23:55, before running it): diagnostic — raw vs refined renders, loss-gradient balance
User request (2026-10-29): compare pilot B / C's raw and refined clothing and face renders, and measure weighted gradient norms
and the agreement of geometry and image losses on shared layers ("raw loss values alone do not show which objective
dominates"). **Diagnostic only:** no training, no parameter change, no decision rule; it informs the next pilot's design.
Checkpoints: pilot B and pilot C latest.pt at 8k (pilot D at 8k later with the same tools). No test data.
- **(A) raw vs refined** (`compare_fb.py --part rawref`, validation targets of dress4d / bedlam2 / motionx (as amendment 3)
  and UBody (a20 rule)): "raw" = the first 3 channels of the rasterised 32-channel Gaussian features (GUAVA's loss supervises
  them as an RGB image), "refined" = the StyleUNet refiner output. Per target: PSNR / SSIM / LPIPS on the full image; on the
  clothing box (256 px, amendment 3's box) and the face box (128 px, the validator's box): LPIPS / SSIM / L1 and high-pass
  luminance **detail** (render minus its masked Gaussian blur, σ = 2 px): correlation with the ground truth's (detail r) and
  energy ratio render / ground truth (1 = as much fine detail as the photo, < 1 = smoother). Reported per track, raw and
  refined. Figures: crops at fixed positions (1/6, 1/2, 5/6 of each track's target list), not selected by result.
- **(B) gradient balance** (`guava_fb_train.py --grad-diag`): 15 fixed training batches (3 per training source; both groups
  of a batch from that source; same batches and StyleUNet noise for every checkpoint), pilot B / C's data options. Loss terms
  at their training weights, each differentiated separately: GUAVA's loss split into raw-render and refined-render parts
  (full image L1 + LPIPS; head + hand crops), clothing box, ArcFace, edge, silhouette, offset / scale regularisers, and the
  geometry terms at pilot C / D weights: flattening, Sapiens normal, Sapiens D-Normal (all sources, as C), scan normal /
  D-Normal / depth / chamfer (4D-DRESS, as D), Gram (as D). Per batch and parameter group: Gram matrix of the term gradients,
  raw and Adam-preconditioned (g · lr_group / (√v + ε), v = the checkpoint's second moment; with β1 = 0 this is the step's
  update). Reported: per-term gradient norm; each term's **share of the update** ⟨u_i, u_total⟩ / ‖u_total‖² (shares sum
  to 1; u_total = the pilot's own objective); cosine between the image total and the geometry total (Sapiens, scan), and
  between raw and refined image terms; fraction of batches with cosine < 0. "Shared layers" = parameter groups where both
  image and geometry gradients are non-zero. Wording used in the report (fixed now): a term group "dominates" a layer if its
  update share is > 0.5; geometry and image "conflict" on a layer if their cosine is < 0 in more than half the batches.

### GUAVA-FB status note (user decision 2026-10-30; not an amendment, no criterion changed)
After pilot C's comparison (amendment 4) and the amendment 6 diagnostic:
- **Pilot B at 8k is the main development checkpoint** and the baseline / recipe for the next controlled experiment. The
  refined renders users see favour B over C: clothing-box LPIPS 4D-DRESS 0.381 / 0.381, BEDLAM2 0.313 / 0.318, Motion-X
  0.257 / 0.262, UBody 0.312 / 0.320; UBody face LPIPS 0.192 / 0.194 (B / C). C's raw clothing LPIPS is also worse on all
  four tracks (0.395 / 0.399, 0.326 / 0.330, 0.270 / 0.274, 0.352 / 0.361), so the refiner is not hiding a better C.
- **Pilot C is kept only as the geometry-supervision ablation**: its checkpoint, configuration and results are preserved
  unchanged (SHA-256 in RUNLOG 81); it is not extended beyond 8k and is no longer a leading candidate.
- **a20 remains the fallback** for upper-body and face preservation. Neither B nor C is an accepted final model: B misses
  the registered best.pt thresholds (UBody SSIM 0.8163 < 0.8193, face LPIPS 0.1918 > 0.1873).
- Interpretation of amendment 6 (B): update shares and cosines describe the directions of the Adam steps at one checkpoint.
  Adam's steps are not a fixed budget shared between terms, so a geometry share of 0.13 does not show that geometry took
  progress from the image terms, and the diagnostic does not establish why C regresses on clothing appearance. A causal
  claim needs a controlled intervention.
- Pilot D's registered comparison (amendment 5) is against B and is unaffected.

### GUAVA-FB amendment 7 (registered 2026-10-31 00:03, commit b8d667c1, before the run started at 00:10): recipe B full run (8k -> 30k)
User request (2026-10-31): full training of recipe B (the main development checkpoint, status note above), with loss curves
of every loss term and the total.
- **Run `guava_fb_B30k`:** pilot B's latest.pt (8k; SHA-256 1eae210b...8087) is **copied** (pilot B's directory is not touched)
  into the new run directory and resumed with pilot B's exact options to 30,000 iterations. Pilot B already ran on the 30k
  schedule (lr 5e-5 -> x0.1, `--stop-iter 8000`), the sampler is deterministic per item and the optimiser / scheduler state is
  restored, so the result equals an uninterrupted 30k run of recipe B. Seed 0. GPU 5 (user choice; idle at launch).
- **Validation / checkpoints:** registered validator every 2k (10k ... 30k), best.pt rule unchanged (eligible: UBody SSIM >=
  0.8193 and face LPIPS <= 0.1873; best = highest mean full-body SSIM). Validation data only. latest.pt + best.pt during the
  run; at 30k latest.pt is deleted only if a best.pt exists — if no checkpoint is eligible, latest.pt (30k) is kept and
  reported, since deleting it would leave no model (code change in this commit).
- **Success ("recipe B full run gives an accepted checkpoint"):** an eligible checkpoint exists. Reported whether or not it
  passes: loss curves of every logged term and the total (`plot_losses.py`, pilot B 0-8k + this run), validation curves, and
  at the end the amendment 3 / 5 comparison (`compare_fb.py` eval / geometry / timing, `--part rawref`) of best.pt (else the
  30k model) against pilot B 8k and v1b_10k. a20 remains the fallback either way. No change to pilots C (ablation) or D.

## AvatarTest-v1: own avatar test split (registered 2026-10-31 19:15, before any model is evaluated on it)
User decision (2026-10-31): GUAVA's 58-ID test set behind its Table 2 is not released (phase2/G0_GUAVA_RELEASE.md §3), so we use
our own test split, kept as a JSON (which datasets, what portion, exact clips / frames) and reused by other projects. Every row of
the GUAVA-style table is **run** on it; paper numbers from GUAVA's split are not used as comparisons.
- **Files:** `eval_split/avatar_test_v1.json` (SHA-256 1212a8e8...f14e) + `avatar_test_v1.md` (readable list); builder
  `eval_split/build_avatar_test_split.py` (CPU; reads only split lists and GT annotations, no model output). Never edited:
  any change is a new version (v2).
- **Composition** (one clip per identity; 70 clips, 5,010 test frames):

  | part | data | GT SMPL-X | clips | GT frames | recon / test frames | share of clips / test frames |
  |---|---|---|---|---|---|---|
  | UBody | upper body, real video (30 fps); C1 clips (UBody official intra-scene test list, 135 identities) | UBody release (pseudo-GT, valid_label != 0) | 30 | 7,065 | 3,525 / 3,540 | 42.9 % / 70.7 % |
  | BEDLAM2 | full body, synthetic video; identity-disjoint test of `bedlam_split_proposal.json` (SPRINT val jobs; 2 static, 19 rotating, 19 translating cameras) | exact (render labels), labelled rows 6 fps with the whole body in view | 40 | 2,931 | 1,461 / 1,470 | 57.1 % / 29.3 % |

- **UBody subject / frames (model-free):** annotated boxes linked into person tracks (greedy, IoU >= 0.3, gaps <= 30 frames);
  subject = the track with the most valid GT frames; frames without a valid subject annotation dropped; a clip is kept iff >= 60
  GT frames remain. This removed 6 of C1's 36 clips (ConductMusic_S34_Trim13_scene001 1, Entertainment_S6_Trim2_scene002 0,
  Olympic_S14_Trim4_scene001 35, Online_class_S1_Trim31 32, VideoConference_S14_Trim8 0, VideoConference_S19_Trim14 0 GT
  frames). Frame indices are on UBody's ffmpeg -r 30 timeline (the GT's), not OpenCV's count.
- **Held-out checks (all pass):** UBody identities are in the official test list and in none of a20's / GUAVA-FB's UBody clips;
  SPRINT's UBody rows exclude both official test lists. BEDLAM2 test subjects and their outfits are excluded from GUAVA-FB's
  training pool and validation. Disclosed: 38 of the 40 BEDLAM2 subjects appear in other BEDLAM2 jobs SPRINT trained on, and
  SPRINT selected checkpoints on the val jobs these clips come from; released GUAVA may have trained on UBody clips.
- **Protocol (GUAVA Table 2):** per clip, the ordered GT frame list; source = its first frame (single-image methods see only
  this); recon frames = the first half (the only footage per-subject methods may train on); test frames = the latter half
  (every method is scored on exactly these). Metrics: GUAVA `main/metrics.py` PSNR, L1, SSIM, LPIPS-VGG on the 512 person
  crop (GUAVA-FB perspective crop: 1.1 x the GT-mesh box), GT = image x StyleMatte mask on black; mean per clip, then
  frame-weighted; reported per part (UBody, BEDLAM2) and overall. Recon input and recon time columns as GUAVA's table.
- **Table rows (user):** GUAVA (released), ExAvatar, GART, GaussianAvatar, **GUAVA-FB [SMPL-X GT]** (driven by the GT of each
  test frame), **GUAVA-FB [SPRINT]** (source avatar from SPRINT's fit of the source frame, driven by SPRINT's per-frame output).
  GUAVA-FB = `guava_fb_B30k` best.pt (28k, SHA-256 3ee8e012...0ab), the checkpoint the user calls final; SPRINT = SPRINT from
  scratch EMA 128k (`eval_programme/sprint_scratch/best_ema.pt`). Baselines: official code, trained per clip on the recon frames
  only with the GT SMPL-X (body-model conversion for SMPL-based methods stated), rendered into the same crop. The conversion of
  each row's inputs is registered before that row runs. Nothing is tuned on this split; every number is reported as it comes out.

### AvatarTest-v1 addendum 1 (registered 2026-10-31 20:10, before any conversion, model run or baseline run): rows, inputs, baselines
User instruction (2026-10-31): start the GUAVA-FB rows and the released GUAVA row, and set up ExAvatar, GART and GaussianAvatar
(official code, default settings). Output root E = `O/eval/avatar_test_v1/` (O = the GUAVA-FB output root on /vault; nothing on
/raid; baseline environments / code under /dev/shm, rebuilt by a setup script in the repo). Only final outputs are kept.
- **Test-data conversion (all rows share it):** every GT frame of every clip -> GUAVA tracked format (FORMAT.md):
  - BEDLAM2: `convert_bedlam2.py`'s geometry, crop, StyleMatte and writer, unchanged, on all of the clip's frames (no 40-row cap).
  - UBody: SMPL-X v1.1 (flat_hand_mean False, as OSX) -> SMPL-X 2020: poses copied (the two models share template, joint
    regressor and pose correctives), hands + the v1.1 hand mean (GUAVA's layer is flat-hand), jaw copied, eyes 0, exp50 =
    `expr_maps_to_smplx2020.npz` v11_expr10_to_2020_exp50 @ expr10, betas200 = least-squares map of the v1.1 shape basis onto
    the 2020 basis (all vertices) applied to the clip's MEDIAN betas10 over its GT frames (one shape per clip; the per-frame
    vertex error of using the median is reported), translation pelvis-aligned. Camera = the annotation's per-frame
    (focal, princpt) (OSX virtual camera, unchanged).
  - Crop (precision of the split's rule): the perspective crop is aimed at, and sized by, the projected GT-mesh box
    **clipped to the image** (UBody clips show the upper body; for BEDLAM2 rows the whole body is in view, so nothing
    changes); 1.1 x, 1024 px; scored at 512 (antialiased resize), GT = image x StyleMatte alpha on black, saved as 8-bit PNG
    once (`E/gt512/`). Conversion checks reported: converted-vs-source mesh distance (mm), reprojection (px at 512).
  - FLAME fields for released GUAVA: flame expression = exp50 (SMPL-X 2020's expression basis on the head = FLAME 2020's),
    jaw / eyes copied, eyelids 0, flame_shape = least-squares fit of FLAME's shape basis to the clip's SMPL-X 2020 head
    (rest pose); EHM-vs-SMPL-X-2020 head vertex error reported. head / hand scale 1, joint offsets 0.
- **Rows (one image = the clip's source frame):**
  - GUAVA-FB [SMPL-X GT]: `guava_fb_B30k` best.pt (28k), source = the GT record + crop of the source frame, targets = the GT
    records of the test frames (GUAVA-FB evaluation code, `fb_common`, seed 0 before every inference call).
  - GUAVA (released, best_160000.pt, GUAVA's EHM model with the FLAME head), same source / targets with the FLAME fields above;
    per-record camera via the camera patch (bit-exact to GUAVA when the invtanfov is 24, RUNLOG 64).
  - GUAVA-FB [SPRINT]: SPRINT = from-scratch EMA 128k, deployed setting (no calibration, focal = image width). Source avatar =
    `sprint_avatar_fit.py` on the source frame's ORIGINAL image (own YOLOX box, own crop, own StyleMatte mask, 150-step
    refinement). Driving = per test frame, SPRINT raw (no refinement) on its own detector-box crop with the shape lock to the
    source fit (Avatar A's SL driver). Rendered at 1024 in SPRINT's crop camera and mapped into the metric crop by the exact
    image homography (SPRINT crop -> original image -> metric crop), then resized to 512. No GT is used in this row.
- **Baselines (official code, default settings of the repository's monocular-video configuration):** ExAvatar
  (mks0601/ExAvatar_RELEASE), GART (JiahuiLei/GART), GaussianAvatar (huliangxiao/GaussianAvatar). Each is trained per clip on
  the recon frames only (512 crops, alpha > 0.5 masks, per-frame intrinsics of the crop) and renders the test frames into the
  same 512 crop on black. Driving poses = the GT of each frame: SMPL-X 2020 for ExAvatar converted to its own SMPL-X layer
  (same rig; shape / expression by least squares), its own FLAME face fitting allowed (on recon and test frames, as its
  pipeline does; stated); SMPL for GART and GaussianAvatar = a neutral SMPL fitted per frame to the GT SMPL-X mesh mapped by
  the SMPL-X -> SMPL vertex correspondence (PEAR assets/SMPLX2SMPL smplx2smpl.pkl; one betas10 per clip; residual reported).
  Training-frame pose refinement inside a method's training is its default and is kept; **no test-time pose fitting** in the
  main table (caption). If a method's published protocol fits test poses, that variant is an extra row. Only the minimal data
  loader changes needed to read these inputs (e.g. per-frame intrinsics) are made; method code and hyper-parameters unchanged.
- **Budget rule:** each baseline's wall time per clip is measured on 2 clips (one UBody, one BEDLAM2) and reported before the
  long runs. GART runs all 70 clips. ExAvatar / GaussianAvatar run all 70 iff (mean measured time per clip x 70) / (free GPUs
  at launch) <= 48 h; otherwise they run on **subset24** (`eval_split/avatar_test_v1_subset24.json`, drawn now:
  numpy default_rng(20261031), 12 of the 30 UBody + 12 of the 40 BEDLAM2 clip ids, 1,535 + 463 test frames), and the report
  gives the subset24 table with ALL methods plus the 70-clip table for the methods run on everything.
- **Scoring (one script for every row):** renders as 8-bit PNG at 512 vs `E/gt512`; GUAVA `main/metrics.py` functions (PSNR,
  L1, SSIM 11 px Gaussian, LPIPS-VGG 0.1); mean per clip, frame-weighted over clips; per dataset (UBody, BEDLAM2) and overall.
  Recon input (one image / first half of the clip) and recon time (avatar creation from the source, CUDA-synchronised, GPU
  model stated; + SPRINT fit for the SPRINT row; training time per clip for the baselines). Caption caveats: as registered
  above, plus "UBody clips are upper body only; full-body baselines may struggle there (per-dataset numbers)".
- **Correction (2026-10-31 21:05, before any row was rendered):** the SPRINT row is rendered at **512** (GUAVA's output size; its
  StyleUNet refiner works at 512) in SPRINT's crop camera and warped into the metric 512 crop by the exact homography (bilinear),
  not rendered at 1024 — rendering at 1024 would run the refiner off its operating point. Everything else unchanged.

### AvatarTest-v1 status (2026-11-01 09:30): budget rule applied
Measured per-clip times (2 clips each, NVIDIA L40S): GART 2.4 / 2.9 min (UBody / BEDLAM2); GaussianAvatar 117 / 31 min; ExAvatar
2.7 h / 52.5 min (incl. its own preprocessing and face fitting). Free GPUs at launch: 0-3. GART ran all 70 clips (done).
GaussianAvatar: 87.7 h / 2 GPUs = 43.9 h <= 48 h -> all 70 clips (GPUs 0, 2). ExAvatar: 125 h / 2 GPUs = 62.6 h > 48 h ->
**subset24** (GPUs 1, 3). Reported: the subset24 table with all methods + the 70-clip table without ExAvatar. (A runner bug
idled GaussianAvatar from 02:08 to 09:27; no effect on any number.)

## PublicBench-v1: People-Snapshot + NeuMan (registered 2026-11-01 09:40, before any data conversion or run; user decision)
User decision (2026-11-01): a second GUAVA-FB evaluation on People-Snapshot and NeuMan (ZJU-MoCap dropped for now: only 1 of
its 6 subjects is on disk). Every method is rerun; nothing is tuned on these sets. Each set keeps its PUBLISHED split and pose
source; rendering, crops and metrics are AvatarTest-v1's (so absolute numbers are not directly comparable with the papers'
tables, which use other resolutions / crops / backgrounds; stated in the caption).
- **People-Snapshot** (InstantAvatar protocol, as GART / GaussianAvatar): male-3-casual, male-4-casual, female-3-casual,
  female-4-casual; train / test frames = InstantAvatar's START_END (GART lib_data/instant_avatar_people_snapshot.py:
  e.g. male-3-casual train [0, 455, step 4], test [456, 675, step 4]); poses = InstantAvatar's Anim-NeRF-refined SMPL for
  train and test frames (the protocol's pose source; if not obtainable, the dataset's reconstructed_poses.hdf5, stated);
  camera = the dataset's calibration. Source (single-image methods) = the first training frame.
- **NeuMan** (official split, as NeuMan / HUGS get_data_splits: every k-th frame held out, test = first half of the held-out
  frames, train = the rest): bike, citron, jogging, seattle (the earlier GUAVA-FB plan's 4 sequences); poses = NeuMan's
  smpl_output_optimized (+ its alignment to the COLMAP cameras), camera = COLMAP per frame. Source = the first training frame.
- **Inputs (all methods):** AvatarTest-v1's conversion and contract (perspective person crop from the projected GT mesh clipped
  to the image, 1.1 x, 1024 -> 512; StyleMatte mask; GT = image x mask on black). GT is SMPL here: GART / GaussianAvatar get it
  directly; GUAVA-FB, released GUAVA and ExAvatar get SMPL-X 2020 fitted to the GT SMPL mesh (pose + one shape per sequence,
  through the SMPL-X -> SMPL vertex correspondence; hands / face neutral; residual reported). GUAVA-FB [SPRINT] uses SPRINT only.
- **Rows / scoring / table:** as AvatarTest-v1 (same six rows, same scorer and layout, per dataset + overall). Baselines: same
  adapters, train on the published train frames only, render the test frames; no test-time pose fitting in the main table;
  GART's published People-Snapshot protocol fits test poses -> extra row "GART + test-pose fitting" (ExAvatar's NeuMan protocol
  likewise -> extra row if run). Budget: 8 sequences; GART / GUAVA rows on all; ExAvatar / GaussianAvatar on all 8 iff the
  projected time on the GPUs free at launch is <= 48 h, otherwise the 4 People-Snapshot subjects first.

## GUAVA-FB improvement phase 1: diagnostics (registered 2026-11-01 10:30, before any diagnostic is computed; user brief)
Goal (user): close the gap to GART while keeping single-image creation and inference speed; improve the practical SPRINT pipeline
separately. GART trains on the first half of each clip, GUAVA-FB sees one image: stated with every comparison. AvatarTest-v1 stays
fixed: its results (RUNLOG 85) were already inspected and are descriptive only; nothing is tuned or selected on it, and the final
test evaluation is reserved for the one configuration selected on validation data.
- **ValBench (development replica of the test protocol, validation data only):** UBody = a20's 10 validation clips (UBody
  SMPL-X annotations as GT, the AvatarTest-v1 subject / >= 60 GT-frame rules), BEDLAM2 = GUAVA-FB's 10 validation clips (all
  full-body rows). Same conversion, crops, StyleMatte masks, GT x mask on black, scorer and first-frame / first-half / latter-half
  protocol as AvatarTest-v1 (`eval_split/` scripts with a different split file and output root `O/eval/valbench_v1/`). Rows run on
  it: GUAVA-FB [GT] (B30k best.pt), its raw Gaussian render (first 3 feature channels, no refiner), GUAVA-FB [SPRINT], GART.
- **A (quality under GT inputs), per target pixel inside GT-mask OR render coverage:** errors (L1, squared error -> PSNR,
  SSIM map) split by (i) body region of the GT SMPL-X surface hit by the pixel (head, hands, torso, arms, legs, feet) and
  "off-body" (mask pixels no body surface covers: hair, loose clothing, held objects); (ii) source visibility of that surface
  (vertex visible in the source frame's rasterisation, depth-tested); (iii) source-to-target view angle (angle between body
  forward axes, bins 0-30 / 30-60 / 60-120 / >120 deg); (iv) motion (root-aligned body-joint distance source -> target, terciles);
  (v) coverage (miss = mask > 0.5 and render < 0.02 in all channels; spill = mask < 0.5 and render > 0.02). The same breakdown
  for GART and for GUAVA-FB's raw render. Share of total squared error per bin is reported, so the dominant source is visible.
- **B (SPRINT degradation), validation targets of ValBench, every GT-assisted variant labelled "diagnostic":** SPRINT outputs
  are mapped into the metric crop camera (orientation through the rotation between SPRINT's crop-centre camera and the metric
  camera; translation keeps SPRINT's ray and image size: depth scaled by true focal / SPRINT's assumed focal). Variants:
  2 x 2 source fit (GT record vs SPRINT fit) x driving (GT vs SPRINT-mapped); the actual row (SPRINT camera + warp) vs SPRINT-mapped
  (the camera / intrinsics effect); from S_gt x D_gt, one component swapped to SPRINT's at a time: root orientation, root
  translation, body pose (21 joints), hands, face (jaw + expression); source side: SPRINT shape only, SPRINT source pose + camera
  only. Same scorer; per dataset and per clip.
- Outputs `O/eval/valbench_v1/` and `O/reports/improve/`; no training in this phase.
- **ValBench UBody extension (2026-11-01 10:55, before any model is run on ValBench):** only 5 of a20's 10 UBody val clips have
  >= 60 GT frames. Added: UBody TRAINING-split videos never used by GUAVA-FB / a20 (not in the 269 tracked c2t clips), identity
  (OSX rule) disjoint from all c2t identities and from both official test lists; 150-450 frames, 720 px high; drawn round-robin
  over scenes with numpy default_rng(20261101) until 15 more clips pass the same >= 60 GT-frame rule. SPRINT trained on all
  non-test UBody videos (incl. a20's val clips), so the SPRINT confound is the same for every ValBench UBody clip (stated).
- **ValBench BEDLAM2 extension (2026-11-01 11:05, before any model is run on ValBench):** 10 val clips = 356 test frames only.
  Added the next 20 subjects of the SAME rule GUAVA-FB's validation used (convert_bedlam2.plan: val-job clips, subjects in neither
  test nor the training pool, MOYO excluded, ordered by SHA-1(subject), one clip per subject, moving-camera job first, most
  full-body rows): subjects 11-30 of that order -> 30 BEDLAM2 clips. Never in any training; SPRINT selected on these jobs (stated).

## GUAVA-FB improvement phase 2: hypotheses and pilots (registered 2026-11-01 15:40, after the phase-1 diagnostics, before any intervention run)
Diagnostics (validation only; `guava_fb/IMPROVE_NOTES.md`): ValBench reproduces the test pattern (GART vs GUAVA-FB [GT]: UBody 20.06 vs
19.06 dB, BEDLAM2 26.71 vs 26.13; GUAVA-FB [SPRINT] 16.68 / 21.92). A: largest error class = off-body pixels (garments, hair, held
objects; 24 % / 33 % of GUAVA-FB's squared error, GART +2.0 / +2.3 dB there); self-reconstruction of the source is high (BEDLAM2
35.6 dB, UBody 24.3) and quality drops with pose / view change -> generalisation (non-skinned garments, unseen surfaces), not texture
capacity, limits GT-input quality. B: 2 x 2 (source x driving, UBody / BEDLAM2 PSNR): GT/GT 19.06 / 26.13, GT/SPRINT 16.87 / 20.95,
SPRINT/GT 17.05 / 22.58, SPRINT/SPRINT 16.91 / 22.09 (own camera); SPRINT's camera assumption costs nothing (mapped into the true
camera: 16.81 / 21.54); single 3D components are not additive (body pose alone -3.4 / -4.5 dB, all driving -2.2 / -5.2).
- **H1 (SPRINT pipeline; inference only; strongest evidence):** per-frame 2D misalignment of SPRINT's raw driving estimate.
  Intervention: each driving frame refined by k Adam iterations of sprint_avatar_fit's own objective (DWPose keypoints + StyleMatte
  mask + priors to SPRINT's prediction; shape locked to the source fit) instead of k = 0. Source fit unchanged (150 iterations).
  Arms on ValBench: control k = 0 (existing renders), k = 20, k = 60; same model (B28k), frames, warp and scorer. Measured: per
  dataset PSNR / SSIM / LPIPS / L1, per-clip deltas, SPRINT time per frame (GPU-synchronised, median of 200 frames, DWPose CPU part
  separate). Decision: adopt the smallest k whose all-frame PSNR gain is >= 80 % of the best arm's gain, iff that arm gains
  >= 0.3 dB all-frame PSNR and >= 0.005 LPIPS and neither dataset regresses (PSNR -0.1 dB or LPIPS +0.003); otherwise H1 rejected.
  Budget ~1.5 GPU-h; no training.
- **H2 (avatar; training only):** loose garments / hair need Gaussians farther from the SMPL-X surface than GUAVA's offset hinge
  allows (`threshold_local_xyz` 3.0; on dress clips 4-6 % of UV Gaussians already exceed it). Single intervention: threshold 3.0 ->
  6.0. Control and intervention identical otherwise: init = B30k best.pt weights (fresh Adam), lr 8e-6 constant (B30k's schedule
  value at 28k), 4,000 iterations, seed 0, recipe B options, registered validator at 2k / 4k with a20's reference values (B30k
  iteration-0 log). Measured at 4k: ValBench GUAVA-FB [GT] per dataset + per clip, diag_regions off-body PSNR / SSE share,
  miss / spill, UBody eligibility, creation / render time and memory. Decision: H2 supported iff vs control: BEDLAM2 off-body
  PSNR >= +0.3 dB AND neither dataset's image PSNR lower by > 0.05 dB AND LPIPS not higher by > 0.002 AND UBody eligibility holds
  AND spill not up by > 0.5 pt. Stopping rule: smoke 200 it (finite losses, renders aligned); abort an arm on non-finite loss or if
  UBody val SSIM at 2k falls below the eligibility threshold by > 0.005. Budget ~2 x 1.5 GPU-h (GPU 5).
- **H3 (source fit; not run now):** SPRINT's source fit alone costs 2.0 / 3.6 dB under GT driving (shape 0.7 / 2.1); more
  source-refinement iterations or a multi-frame-free shape prior are one-time costs, but the 2 x 2 non-additivity makes the deployed
  gain uncertain; ranked third.
- AvatarTest-v1: untouched by these choices; only an adopted configuration gets one final test evaluation, reported with the
  disclosure that earlier test results were inspected.
- **H1 result (2026-11-01 14:55, applying the rule):** ValBench all-frame PSNR / LPIPS: k = 0 18.10 / 0.2009, k = 20 19.39 / 0.1814,
  k = 60 20.37 / 0.1685 (UBody 18.93, BEDLAM2 24.21; GT-driven 19.06 / 26.13). k = 20 reaches 57 % of k = 60's gain (< 80 %) ->
  **k = 60 adopted**; all conditions met (+2.27 dB, -0.032 LPIPS, no dataset regresses; per clip: PSNR up on 20 / 20 UBody and
  30 / 30 BEDLAM2 clips, LPIPS up on 19 / 20 and 30 / 30). Cost (idle GPU 5, median per frame): 266 -> 365 ms (+99 ms, refinement
  116 ms; DWPose CPU 168-177 ms and StyleMatte 63 ms dominate either way).
- **H1b (registered 2026-11-01 15:05, before running; pipeline correctness):** the deployed fitter takes the LARGEST detected person
  in every frame; on 2 / 20 UBody ValBench clips this is another person in the source frame and in 62 % / 92 % of the test frames
  (detector-only check, `reports/subject_check_valbench.json`). Change: the subject is selected once in the source frame by its box
  (identity selection, as a user click would: the split's subject box for UBody; largest box for single-person BEDLAM2), then
  tracked: each frame takes the detection with the highest IoU to the previous frame's box (kept if no detection has IoU >= 0.1).
  No pose / shape / camera GT is used. Arm: k = 60 + tracking vs the adopted k = 60. Adopt iff all-frame PSNR does not drop and
  the two wrong-person clips improve; reported per clip (other clips must be unchanged up to numerical noise).
- **H1b result (2026-11-01 15:50):** all-frame PSNR 20.37 -> 20.50, the two wrong-person clips +4.13 / +3.24 dB, BUT 3 single-person
  BEDLAM2 clips regressed (-3.24 / -2.36 / -0.62 dB): the stale-box fallback locked onto the source frame's box when the first test
  frame (half a clip later) had no overlapping detection (seq_000515: 17 / 17 test frames). The registered "other clips unchanged"
  condition fails -> **H1b rejected as implemented.**
- **H1c (registered 2026-11-01 15:55, before running):** H1b with one change: if no detection has IoU >= 0.1 with the previous box,
  take the detection whose centre is nearest to the previous box's centre (instead of keeping the stale box). With one detection
  this equals the deployed largest-box choice. Arm: k = 60 + H1c tracking vs k = 60. Adopt iff all-frame PSNR does not drop, the
  two wrong-person clips improve, and no other clip changes by more than 0.1 dB PSNR or 0.003 LPIPS.
- **H1c result (2026-11-01 17:10, applying the rule):** ValBench all-frame PSNR / LPIPS 20.37 / 0.1685 -> 20.55 / 0.1645 (UBody 18.93 ->
  19.19, BEDLAM2 24.208 -> 24.208); the two wrong-person clips +4.13 dB / -0.056 LPIPS (Olympic_S6) and +3.24 dB / -0.078
  (ConductMusic_S29); every other clip within 1e-5 dB (47 bit-identical; the three BEDLAM2 clips H1b broke are identical to k = 60;
  the nearest-centre rule fired on one frame of each and picked the same box). All conditions met -> **H1c adopted** (k = 60 +
  H1c tracking is the selected SPRINT pipeline). No extra GPU work (IoU / centre distance on the detector's existing boxes).
- **Record correction (2026-11-01 17:15):** (i) the phase-2 header's "registered 15:40" is wrong: the section was committed in
  6a3b56f1 at 13:54 CST, before the first intervention run (H1 launched 13:57; H2 control 14:46). (ii) H1b and H1c were written
  into this file before their runs (session log: H1b 14:59:38, run 14:59:48; H1c 15:51:31, run 15:51:36; their "15:05" / "15:55"
  labels are late) but NOT committed before running, a deviation from the commit-before-run rule; they are committed with their
  results. Neither rule was changed after a result was seen.
- **H2 result (2026-11-01 17:11, applying the rule; 4k latest.pt of each arm):** ValBench GUAVA-FB [GT], intervention (hinge 6.0) vs
  control (3.0): BEDLAM2 off-body PSNR 14.06 vs 14.10 (-0.04 dB; required >= +0.3) -> FAILS; image PSNR UBody 19.0030 vs 19.0031,
  BEDLAM2 26.115 vs 26.120 (within 0.05); LPIPS within 0.0001; spill BEDLAM2 7.14 vs 7.18 %, UBody 7.79 vs 7.79 %; registered
  validator at 4k: UBody SSIM 0.819254 (control 0.819324; threshold 0.819331) -> neither arm eligible at 4k (both eligible at 2k).
  Per clip (off6 - control): PSNR -0.13 .. +0.05 dB (largest drop on the dress clip vcam_050), LPIPS within 0.0006. **H2 not
  supported.** Manipulation check (5 diagnosis clips, source pose): Gaussians beyond 3.0 on the dress clips 3.8 -> 4.1 % and
  5.8 -> 6.2 % (B28k 4.1 / 6.1 %); offset hinge term 1.6e-4 -> 2e-5 of a total loss ~0.2. In this bounded continuation (4k it,
  lr 8e-6) relaxing the hinge barely changed the offsets; whether the hinge limits garments when training from scratch is untested.
  For scale: the control's own 4k continuation moved per-clip PSNR by -0.27 .. +0.36 dB vs B28k (larger than the arm difference).
- **Record correction 2 (2026-11-01 17:18):** in commit 42b7f692 the H1c result's "17:10" was written at 17:08 and the first record
  correction's "17:15" at 17:09 (session log); text otherwise unchanged. Committed entries are corrected by notes, not edited.

## GUAVA-FB improvement phase 3: final test evaluation (registered 2026-11-01 17:17, before any run of the selection on AvatarTest-v1)
- Selected on validation data only (phase 2): B28k unchanged + SPRINT driving refined per frame with k = 60 + H1c subject tracking.
  H2 failed; no model change. Nothing in this phase was tuned on AvatarTest-v1.
- One run on AvatarTest-v1 (all 70 clips; same conversion, crops, StyleMatte masks, 512 px, black background, scorer, LPIPS VGG,
  frame-weighted aggregation as RUNLOG 85). Reported as an EXTRA row "GUAVA-FB [SPRINT, refined + tracked]" next to the
  registered GUAVA-FB [SPRINT] row, which stays as is (k = 0, largest box); per dataset, overall and per clip vs [SPRINT],
  regressions included. No decision depends on the result.
- Caption disclosures: (1) AvatarTest-v1 results (RUNLOG 85) were inspected before this phase; this is the only test run of the
  selection. (2) The refinement fits each target frame's own DWPose keypoints and StyleMatte matte (inference-time signals, no GT);
  the metric's mask is the same StyleMatte matte. (3) UBody source subject = the split's subject box (as a user's click); BEDLAM2
  = largest box. (4) GART trains on the first half of each clip; GUAVA-FB uses one image.
- Command: `eval_guava_rows.py --row fb_sprint --drive-iters 60 --track-subject --track-nearest --name fb_sprint_k60_trkc
  --data-root <local copy of E/data>` (E = O/eval/avatar_test_v1), then `score.py --methods fb_sprint_k60_trkc --tag full`.
- Runs only after the user's go-ahead (GPU job > 15 min; ~1.5-2 GPU-h at the ValBench rate).

## GUAVA-FB improvement phase 4: options that keep live inference cost (registered 2026-11-02 00:25, before any run of this phase)
- Source: a reviewer note the user pasted on 2026-11-02 ("can you try these?"): (i) test H1c tracking without the k = 60 loop;
  (ii) train SPRINT to predict H1's refined fits (H1 run offline on training images, unreliable fits filtered; live inference
  stays one forward pass); (iii) improve avatar training without changing the architecture or Gaussian count. Validation only
  (ValBench-v1); AvatarTest-v1 is not touched in this phase; phase 3 stays pending.
- Timing fact behind (ii) (RUNLOG 90, idle GPU 5, median of 178 frames): one SPRINT driving frame at k = 0 is 239-266 ms =
  DWPose on the CPU 130-154 ms (YOLOX-L detector + whole-body keypoints) + StyleMatte 63 ms + SPRINT forward 7 ms + ~6 ms
  overhead; k = 60 adds 109-116 ms of GPU refinement. The 12.2 ms figure is the avatar render only. The deployed k = 0 loop
  computes the keypoints and the matte but uses only the detector box; a lean k = 0 loop (detector + crop + SPRINT) is not timed yet.
- **H3 (k = 0 + H1c tracking; inference only, no added cost):** deployed SPRINT row (k = 0) with H1c subject tracking
  (`--track-subject --track-nearest`) vs `fb_sprint` (k = 0, largest box). Subject selection uses the detector boxes only and does
  not depend on k: the k = 60 rows with and without H1c have bit-identical renders on 47 / 50 clips (two separate runs, md5 of every
  frame), so on those clips the arm's renders are `fb_sprint`'s, copied. The 3 clips that differ (ubody/ConductMusic_S29_Trim13,
  ubody/Olympic_S6_Trim23_scene001, bedlam2/20240731_1_1827_hdri__seq_000532) are rerun at k = 0 with H1c (`eval_guava_rows.py
  --row fb_sprint --track-subject --track-nearest --name fb_sprint_k0_trkc --clips <3 ids>`). Rule (as H1c): adopt for the k = 0
  pipeline iff all-frame PSNR does not drop, the two wrong-person UBody clips improve, and the BEDLAM2 clip changes by <= 0.1 dB
  PSNR and <= 0.003 LPIPS. Also reported: the share of the selected pipeline's gain (k = 0 -> k = 60 + H1c, +2.45 dB) that
  tracking alone gives.
- (ii) and (iii) are registered below before any of their runs (smoke tests included).
- **H3 result (2026-11-02 00:29, applying the rule):** ValBench all-frame PSNR / SSIM / LPIPS 18.10 / 0.769 / 0.2009 -> 18.22 /
  0.775 / 0.1988 (UBody 16.68 -> 16.83 dB; BEDLAM2 21.92, unchanged to all printed digits); wrong-person clips Olympic_S6 +3.46 dB /
  -0.038 LPIPS, ConductMusic_S29 +1.82 dB / -0.039; bedlam2 seq_000532 unchanged (0.0000 dB). All conditions met -> **H3 adopted for
  the k = 0 pipeline** (zero added cost). Tracking alone gives 0.11 of the selected pipeline's +2.45 dB (4.7 %); the rest is the
  k = 60 refinement. With tracking the two clips reach 14.92 / 13.98 dB at k = 0 vs 15.69 / 15.41 at k = 60.
- **Finding that changes how SPRINT rows on ValBench UBody read (2026-11-02 00:29, checked before any H4 run):** all 20 ValBench
  UBody clips are in SPRINT's UBody training rows (1,932 rows of these clips, every 3rd annotated frame; 25,701 rows of their 20
  identities; SPRINT's `ubody_labels` train split), so every SPRINT row on ValBench UBody (k = 0, H1, H1c, H3) is measured on
  clips SPRINT was trained on with UBody's pseudo-GT. ValBench BEDLAM2 (30 clips) and all of AvatarTest-v1 (30 UBody + 40 BEDLAM2)
  are unseen by SPRINT (BEDLAM2 val jobs; UBody official test lists). The H1 / H1c / H3 decisions compare arms that share the same
  SPRINT, so they stand; the size of the H1 gain on unseen real clips is not measured by ValBench UBody.
- **Record correction 3 (2026-11-02 00:29):** the phase-4 header's "registered 00:25" is late: the section was written at ~00:20 and
  committed in 1b0b94e3 at 00:21:19, before the H3 run started (00:21:30). The two entries above were first labelled 00:47 / 00:45
  and fixed to 00:29 (the clock time when they were written) before committing.

### Phase 4, H4: SPRINT trained on its own offline H1 fits (registered 2026-11-02 00:34, before any H4 run, smoke tests included)
- **Hypothesis:** fine-tuning SPRINT on H1 fits of its own UBody training images (the k = 60 driving refinement's objective:
  DWPose keypoints + StyleMatte matte + priors; unreliable fits filtered) recovers part of H1's gain with ONE forward pass
  (k = 0; live cost unchanged), relative to a matched control fine-tune with the original labels. Expected only on real
  (UBody-like) clips: BEDLAM2 rows already carry exact GT, which no H1 fit can improve on.
- **Evaluation set ValBench-S (UBody, built after this registration by , validation only):**
  clips of SPRINT's  VAL split (2 % of clips held out when SPRINT's UBody rows were built) whose identity has no
  SPRINT training row, never tracked for GUAVA-FB / a20 (c2t), identity-disjoint from ValBench-v1 and AvatarTest-v1; video rules
  of ValBench-v1's UBody extension (720p, 150-450 frames at 30 fps) and AvatarTest-v1's subject-track / >= 60 GT-frame rules;
  converted with  (same protocol: first frame source, latter half test, StyleMatte mask, 512 px, black
  background). 20 candidates pass the identity rules (counted 00:28; Movie 9, Speech 5, SignLanguage 5, ConductMusic 1).
- **Labels ():** rows of SPRINT's UBody train split, identities of ValBench-v1 /
  ValBench-S / AvatarTest-v1 excluded; clips in seeded random order (seed 20261102), up to 20 evenly spaced rows per clip, until
  24,000 rows. Per row: the stored 512 px context crop, the row's UBody box mapped into it,  with
  the deployed SPRINT (scratch EMA 128k) and H1's objective / weights unchanged; SPRINT's camera input computed in the original
  frame (as deployed); the 20 shape offsets free (a training row has no source frame). Kept per row: refined root / body / hand
  rotations, shape, camera, DWPose keypoints + scores, fit diagnostics.
- **Filters (fixed now):** drop a row if (F1) < 6 of the 14 body keypoints (OpenPose 0-13) have score >= 0.3 inside the crop;
  (F2) the refinement raised the keypoint error; (F3) keypoint error after > the 90th percentile of the set; (F4) mask-outside or
  mask-coverage term after > its 95th percentile. Per-joint validity: a hand's rotations and 2D joints only if >= 15 of its 21
  keypoints have score >= 0.3; lower body (hips, knees, ankles, feet) only if both knees and ankles do; jaw / eye 2D joints only
  if nose and both eyes do.
- **Label use ( options , ; default unchanged):** on the labelled rows the
  arm replaces UBody's pseudo-GT root / body / hand rotations by the H1 rotations and adds 55-joint 2D targets (the trainer's
  kinematic SMPL-X joints of the H1 fit projected with its camera into the context crop; has_cam = 1, so the trainer's camera
  and joint-2D terms apply as for BEDLAM2 rows). Unchanged in both arms: images, augmentation, COCO-17 2D keypoints, UBody face
  labels (jaw / expression), no shape term on these rows (has_shape = 0).
- **Fine-tune (both arms identical except the labels):** from SPRINT scratch EMA 128k, 4,000 steps, SPRINT's final batch mix
  and losses (configs/sprint/sprint_final.yaml), the UBody slice (9 rows of 64) restricted to the same labelled rows in both
  arms (), lr 3e-5 (backbone x 0.1), warmup 200, cosine to 0.05 x, EMA decay 0.999, seed 20261024, validation (3DPW-val
  in-loop + the four hold-outs) at step 4,000 only; the step-4,000 EMA weights are evaluated (latest.pt kept, best.pt by the
  trainer's rule is the same step).
- **Endpoints:** E1 (primary) ValBench-S, GUAVA-FB [SPRINT] row (B28k) at k = 0 + H1c tracking with the arm's vs the control's
  SPRINT; references: original SPRINT at k = 0 + H1c and the teacher pipeline (original SPRINT, k = 60 + H1c). E2 ValBench BEDLAM2
  (30 clips, unseen by SPRINT), arm vs control at k = 0 + H1c. E3 ValBench UBody (SPRINT-seen): reported only. Guard: SPRINT's
  own validation at 4,000 (EMA): 3DPW-val PA-MPJPE and BEDLAM2-val PA-MPJPE, arm <= control + 1.0 mm.
- **Rule (all must hold):** (a) E1 all-frame PSNR arm - control >= +0.30 dB and LPIPS arm <= control; (b) E1 PSNR up on
  >= 2/3 of clips; (c) E2 PSNR arm - control >= -0.10 dB; (d) the guard. Met -> H4 supported (distilled SPRINT becomes the k = 0
  candidate); reported with the share of the teacher's E1 gain it recovers. Not met -> not supported, reported with per-clip
  results and label statistics.
- **Smoke test (< 15 GPU-min) before the pilot:** 200 rows labelled (throughput; projected kinematic wrists / elbows vs the
  fit's own DWPose-layout points; DWPose hand side vs SMPL-X hand side), 30 training steps per arm (finite losses, H1 rows carry
  has_cam = 1 and in-crop 2D targets; augmented samples dumped), CPU time of the YOLOX-L detector alone per frame (the lean
  k = 0 loop's extra cost besides SPRINT's 7 ms). The pilot (labelling + 2 fine-tunes + evaluation, > 15 GPU-min) runs only
  after the user's go-ahead with the measured budget.
- **(iii) avatar training without architecture / Gaussian-count change:** not run in this phase (assessment in IMPROVE_NOTES).
- **Record correction 4 + H4 amendment 1 (2026-11-02 00:36, before ValBench-S is built and before any H4 run):**
  (i) In commit 74b9b8f7 an unquoted shell heredoc executed nine back-quoted code names in the H4 entry above as commands (all
  "command not found" / "permission denied", nothing ran) and dropped them from the text. They read, in order:
  "built after this registration by `eval_split/build_valbench_s.py`"; "clips of SPRINT's `ubody_labels` VAL split"; "converted
  with `convert_test.py`"; "**Labels (`phase3_dnt/sprint_distill/h1_label.py`):**"; "mapped into it, `SprintFitter.fit(iters =
  60)` with"; "**Label use (`dataset/ubody_label_dataset.py` options `rows`, `h1_labels`; default unchanged):**"; "arms
  (`rows`)". No rule, number or criterion is affected.
  (ii) The count "20 candidates pass the identity rules (... Movie 9, Speech 5, SignLanguage 5, ConductMusic 1)" used a wrong
  identity function (only `_Trim` stripped). With the project's rule (`build_avatar_test_split.ub_identity`, OSX: split at
  `_Trim` / `_scene`), 5 SPRINT-val clips have identities without SPRINT training rows and only 2 pass the video rule: unusable.
  (The H3 entry's "25,701 rows of their 20 identities" is re-checked with the project's rule: 25,701, unchanged.)
  (iii) **Amendment: ValBench-S is clip-level held out.** Pool = clips of SPRINT's `ubody_labels` VAL split (no frame of them was
  used to train SPRINT), not c2t clips (no frame used to train GUAVA-FB / a20), identity (project rule) not in ValBench-v1 or
  AvatarTest-v1; other clips of the same identity MAY be in SPRINT's or GUAVA-FB's training (disclosed per clip). Video rule
  relaxed to: readable, 720p or 1080p, <= 600 frames at 30 fps; AvatarTest-v1's subject-track / >= 60 GT-frame rules unchanged.
  Selection: seeded order (seed 20261102), at most 2 clips per identity, up to 20 clips. Counted before writing this: 55 pool clips,
  41 with >= 20 SPRINT val rows (~>= 60 annotated frames), 25 identities, 9 scenes. The H4 label set excludes ValBench-S's
  identities as well. Everything else in H4 is unchanged.
- **H4 amendment 2 (2026-11-02 00:44, after the label smoke test, before the label run):** smoke test (200 rows, context-crop
  input): 3.85 rows/s on one process (0.25 s/row); on 9 / 200 rows the fit is non-finite: in the stored 512 px context crop the
  person is ~200-250 px tall, the 1024 deployment crop is a ~4x upsample, StyleMatte returns an empty matte, and an empty matte
  turns the mask term into NaN (its distance transform has no zero), so the refinement diverges. Finite rows: keypoint error
  24.6 -> 8.8 px (1024 crop, median). Changes: (1) labels are fitted on the ORIGINAL video frame (decoded as the crop builder
  does: ffmpeg -r 30, image2 muxer) with the row's UBody box, exactly as deployed (no `cam_outer`); per row, the frame warped into
  the context crop must match the stored crop (mean absolute difference <= 4 / 255), else the row is dropped (frame alignment);
  (2) F0: rows with a non-finite fit are dropped, and the F3 / F4 percentiles are taken over finite rows. Rows are sharded by clip
  (one decode per clip). The latent empty-matte NaN in sprint_avatar_fit's refinement also affects the deployed pipeline; it is
  reported, not changed here; whether any ValBench / AvatarTest frame hit it is not checked yet.
- **Precision note (2026-11-02 11:42):** "unseen by SPRINT" in the H3-finding entry is at the clip level (BEDLAM2 val jobs; UBody official
  test lists). BEDLAM2 subjects recur across jobs: 38 / 40 AvatarTest-v1 BEDLAM2 identities appear in sequences SPRINT trained on (as the
  AvatarTest-v1 table caption states). AvatarTest-v1 UBody identities have no SPRINT training row (checked with `ub_identity`).

## Phase 5: person-box source for the live SPRINT path (registered 2026-11-02 13:54, before any run; user request)
- **Question:** which box source to use for live driving, with no retraining: (a) YOLOX-L per frame (current reference); (b) MediaPipe
  Pose per frame; (c) a box tracked from SPRINT's own previous-frame keypoints, with re-detection on loss. Choose by quality
  first, then speed. Validation only: ValBench-v1 for the one tuned value (margin), ValBench-S (13 UBody clips SPRINT never trained
  on) for the comparison. AvatarTest-v1 not used.
- **Common path (all arms):** GUAVA-FB B28k, original SPRINT (scratch EMA 128k), k = 0 (raw driving, the lean backend
  `sprint_fast_driving.drive_raw`, raw outputs identical to the full fitter per NO_LATENCY_PREREG), the same source fit and H1c
  source subject (UBody: split box; else largest YOLOX box), same renders and scorer. Only the driving-frame box differs. Bounded threads (OMP 4, ONNX 4).
- **(a) YOLOX:** detector every frame + H1c linking (IoU with the previous box, nearest-centre fallback). Quality must equal the existing
  `fb_sprint_k0_trkc` ValBench-S row (checked); this run is for timing and per-frame boxes.
- **(b) MediaPipe Pose** (mediapipe 0.10.21, legacy Pose, full landmark model, static mode) run on a square region around the previous
  frame's box (2.0 x its long side, clipped to the frame), so it cannot jump to a person outside that region; box = axis-aligned box of
  landmarks with visibility >= 0.5, scaled by margin m about its centre, then SPRINT's standard crop rule (square, 1.1 x long side).
  Lost = no landmarks, < 6 visible landmarks, or IoU with the previous box < 0.1 -> YOLOX + H1c linking for that frame.
- **(c) Track:** the box for frame t+1 = axis-aligned box of SPRINT's projected body keypoints at frame t (OpenPose body 0-17 of the
  134-point layout, through the frame's crop camera and back to frame pixels), scaled by margin m, same crop rule. Lost (re-detect with
  YOLOX + H1c for that frame) if < 50 % of that box lies inside the image, or its long side changes by > 1.5x from the previous box.
- **Margin tuning (b, c separately), on ValBench-v1 only:** m in {1.1, 1.25, 1.4}, on a seeded subset of 10 UBody + 10 BEDLAM2 clips
  (seed 20261102); pick the m with the highest all-frame PSNR (ties within 0.05 dB -> lower LPIPS). The chosen m is then fixed.
- **Report on ValBench-S, per arm:** PSNR / SSIM / LPIPS (all frames and per clip); wrong-person frames = test frames whose chosen box
  has IoU < 0.3 with the GT subject box (UBody annotation) ; lost / re-detected frames; end-to-end ms / frame p50 / p95 of box ->
  SPRINT -> GUAVA-FB render (synchronised), on the server with host load recorded, and on the user's laptop (script provided; run by
  the user).
- **Choice rule:** quality first: an arm is acceptable iff all-frame PSNR >= (a) - 0.10 dB, LPIPS <= (a) + 0.003, and wrong-person
  frames <= (a). Among acceptable arms the fastest end-to-end p50 on the laptop wins (server p50 if laptop numbers are not available,
  disclosed). If none is acceptable, (a) stays.
- **Budget:** smoke test < 15 GPU-min; the tuning (6 runs x ~1,400 frames) and ValBench-S runs (3 x 1,325 frames) run only after the
  user's go-ahead with the measured cost.
- **Phase 5 amendment 1 (13:54, before any run, from reading the design against UBody's framing):** in (c), projected hips / knees /
  ankles of upper-body clips fall below the image, so a box over all 18 points would trigger re-detection almost every frame. (c) uses
  the projected OpenPose body points 0-17 that fall INSIDE the image; lost = fewer than 4 such points, or the box's long side changes by
  > 1.5x from the previous box. Nothing else changes.
- **H4 result (2026-11-02 15:38, applying the rule; `O/reports/improve/h4_report.json`):** ValBench-S all frames, arm vs control at k = 0 + H1c:
  17.79 / 0.774 / 0.1992 vs 17.75 / 0.771 / 0.2011 (PSNR +0.035 dB, LPIPS -0.0019); 10 / 13 clips up; original SPRINT 17.85, teacher
  (k = 60 + H1c) 20.99 -> the arm recovers 1.1 % of the teacher's +3.15 dB. ValBench BEDLAM2 +0.026 dB (22 / 30 up); ValBench UBody
  (SPRINT-seen, reported only) +0.036 dB. Guard passes. (a) fails (needs >= +0.30 dB) -> **H4 not supported**. Both 4k fine-tunes
  sit ~0.06-0.10 dB below the original SPRINT, so 4k steps of either label set do not move the k = 0 avatar quality: the gain H1
  gives is per-frame fitting that this regressor does not absorb from 17k extra labelled rows at this scale.
- **Phase 5 result (2026-11-02 16:23, applying the rule; `O/reports/improve/p5_report.json`).** Margins chosen on the ValBench-v1 subset: MediaPipe
  1.25 (PSNR 17.66 / 18.08 / 17.86 for 1.1 / 1.25 / 1.4), track 1.25 (17.74 / 17.93 / 17.85). ValBench-S, PSNR / SSIM / LPIPS, wrong-person / lost
  frames of 1,325, e2e p50 / p95 ms on the server (host load in brackets): YOLOX + H1c 17.85 / 0.774 / 0.1987, 0 / 0, 889 / 2,156 (~270);
  MediaPipe 17.32 / 0.762 / 0.2086, 5 / 172, 136 / 1,761 (~380); track 17.39 / 0.761 / 0.2084, 2 / 1, 36 / 85 (~230). Neither alternative
  meets the quality condition (>= YOLOX - 0.10 dB, LPIPS <= +0.003, wrong-person <= YOLOX) -> **YOLOX + H1c stays**. The first track runs were
  blocked by a check added in b5f51fb2 (removed in 7c252b12) and rerun; the fast backend's output packing changed in b5f51fb2 mid-phase
  (YOLOX arm within 0.003 dB of the reference row). Laptop timing not supplied yet (script: eval_split/time_live_boxes.py).

## Fairness rule for every benchmark (registered 2026-11-02 21:25, user rule; applies to AvatarTest-v1, PublicBench-v1, PublicBench-v2 and FinalBench)
1. Every method uses exactly the same subjects / clips, the same training frames and the same test frames (the benchmark's
   registered split; for X-Humans the official train / test split).
2. Per-person methods (GART, ExAvatar, GaussianAvatar) train only on that subject's training frames. GUAVA and GUAVA-FB use the
   FIRST training frame as their only source image (declared in every caption).
3. Every method is driven by the same GT body model on the test frames (converted to each method's body model), plus the
   GUAVA-FB [SPRINT] rows, which use no GT. No test-time pose fitting in a main table: no pose, face or expression parameter of
   a test frame may be fitted to that test frame's image. Training-frame refinement inside a method's own training is allowed.
4. Same 512 person crop, same masks, same metrics (PSNR, SSIM, LPIPS-VGG, L1), computed by the one scoring script
   (`eval_split/score.py`) for every row.
5. Published numbers of other methods (e.g. ExAvatar's own X-Humans numbers) appear only on a separate "as reported" line,
   never in our tables (different crops / masks / metric pipelines).

### Audit of the existing tables against the rule (2026-11-02 21:25)
- Splits / sources (checked on every `frames.json`, 70 AvatarTest-v1 + 8 PublicBench-v1 clips): source key = first recon
  (training) key in all 78 clips; recon and test keys disjoint in all 78. GART (`run_gart.py`) and GaussianAvatar
  (`ga_convert.py`) train on the recon keys only and render the test keys with the test frame's GT pose; no test-time fitting.
  GUAVA / GUAVA-FB rows: source = first frame, targets = GT records of the test frames. One scorer, one `gt512` per benchmark.
  -> these rows conform.
- **Non-conforming: every ExAvatar row** (AvatarTest-v1 subset24 / subset23, PublicBench-v1). Its body is the GT, but its
  test-frame face parameters (jaw, eyes, expression) are fitted to the test images (DECA + DWPose keypoints + `fit_face_gt.py
  --stage test`, as registered in AvatarTest-v1 addendum 1: "its own FLAME face fitting allowed (on recon and test frames, as
  its pipeline does; stated)"). Under rule 3 this is test-time fitting. On PublicBench-v1 the GT is SMPL, so every other row
  has a neutral face while ExAvatar's face follows the test image.
- **Non-conforming: AvatarTest-v1 `table_subset24.md`**: ExAvatar's row covers 23 of the 24 clips (DECA finds no face on
  bedlam2/..._seq_000169) while the other rows cover 24. `table_subset23.md` (all rows on the same 23 clips) conforms on clips.
- Non-conforming (interim, chat only, 2026-11-02): the PublicBench-v1 partial table with ExAvatar / GaussianAvatar on 1 of 8
  clips. Not a reported table; replaced by the full one.
- Existing tables stay as they are (supporting material, user decision); the deviations above go into their captions.
- **ExAvatar, conforming variant (used in PublicBench-v2 and FinalBench):** same runner, except that the test-frame face is the
  GT's (converted jaw / expression as in `exa_common.py`; eyes 0; neutral where the GT has no face, i.e. SMPL GT), with no DECA,
  keypoints or face fitting on test frames. Face fitting on training frames (its training-frame refinement) is kept. Written as
  a new runner (`baselines/exavatar/run_clip_gtface.sh`); the running PublicBench-v1 jobs are not touched.

## PublicBench-v2: PublicBench-v1 + X-Humans 00028, 00034, 00087 (registered 2026-11-02 21:25, before any conversion or model run on X-Humans; user decision)
- **Data:** X-Humans (Shen et al., CVPR 2023; ETH AIT; research / non-commercial licence; never used for training), subjects
  00028 (male), 00034 (male), 00087 (male), the subjects of ExAvatar's X-Humans evaluation. 16.5 GiB downloaded to
  `/vault/research/COS/VIRLab/ubx858/ubx858/X_Humans/` (user-chosen location; nothing on /raid; the access link is not stored).
  Per take: rendered RGB (800 x 1200, white background) + depth (renders of textured scans), `cameras.npz` (one intrinsic,
  f = 2239 px; one extrinsic per frame), SMPL-X registrations per frame (`global_orient`, `body_pose` 63, hands 45 + 45
  (no PCA), `jaw` / eyes, `expression` 10, `transl`; betas = the subject's `mean_shape_smplx.npy`, gendered model).
- **Split (official):** README counts, train / test frames: 00028 1,444 / 448 (train Takes 2, 3, 4, 7, 9, 11, 13, 15; test
  Takes 6, 8, 14), 00034 1,411 / 300 (test Take 7), 00087 1,780 / 360 (test Takes 2, 5). One clip per subject; the clip's test
  frames = all frames of its test takes, training frames = all frames of its train takes (counts verified after extraction).
  Source (single-image methods) = the FIRST training frame: lowest train take number, lowest frame index.
- **Conversion (all rows share it; AvatarTest-v1 contract):** GT mesh = the dataset's SMPL-X parameters through SMPL-X v1.1
  (subject gender, mean_shape betas; `flat_hand_mean` chosen by the smaller vertex error against the provided `.ply` meshes,
  error reported). SMPL-X 2020 (neutral; GUAVA-FB, GUAVA with FLAME fields, ExAvatar's layer through `exa_common.py`) =
  pose + one shape per subject + jaw / expression fitted to that mesh, as PublicBench-v1's SMPL -> SMPL-X fit; SMPL for GART /
  GaussianAvatar = the SMPL-X -> SMPL vertex-correspondence fit used on AvatarTest-v1 (the dataset's own SMPL files are not
  used). Residuals reported. Camera = the dataset's intrinsic + per-frame extrinsic (reprojection check reported). Crop =
  1.1 x the projected GT-mesh box clipped to the image, 1024 -> 512. **Mask = rendered depth > 0** (exact: the images are
  renders of the scans; the same mask for every method; GT = image x mask on black). The SPRINT rows keep their own detector,
  crop and StyleMatte matte (inference-time signals).
- **Rows:** the six PublicBench-v1 rows (GUAVA released, GART, GaussianAvatar, ExAvatar [conforming variant above],
  GUAVA-FB [SMPL-X GT], GUAVA-FB [SPRINT]); official code, default settings, the same adapters. Test rendering of every test
  frame; scored by `score.py` (PSNR, SSIM, LPIPS, L1), per subject and overall (mean over clips and frame-weighted).
  ExAvatar's published X-Humans numbers only on an "as reported" line.
- **Budget (GPU jobs > 15 min need the user's go-ahead):** per-person training time scales with the number of training
  frames (AvatarTest-v1 rates: GaussianAvatar ~1 min, ExAvatar ~1.4 min per training frame), so all official training frames
  would take ~25-30 h (GaussianAvatar) and ~35-42 h (ExAvatar) per subject. Each method's time is measured on one subject
  (<= 15 min of its run, extrapolated) and reported before the long runs. If the projected finish on the GPUs free at launch is
  after 2026-11-07 18:00, every per-person method trains on every k-th official training frame (the same k, smallest of 2, 4,
  8 that fits; sampled within each take from its first frame), stated in the caption; test frames are never subsampled. The
  user decides at the go-ahead.

## FinalBench: final comparison tables (registered 2026-11-02 21:25, before scoring anything new; user design)
- **Clip pool:** AvatarTest-v1 UBody (30 clips) + PublicBench People-Snapshot (4) + NeuMan (4) + X-Humans (00028, 00034, 00087
  once converted) = 41 clips. Training / test frames, source, 512 crop, masks, GT = each benchmark's registered split and
  conversion (unchanged). BEDLAM2 (AvatarTest-v1, 40 clips) gets a separate synthetic table with the same rules.
  ValBench-v1 / ValBench-S are validation only and never appear in a final table.
- **One clip set for every method:** the intersection of the clips on which ALL methods of both tables finished. Excluded clips
  are listed with the reason (e.g. ExAvatar's DECA face detector finds no face).
- **Main table, single-image methods** (source = first training frame):
  1. GUAVA (released, best_160000.pt), GT-driven;
  2. GUAVA-FB [SMPL-X GT] (B28k, `guava_fb_B30k` best.pt), GT-driven;
  3. GUAVA-FB [SPRINT] (registered row: k = 0, largest box);
  4. GUAVA-FB [SPRINT, k = 0 + H1c] (H3, adopted for the cheap path in phase 4; the matched baseline of row 5);
  5. GUAVA-FB [SPRINT-H4, k = 0 + H1c] (`sprint_h4/train_h1/ema_4000.pt`, SHA-256 37a3a775...; failed its validation rule,
     shown as such, not a candidate);
  6. GUAVA-FB [SPRINT, k = 60 + H1c] (phase-3 selection; its registered command, extended to the FinalBench clips).
  Rows 4-6 are new test runs; their numbers are reported as they come out; nothing is chosen on them.
  **Amendment (2026-11-05, user decision, before any of rows 4-6 ran on a FinalBench clip): rows 4-6 are dropped; the main
  table is rows 1-3.** Row 6 fits each test frame's pose to that frame's own DWPose keypoints and StyleMatte matte, which rule 3
  of the fairness rule (registered at the same time) excludes from a main table; k = 60 is also outside the 33 ms live budget.
  The user's final-evaluation list (2026-11-03) has three single-image rows. No render of rows 4-6 exists under any test
  benchmark; their validation results stay in IMPROVE_NOTES.md, PHASE4_REPORT.md and the phase-6 records.
- **Reference table, video-trained per-person methods:** GART, ExAvatar (conforming variant only), GaussianAvatar on the same
  clips, captioned "trained on each person's video (training frames); not directly comparable with single-image methods".
- **Metrics / statistics:** `score.py` (PSNR, SSIM, LPIPS-VGG, L1); per clip = mean over its test frames; per dataset and
  overall = mean over clips (each clip weighs the same; frame-weighted means also in the CSV). 95 % CIs = percentile bootstrap
  over clips (10,000 resamples, numpy default_rng(20261108); overall resampled within each dataset). Paired differences vs
  GUAVA (released): per-clip row - GUAVA, mean with the same bootstrap CI, per dataset and overall (higher is better for PSNR /
  SSIM, lower for LPIPS / L1).
- **Disclosures (caption):** AvatarTest-v1 and PublicBench-v1 numbers of rows 1-3 and the per-person methods were seen before
  this registration; UBody clips are upper body only; NeuMan test sets are 3-10 frames; released GUAVA may have trained on
  UBody; X-Humans images are renders of scans; the k = 60 refinement fits each test frame's own DWPose keypoints and StyleMatte
  matte (inference-time signals, no GT).
- **Output:** `phase3_dnt/RESULTS_FINAL.md` + `phase3_dnt/results_final.csv` (per-clip rows + aggregates), built by
  `eval_split/final_tables.py` from the score files; by the 2026-11-08 freeze. Existing per-benchmark tables stay unchanged.

## Phase 6: background worker for the full-body live pipeline (registered 2026-11-03 01:15, before any code of this phase runs; user brief)
Goal (user): improve the live full-body avatar with an asynchronous background worker (pose correction, C3 UV-feature appearance
update, stretch: geometry refinement) WITHOUT raising the live per-frame cost and without retraining any network. Validation only;
test sets (FinalBench) only by amendment after a pass. Freeze 2026-11-08: anything not passed by then is reported as future work.
Prior evidence (disclosed): E2 (RUNLOG 30, GUAVA upper body, 1 clip, tracker poses) made the avatar worse (future LPIPS
0.1460 -> 0.1512, frame-0 LPIPS 0.106 -> 0.131); H4 (phase 4) found the k = 60 gain is per-frame and not learnable by SPRINT.

### Data (validation only; replay mode)
- **4D-DRESS validation** (GUAVA-FB's val split: 4 subjects, 66 takes). Split BY SUBJECT (numpy default_rng(20261103)
  permutation of the sorted ids): **tuning 00127, 00134; verdict 00137, 00151**. A replay session = all val takes of one
  (subject, outfit) in take order, one static camera, raw 30 fps frames (180 per take); cameras **0004 and 0052** (front / back,
  90 deg apart from the other two) -> 8 tuning + 8 verdict sessions (~40-60 s each). Source = the session's first frame.
  GT per frame = the dataset's SMPL-X (v1.1, gendered) converted as X-Humans (SMPL-X 2020 fitted to the mesh; SMPL for nothing);
  mask = the dataset's rendered mask (exact). Scored frames = every 3rd frame (10 fps) after the first 30 frames.
- **BEDLAM2**: ValBench-v1's 30 BEDLAM2 clips (a superset of GUAVA-FB's 10 val clips), split BY IDENTITY (same rng):
  tuning it_4102, it_4590, it_4591, it_4773, it_4781, nl_5820, us_0420, us_1117, us_1143, us_1479, us_1735, us_1863, us_2447,
  us_2495, us_2901; verdict it_4224, it_4548, it_4551, it_4634, nl_5510, nl_5764, nl_5955, nl_6929, us_1152, us_1494, us_1675,
  us_1765, us_1978, us_2343, us_2660. Replay = the clip's 30 fps video from its source frame; scored = its GT rows (6 fps) after
  the source; ValBench-v1 crops / masks / gt512 unchanged.
- **Tuning set** = 4D-DRESS 00127 / 00134 sessions + the 15 tuning BEDLAM2 clips; **verdict set** = 4D-DRESS 00137 / 00151
  sessions + the 15 verdict BEDLAM2 clips (rules are applied to the verdict set, pooled: mean over sessions / clips, each weight 1).
  **ValBench-S** (13 UBody clips): upper-body no-regression check only (k = 0 + H1c reference, PSNR not lower by > 0.10 dB).
- Disclosures: ValBench BEDLAM2 / ValBench-S were used for phase 2-5 decisions; 4D-DRESS val drove GUAVA-FB checkpoint selection.

### Replay mode (server)
- One process per session on one GPU. Frames released in order; the live loop processes EVERY frame (no drops) at
  clock_t = max(timestamp_t, finish_{t-1}); the worker runs in a separate thread on a low-priority CUDA stream (live stream
  high priority), sees frames when they are released, takes its real wall time; the live frame t uses only the latest version
  published (double-buffer swap) at a wall time <= the start of frame t's live processing. No locks on the render path.
- Live loop = the deployed k = 0 + H1c pipeline (YOLOX + H1c linking -> SPRINT from-scratch EMA 128k raw on its detector crop,
  shape lock to the source fit -> + current pose correction -> GUAVA-FB B28k render at 512 in SPRINT's crop camera -> warped into
  the metric crop by the exact homography, as the GUAVA-FB [SPRINT] row). YOLOX detections are computed once per frame and cached
  (identical for every arm: the box does not depend on the avatar); their measured CPU time enters the e2e numbers.
- Source avatar = sprint_avatar_fit's full fit of the source frame (150 steps, as the [SPRINT] row).

### Worker (common to steps 1-3)
- **Frame gate:** a released frame is offered only when no update is in flight (budget: one). Accepted iff
  (pose novelty >= 15 deg = min over accepted frames of the camera-frame root-orientation geodesic distance, OR new UV coverage
  gain >= 0.005 of the UV mask, OR it is the n-th frame since the last acceptance, n = 10) AND the refined fit is good: mean 2D
  keypoint error of the k = 60 fit <= 20 px at the 1024 crop (DWPose score >= 0.3 keypoints) AND the StyleMatte matte is
  non-empty (alpha > 0.5 on >= 1 % of the crop).
- **Empty-matte fix** (H1 refinement, all uses in this phase): if the matte is empty the mask / silhouette terms are dropped
  (keypoint-only refinement) and the frame is rejected by the gate.
- Then k = 60 refinement (H1 objective, as deployed) -> step 1, step 2, step 3 in that order; publish by atomic swap.

### Step 1: pose correction
- Correction from each refined frame = refined minus raw SPRINT: per-joint rotation offsets in the parent frame
  (R_ref R_raw^T for root, 21 body joints, jaw, 30 hand joints) + camera scale bias (log ratio of the crop-camera depth) and
  translation bias (x, y in crop units). Kept as an EMA over refined frames (rotations averaged in the tangent space at the
  current mean); applied to every live frame (R_live = R_offset R_raw) until the next update.
- Tuned on the tuning set only: EMA rate alpha in {0.2, 0.5, 1.0}; which parts are corrected in {all, body + root + camera,
  camera only}; offsets capped at 30 deg per joint. One setting is chosen by tuning-set PSNR (LPIPS as tie-break), then frozen.
- **Rule (verdict set):** PSNR >= +0.30 dB AND LPIPS not worse vs k = 0 + H1c; ValBench-S PSNR not lower by > 0.10 dB;
  live-loop p95 with the worker running <= p95 without + 2 ms, measured on the GPU segment (SPRINT forward + correction + render,
  CUDA events) with worker-on / worker-off blocks interleaved on the same frames (the CPU detector's time on this shared host is
  reported but not part of the rule).
- **Step 1 tuning result (2026-11-04, before any verdict-set run; scores `phase6_val/scores/*_tuning_full.json`).** Tuning PSNR
  pooled as the rule's pooling (mean over the 23 tuning clips, weight 1), difference vs k = 0 + H1c (19.74 dB clip-pooled frames):
  alpha 0.5 body +0.282 dB (LPIPS -0.0012), 0.2 body +0.276, 0.5 all +0.245, 0.2 all +0.240, 0.2 cam +0.169, 0.5 cam +0.151,
  1.0 all +0.125, 1.0 cam +0.059, 1.0 body -0.064. Every arm gains on BEDLAM2 (+0.26 to +0.76 dB) and loses on 4D-DRESS
  (-0.14 to -0.98 dB). **Frozen: alpha = 0.5, corrected parts = body + root + camera, cap 30 deg** (other worker settings as
  registered). Disclosure: 5 arms first crashed (CUDA illegal memory access) while 4 worker-on replays shared GPU 1; a
  single-arm repro of a crashed clip ran clean, and the missing clips were rerun 2 at a time (clips already finished were not
  rerun); the first, partial scores are kept under tag `tuning`.
- **Step 1 verdict (2026-11-04; `phase6_val/scores/*_verdict.json`, frozen setting from 9779ca48): NOT supported.** Verdict set,
  mean over the 23 clips (weight 1): k = 0 + H1c 21.147 dB -> step 1 21.192 dB (+0.045; rule needs >= +0.30), LPIPS -0.0005,
  11 / 23 clips up; BEDLAM2 +0.368 dB (11 / 15 up), 4D-DRESS -0.560 dB (0 / 8 up). The PSNR condition fails, so the ValBench-S and
  timing conditions cannot change the verdict (not run). Offline upper bound, per-frame k = 60 + H1c: +2.371 dB, LPIPS -0.0249,
  23 / 23 clips up (BEDLAM2 +2.32, 4D-DRESS +2.46): the pose information is per frame; offsets carried from refined frames to
  later frames do not transfer on 4D-DRESS. Step 2 therefore runs on top of k = 0 + H1c, as registered.

### Step 2: C3 appearance update (UV feature fusion), only after step 1 is decided (it runs on top of the adopted step 1, or of k = 0 if step 1 fails)
- For each accepted frame: GUAVA-FB's image encoder -> inverse texture mapping with the REFINED pose -> per-texel feature map +
  visibility (as e2_uv_fusion.py, GUAVA-FB's inferer). Texel weight = visibility x facing (|cos|) x fit quality
  (exp(-kpt_err / 20 px)) x sharpness (min(1, LapVar(crop) / LapVar(source crop))).
- Fusion: weighted running mean of the stored UV map; the source frame is the anchor with fixed prior weight 2 x its own weight;
  a texel is updated only where the frame weight >= 0.3 AND the stored confidence (accumulated weight) < 4; outlier texels
  (mean |RGB - fused RGB| > 0.25 where the stored weight >= 1) get weight 0; outlier frames (render-vs-image LPIPS of the frame
  under the current avatar > median + 3 MAD of the accepted frames so far, after >= 5 frames) are rejected.
- Face texels (SMPL-X head region) only if the head is frontal (yaw <= 30 deg, pitch <= 20 deg), sharp (face-crop LapVar >= 0.8 x the
  source's) and the face keypoint error <= 10 px; after every update the ArcFace similarity (onnx_arcface_torch) of the current
  avatar rendered in the source camera to the source image must not drop by > 0.01 vs the previous version, else that update is
  rolled back.
- Re-run the UV decoder on the fused map (vertex branch, style and camera features stay the source's; "template branch with
  averaged vertex features" only as a tuning option) and publish the new Gaussians.
- Stop per body region (head, torso, arms, legs; SMPL-X part labels of the UV texels): coverage >= 0.9 of ever-seen texels AND
  held-out gain < 0.002 LPIPS over the last 3 updates (held-out = every 6th released frame not fused); reopen when the mean
  person-region luminance changes by > 15 % vs the stop time.
- Tuning (tuning set): prior weight in {2, 4}, update threshold in {0.3, 0.5}, template-branch averaging off / on.
- **Rule (verdict set; scored frames never fused):** (>= 5 % relative LPIPS gain OR >= +0.30 dB) vs the static avatar (same
  driving); LPIPS on regions unseen at frame 0 improves (pixels whose surface texels had zero source visibility, rendered as a
  mask; masked LPIPS); CSIM (ArcFace, frames with a detected face) not lower by > 0.01; tLPIPS (mean over consecutive scored
  frames of |LPIPS(R_t, R_t-1) - LPIPS(G_t, G_t-1)|) not higher; ValBench-S no regression (as step 1).

### Step 3 (stretch; only if steps 1 and 2 pass before 2026-11-06): geometry refinement at fixed Gaussian count
- Reservoir of 8 accepted frames (greedy farthest root orientation). Background Adam, 300 steps, on per-Gaussian position offsets
  (|offset| <= 2 cm via tanh) and log-scales (+-0.5) only; colours / features frozen; loss = silhouette L1 (render alpha vs matte)
  + L1 + 0.2 LPIPS + 1e-2 x offset^2 (regulariser to the decoder's offsets). Relocation: Gaussians with opacity < 0.005 are moved
  to pixels where the silhouette misses the matte in >= 3 reservoir frames (back-projected onto the nearest surface point);
  Gaussian count unchanged.
- **Rule (verdict set, held-out frames):** silhouette IoU >= +0.01 AND LPIPS not worse; Gaussian count unchanged; live render ms
  unchanged within noise (p50 within +-3 %).

### Reporting
For k = 0 + H1c, step 1, step 1 + 2, step 1 + 2 + 3 and per-frame k = 60 + H1c (offline upper bound): PSNR / SSIM / LPIPS on the
scored (future) frames, unseen-region LPIPS, silhouette IoU, CSIM, tLPIPS; live-loop ms / frame p50 / p95 with and without the
worker (GPU segment and e2e); worker update rate and time per update. Per dataset (4D-DRESS, BEDLAM2) and pooled; tuning results
reported separately. Results in PHASE6_REPORT.md, here and RUNLOG; an adopted step enters FinalBench only by amendment before its
test run. Laptop: `--worker {off, pose, pose+uv, pose+uv+geo}` in the live app and the timing script, a recording mode (frames +
timestamps + live outputs; the user's consented "webcam set", never used for tuning), README with full-body framing and commands.
GPU: only GPUs not needed by the queued FinalBench runs (plan reported to the user before any phase-6 GPU job).

## Gap diagnosis D1-D3 (registered 2026-11-05 10:35, before any of its runs; user brief 2026-11-05: find, implement and validate a fix for the gap of GUAVA-FB [GT] to GART / GaussianAvatar / ExAvatar on People-Snapshot and NeuMan, within the RAGA live rules)
Evidence in hand (descriptive, `O/reports/improve/view_gap/view_gap.json`, PSNR by source->test view change): People-Snapshot, GUAVA-FB [GT]
vs GART: [0,30) 29.64 vs 30.70 (75 frames), [30,60) 27.01 vs 30.96 (20), [60,90) 27.19 vs 33.11 (23), [90,135) 26.72 vs 32.82 (31),
[135,180] 26.68 vs 31.02 (41); NeuMan [0,30) 23.72 vs 24.62 (13 of 27 frames); X-Humans [0,30) 26.46 vs 27.34 -> [135,180] 24.31 vs 26.40.
The gap is ~1 dB where the test view is near the source view and 4-6 dB once the view turns, on all three datasets.
- **D1, GT-pose appearance fusion over the training frames** (`eval_split/fusion_gt_row.py`): the clip's training frames (the frames
  the per-person baselines train on) are passed through GUAVA-FB's image encoder + inverse texture mapping with their GT poses and
  fused into the UV feature map by the step-2 running weighted mean (weight = visibility x |cos|, source anchor prior, update where
  weight >= threshold and accumulated weight < cap; no live gates), the UV decoder is re-run once, test frames are rendered as
  fb_gt. Feed-forward, no gradient step, no per-person training. Two arms, fixed now: `diag_fuse_reg` = prior 2, threshold 0.3,
  cap 4 (the registered step-2 defaults); `diag_fuse_oracle` = prior 1, threshold 0, no cap (every visible texel of every frame).
  Data: ValBench-v1 (validation, 50 clips) and PublicBench-v1 (test, 8 clips: descriptive, disclosed, no selection on it).
  Purpose: how much of the far-view gap is missing information (recoverable by feed-forward fusion of frames the baselines see)
  and how much is a model limitation. Interpretation rule (fixed): if the oracle arm closes >= 50 % of the People-Snapshot
  [30,180] gap to GART, the bottleneck is information and the deployable direction is online fusion (its own registration, validation
  rule on ValBench-v1 / ValBench-S, a pass before any FinalBench entry); below 50 %, the direction is training-side. Diagnostic rows
  never enter a results table; they are reported as diagnostics with these settings.
- **D2, cheap oracles on finished renders** (`eval_split/diag_oracles.py`, CPU, descriptive): per-frame PSNR of each row plain /
  x GT mask (spill removed) / after a per-frame affine colour fit / low-pass (sigma 4) / high-pass residual, by dataset and view bin,
  on PublicBench-v1 and ValBench-v1; bounds for silhouette, colour and detail explanations of the near-view gap.
- **D3, region / visibility classes on PublicBench-v1** (`eval_split/diag_regions.py` as in the improvement phase; descriptive).
GPU: GPU 1 (shared with another user's job), every job < 15 min. The FinalBench main table is unchanged by this section.
- **D1b (added 2026-11-05 ~18:35 after the D1 results, before its runs):** D1 result (disclosed): ValBench-v1 oracle +0.87 dB BEDLAM2 / +0.51 UBody
  (BEDLAM2 far-view bins within 0.1-0.15 dB of GART); PublicBench-v1: NeuMan 23.04 -> 24.14, People-Snapshot 27.95 -> 27.78 with the
  near-view bin [0,30) 29.64 -> 28.12 (blur + silhouette shift from averaging) and the far bins +0.5 to +1.0 dB. Two more fixed arms of
  the same diagnostic separate "averaging blurs / shifts the seen texels" from "filling the unseen texels is what helps":
  `diag_fuse_wta` = per texel the features of the single best-facing frame (no averaging; prior 1, threshold 0);
  `diag_fuse_fill` = texels visible in the source keep the source's features exactly, unseen texels take the best-facing frame's.
  Same data (ValBench-v1, PublicBench-v1 descriptive), same status (diagnostic, never a table row).

## Phase 7: fusion-aware GUAVA-FB (registered 2026-11-05 18:20, before any training run; follows the gap diagnosis above; user brief 2026-11-05)
**Finding behind it (RUNLOG 114):** feed-forward GT-pose fusion of a clip's training frames into the UV map already recovers most of the
far-view gap on exact-pose data (ValBench BEDLAM2 26.13 -> 27.00 dB, GART 26.71) and on NeuMan (23.04 -> 24.14), but on People-Snapshot
the near-view bin falls 29.64 -> 28.12 dB and the coverage misses double even when the source's own texels are left untouched (fill
arm): the UV decoder (StyleUNet + point decoder of B28k) has only ever seen single-view maps, zero where unseen, and is out of
distribution on filled maps. The deployed coverage misses themselves (hair, shoes, hands; 43 % of the PS squared error) are a second,
separate limit that this phase does not address.
**Hypothesis:** training the same network with multi-frame fused inputs makes fused maps in-distribution: the fused rows improve
on every validation set including the near-view bins, and the single-image path does not regress.
**Change (code 115, `fb_fusion.py`, `guava_fb_train.py --fuse-*`):** per training item, K support frames of the same group (K = 0 with
p = 0.3, else uniform 1..4; never the source or a target; 4D-DRESS: any view / take of the subject + outfit) are encoded like the
source and fused into its UV map by the mean rule (weight = visibility x |cos|, source prior 2, no threshold, no cap); the decoder
runs on the fused map with the source's style / camera / vertex branch; losses, targets and everything else = recipe B.
**Arms (fixed):** A = `--fuse-kmax 4 --fuse-p1 0.3 --fuse-mode mean --fuse-prior 2 --fuse-thr 0 --fuse-cap 0 --fuse-val-k 8`;
B (control) = identical command without fusion (`--fuse-kmax 0`). Both: B28k init, 4,000 steps, lr 8e-6 constant, recipe B options,
seed 0, validation every 2,000 (H2 continuation protocol); the checkpoint evaluated is step 4,000 of each arm (best.pt by the
unchanged eligibility rule is kept for the record). Budget ~2.3 h per arm on one L40S (0.7 it/s, peak 20 GB); GPUs 4 and 5 (free
of my jobs), after the user's go-ahead.
**Evaluation (ValBench-v1, GT-driven, score.py --tag vb, frame-weighted):** per arm the single-image row (`eval_guava_rows.py --row fb_gt
--fb-ckpt <arm 4k>`, name `p7_<arm>_single`) and the fused row over all recon frames (`fusion_gt_row.py --mode mean --prior 2 --thr 0
--cap 0 --fb-ckpt <arm 4k>`, name `p7_<arm>_fused`), view-bin tables by `view_bins.py`. References: fb_gt 20.98 dB / LPIPS 0.1553
(UBody 19.06, BEDLAM2 26.13); naive fusion with B28k = `diag_fuse_oracle` 21.59 / 0.1545 (19.57, 27.00).
**Decision rule (all must hold):** (1) A_fused >= diag_fuse_oracle + 0.30 dB on ValBench-v1 all and LPIPS not higher by > 0.002;
(2) A_single >= fb_gt - 0.10 dB and LPIPS not higher by > 0.002 (no regression of the deployed single-image path);
(3) A_fused >= A_single - 0.10 dB in the [0,30) view bin of both ValBench datasets (fusion no longer hurts near views).
Attribution (reported, not a pass condition): if A_fused - B_fused < 0.20 dB, the gain is attributed to continued training, not to
fusion-aware training, and is reported as such.
**On a pass, FinalBench addition (registered now):** a separate "video fusion" table on the 41 FinalBench clips (+ BEDLAM2): GUAVA-FB [GT]
+ feed-forward fusion of the clip's training frames (GT poses, no optimisation) with B28k (naive) and with the phase-7 model, captioned as
using the same frames the per-person methods train on; the phase-7 single-image row is added to the main table as an extra row only if
rule (2) holds (B28k rows 1-3 unchanged). Disclosure: the People-Snapshot / NeuMan numbers of the naive B28k fusion were seen (D1).
**RAGA:** no per-frame change (same architecture, Gaussian count and render path); the fused decode runs in the background worker as
in phase 6 step 2; the laptop 30 fps / render-equivalence checks apply to the deployed checkpoint swap.
**Amendment 1 (2026-11-06 01:55, before any arm is launched; user go-ahead 2026-11-06: "start train … adapting idea from here that is workable", RAGA constraints unchanged).**
- GPUs: 4 and 5 were at 80-84 % utilisation by other users' jobs at launch (nvidia-smi 01:49); the arms run on the least loaded cards instead: A on GPU 0, B on GPU 6, C on GPU 1. Wall time per arm will exceed the 2.3 h estimate; the comparison is by steps and unaffected.
- Arm C (added; same init, steps, lr, seed, options as A): A + `--fuse-jitter 0.5` (`support_jitter` in `guava_fb_train.py`): each support frame gets, with probability 0.5, an exposure / colour jitter (brightness gain log-uniform [0.8, 1.25], saturation [0.7, 1.3], per-channel white balance [0.95, 1.05]); the source and the targets are never jittered. Motivation: the laptop session analysis (user paste 2026-11-06) found the reference frame over-exposed by 18-20 % in brightness and 25-33 % in saturation relative to the first live frames, so the live worker fuses frames of different exposure into the source's map. Hypothesis: the decoder learns to render the target's exposure from a map whose filled texels carry another one (a learned colour alignment, free per frame), at no benchmark cost.
- Exposure-shift probe (registered now): the ValBench-v1 fused row with every support frame brightened x1.2 and desaturated x0.75 (`fusion_gt_row.py --support-gain 1.2 --support-sat 0.75`, to be added before evaluation; the fusion itself unchanged), for B28k, A and C; names `p7_<arm>_fused_exp`.
- Adoption rule between A and C: A stays the registered primary and the decision rule (1)-(3) applies to A. C is adopted instead of A if and only if C also passes (1)-(3), C_fused >= A_fused - 0.05 dB and C_single >= A_single - 0.05 dB on ValBench-v1 (no measurable benchmark cost), and C_fused_exp >= A_fused_exp + 0.20 dB (a measurable robustness gain). If A fails and C passes, C is reported as exploratory, not adopted.
- Checkpoint evaluated per arm: `checkpoints/latest.pt` at step 4,000 (saved at the step-4,000 validation; H2 protocol).
- The other laptop-side ideas of the same analysis (lock exposure / white balance before the reference capture, best-of-N reference frame, chunked worker) are live-app code changes outside this training and are checked by the RAGA laptop procedure; per-texel selection instead of the mean was already tested (D1b: 27.58 / 27.91 vs 28.12 dB on the PS near-view bin) and is not pursued; gradient steps in the worker (test-time fitting) are excluded by fairness rule 3.
