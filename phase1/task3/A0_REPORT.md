# Phase 1 · Task 3 — A0 go/no-go result (2026-10-11)

**Question.** Does a render of the subject's *own* GUAVA avatar tell the student which pose is
right, and does it do so where 2D keypoints don't?

**Answer.**
- **No-go** for AvatarLoop as designed, i.e. as a gradient loss for online adaptation. Fitting the
  student's pose to any avatar loss makes 3D error much worse.
- **One narrow signal survives:** the avatar's appearance features as a check for *leg swaps*. A
  smaller one shows on the upper body. Details and the recommendation are at the end.

All numbers are *measured* unless marked otherwise.

## Setup

- **Student.** (c-vitpose) `best.pt` (step 32,000, SHA-256 `dc7337170ef4443c…`).
- **Data.** 3DPW validation, 3 single-person tracks: `courtyard_jumpBench_01`, `outdoors_golf_00`,
  `outdoors_parcours_01`. 50 evenly spaced frames per track, all after the first 5 s, giving 150
  frames. GT-keypoint crop.
- **Avatar.** One per subject, enrolled from that subject's first 5 s:
  - EHM-Tracker on 150 frames at interval 6; 23, 21 and 24 of 25 frames kept;
  - GUAVA built from the kept frame with the highest DWPose face score;
  - about 204k Gaussians each.
- **Losses** (as in `phase1/task3/a0_gonogo.py`):
  - silhouette: 1 − soft IoU of the avatar's solid silhouette vs the StyleMatte matte;
  - part labels: set-valued cross-entropy of the avatar's rendered 15-part map vs Sapiens-Seg 0.3B
    labels;
  - DINOv2 ViT-B/14 feature distance between the avatar's coarse RGB render (composited over the
    frame with the person greyed out) and the frame, in windows that follow the pose: upper body,
    face and hands;
  - control: DWPose 2D keypoints, confidence-weighted Geman–McClure.
- **Ranking test** (no optimisation). Both poses are rendered with the true 3DPW camera
  (intrinsics carried through the crop):
  - *Overall*: the GT pose vs the student's pose.
  - *Per part*: the GT pose vs the GT pose with that part's local rotations taken from the student
    (legs; upper body incl. neck and head; hands = wrist rotations only, since 3DPW has no finger
    GT).
  - "GT pose" means 3DPW's body rotations on SMPL-X with the student's shape and fingers, so it is
    a hybrid.
  - Ranking accuracy = how often the loss is lower for the GT pose.
- **Fitting test.** 10 frames per track (30 in all), in the student's own camera. Adam for 60
  steps on axis-angle offsets of root + 21 joints and the camera translation; objective
  loss / loss₀ + 0.1 · Σ offset².

**Sanity checks** (all 150 frames).
- GT joints project 3.9 px (mean) from 3DPW's 2D ground truth.
- Converting the student to the true camera moves its keypoints by 4.2 px.
- Each per-part counterfactual is worse in 3D than the GT pose:

  | Part | GT worse-off than counterfactual in | Part error, GT vs counterfactual |
  |---|---|---|
  | Legs | 100% of frames | 16.8 vs 84.0 mm |
  | Upper body | 100% | 21.7 vs 83.5 mm |
  | Hands | 92% | 46.4 vs 56.2 mm |

**Runs** (GPU 6, shared with another user's 3.8 GB job):

| Run | Wall time | Output |
|---|---|---|
| Enrollment, 3 tracks | 5.3 + 6.4 + 6.7 min | `/raid/ubx858/outputs/phase1_task3/a0/ehm_run/tracked/` |
| Main A0 run (150 frames ranked, 30 fitted) | 21.0 min | `/raid/ubx858/outputs/phase1_task3/a0/run1/` (`run_info.json`: commits, command, seed) |
| Ranking on the student's swap and flip frames | 2.6 min | `/raid/ubx858/outputs/phase1_task3/a0/run1_special/` |
| Swap and flip rates, full validation (GPU 1) | 1.2 min | `/raid/ubx858/outputs/phase1_task3/swap_flip/c_vitpose_val_gtcrop/` |

- Peak GPU memory was **not measured**: the sampler failed. Peak host RSS of the main run was 4.5 GB.
- Cost per fitting step: silhouette 123 ms, part labels 122 ms, features 264 ms, DWPose 108 ms.
- Ranking takes 2.0 s per frame.

## 1. Ranking accuracy (how often the GT pose scores better), 150 frames

| Loss | Overall: GT vs student | Legs | Upper body | Hands (wrists) |
|---|---|---|---|---|
| Silhouette (solid) | 35% (53/150) | 69% (104/150) | **94%** (141/150) | 41% (62/150) |
| Part labels (Sapiens) | 39% (58/150) | 66% (99/150) | **91%** (137/150) | 55% (83/150) |
| Features: upper body | 39% (59/150) | 63% (94/150) | 87% (131/150) | 48% (72/150) |
| Features: face | 53% (79/150) | 44% (66/150) | 90% (135/150) | 45% (67/150) |
| Features: hands | 30% (45/150) | 60% (90/150) | 56% (84/150) | 59% (89/150) |
| Features: sum of the three | 32% (48/150) | 58% (87/150) | 79% (119/150) | 55% (82/150) |
| **DWPose keypoints (control)** | 43% (64/150) | 69% (104/150) | 80% (120/150) | **83%** (124/150) |

Overall ranking by track:

| Loss | jumpBench_01 | golf_00 | parcours_01 |
|---|---|---|---|
| Silhouette | 34% | 58% | 14% |
| Part labels | 40% | 62% | 14% |
| Features: face | 52% | 94% | 12% |
| Features: sum | 26% | 60% | 10% |
| DWPose (control) | 26% | 50% | 52% |

Each track has 50 frames.

**What this shows.**
- **No loss prefers the whole GT pose over the student's**, the keypoint control included (30–53%).
  The student is already near-optimal in 2D, and most of its 3D error is invisible in the image.
  The hybrid GT also has a few pixels of its own 2D misalignment. The per-part comparison removes
  this confound.
- **Upper body: the avatar beats keypoints** (silhouette 94%, part labels 91% and face features 90%,
  vs DWPose 80%).
- **Legs on ordinary frames: no advantage** (silhouette 69% = DWPose 69%).
- **Hands (wrist rotation only): keypoints are clearly better** (83% vs at most 59%).

## 2. Special frames

The 150 evenly spaced frames contain **no** student leg swap or flip. So the student's actual swap and
flip frames in an enrolled track were found from the full-validation run (§5) and ranked separately:
26 leg-swap frames and 1 flip frame, all in `outdoors_parcours_01` after its first 5 s. The DWPose
left/right-flip frames are pooled from both passes.

| Loss | Student leg swaps (26 frames): GT vs GT + student legs | Student flip (1 frame): GT vs student | DWPose L/R flips (5 frames): GT vs student | DWPose L/R flips (5 frames): GT vs GT + student legs |
|---|---|---|---|---|
| Silhouette (solid) | 69% (18/26) | 0/1 | 1/5 | 2/5 |
| Part labels (Sapiens) | 50% (13/26) | 0/1 | 1/5 | 3/5 |
| Features: upper body | 85% (22/26) | 0/1 | 2/5 | 4/5 |
| Features: face | 54% (14/26) | 0/1 | 1/5 | 2/5 |
| Features: hands | 69% (18/26) | 0/1 | 2/5 | 3/5 |
| **Features: sum of the three** | **88% (23/26)** | 0/1 | 1/5 | 4/5 |
| DWPose keypoints (control) | 38% (10/26) | 0/1 | 2/5 | 3/5 |

**Leg-swap frames.** The student's lower-body error there is 265 mm (median swap gain 267 mm).
- The avatar features pick the correct legs in 88% of frames, while DWPose prefers the swapped legs
  in 62%. DWPose is itself left/right-flipped on 4 of these 27 frames.
- **This is the only place where the avatar gives a signal keypoints don't**, but the evidence is
  thin: one subject (who wears shorts, so the legs show), and 26 frames forming about 8 separate
  swap events.
- Part labels do no better than chance there (50%).

**Flips and DWPose flips** (1 and 5 frames): too few to conclude anything.

## 3. Effect of fitting each loss (30 frames; mean, then change vs the student and number of frames improved)

| | MPJPE | PA-MPJPE | PVE | Lower body | Loss after / before |
|---|---|---|---|---|---|
| Student (before) | 73.3 | 48.1 | 78.4 | 103.3 | — |
| Silhouette | 101.8 (+28.5; 4/30) | 64.1 (+15.9; 4/30) | 120.8 (+42.4; 1/30) | 101.9 (−1.4; 19/30) | 0.66 |
| Part labels | 191.4 (+118.1; 2/30) | 110.3 (+62.1; 3/30) | 214.1 (+135.7; 2/30) | 244.0 (+140.7; 11/30) | 0.68 |
| Features (sum) | 171.2 (+97.9; 0/30) | 115.9 (+67.8; 0/30) | 217.8 (+139.4; 0/30) | 118.3 (+15.0; 13/30) | 1.42 |
| DWPose keypoints (control) | 81.0 (+7.7; 5/30) | 50.4 (+2.3; 12/30) | 88.9 (+10.5; 4/30) | 111.5 (+8.2; 10/30) | 0.38 |

**What this shows.**
- **Every loss makes 3D worse when the pose is fitted freely per frame**, the keypoint control too.
- **The avatar losses are 4–15× worse than keypoints.**
  - Silhouette and part labels lower their loss by about a third while 3D error rises sharply, so
    their minima are far from the true pose.
  - The feature loss rises during fitting (×1.42): its windows follow the pose and destabilise the
    optimisation.
- **Silhouette is the only loss that slightly helps the lower body** (−1.4 mm, 19/30 frames), while
  it wrecks the upper body (+31.8) and hands (+71.8).

## 4. Enrollment-shape test (student pose; body shape replaced by the EHM-Tracker enrollment shape)

| | MPJPE | PA-MPJPE | PVE |
|---|---|---|---|
| Student's shape | 78.1 | 49.5 | 85.8 |
| Enrollment shape | 85.2 (+7.1) | 50.5 (+1.0) | 94.5 (+8.7) |

- The enrollment shape is worse; it gives lower PVE on only 13% of the 150 frames.
- By track (PVE / MPJPE change): jumpBench +2.3 / +3.0, golf +20.3 / +16.5, parcours +3.4 / +2.1.
- EHM-Tracker's shape from 5 s of full-body 3DPW video, worst on the small, distant golf subject, is
  not better than the student's per-frame shape.

## 5. How often the student swaps legs or flips (all 3DPW validation, 10,413 frames, GT-keypoint crop)

| Event | Threshold | Per 1,000 frames | 95% CI (sequence bootstrap) | Frames |
|---|---|---|---|---|
| Leg swap | ≥ 50 mm gain from swapping knee+ankle | **5.7** | [0.6, 12.0] | 59 |
| Leg swap | ≥ 25 / ≥ 100 mm | 6.5 / 4.5 | — | 68 / 47 |
| Front/back flip | facing angle > 120° | **2.3** | [0.0, 6.8] | 24 |
| Front/back flip | > 90° / > 150° | 2.9 / 1.2 | — | 30 / 13 |

- The events are concentrated in a few sequences:
  - leg swaps: `outdoors_parcours_01` 30, `courtyard_dancing_00` 21, `outdoors_parcours_00` 5,
    `courtyard_hug_00` 2, `outdoors_crosscountry_00` 1;
  - flips: `courtyard_dancing_00` 21, `outdoors_parcours_00` 2, `outdoors_parcours_01` 1.
- Under DESIGN.md §5 (assessable if (a)'s rate is ≥ 5 per 1,000 on validation), leg swaps are
  **just** assessable and flips are **not**.
- **Upper bound on what fixing swaps can buy** (*estimate*): swap frames carry about 265 mm of
  lower-body error against a mean near 100 mm. Removing every swap would therefore lower the mean
  lower-body error by only about 1 mm.

## Caveats

- **Small samples.** 150 frames from 3 subjects for the main ranking. The swap result rests on one
  subject and about 8 events. The flip and DWPose-flip cells (n = 1 and 5) carry no information.
- **Hybrid GT.** The GT pose transfers 3DPW SMPL rotations to SMPL-X with the student's shape. Its
  3D error vs 3DPW is 24.1 mm MPJPE and 16.8 mm lower body, and its 2D error is a few pixels; this
  is why the "overall" column is confounded.
- **Matte failures.** StyleMatte loses large parts of the torso on some back-view frames (seen in
  `run1/courtyard_jumpBench_01_00169.jpg`), which corrupts the silhouette target. Sapiens' person
  mask looked cleaner on the same frames, but that was not measured.
- **Enrollment and appearance.** The avatar comes from one frame, with the back hallucinated;
  3DPW has no turnaround. Its legs come from GUAVA's untrained leg Gaussians, rendered solid.
- **Fitting setup.** Only one anchor (0.1) and 60 steps were tested. Fitting in the student's crop
  camera (focal 3,072 px vs about 460 px true) also limits how well any 2D loss can align.
- **Shared GPU.** Timings are from a shared GPU, and DWPose ran on the CPU (onnxruntime CUDA
  mismatch).

## Recommendation: **No-go** for (c) as designed; **partial go** only for a narrowed leg-swap test

- **No-go for gradient-based adaptation with avatar losses** (DESIGN.md §2–3 method (c)).
  Optimising the pose against silhouette, part labels or features makes MPJPE 28–118 mm worse per
  frame, even with the subject's own avatar and an anchor. LoRA updates on the same losses would
  follow the same gradients. I don't expect them to beat the keypoint control, which itself
  worsens 3D (+7.7 mm) when fitted freely.
- **Which body parts show a usable signal (as a ranking, not as a gradient):**
  - **Legs, swap frames only.** The avatar features pick the correct legs 88% of the time vs 38%
    for DWPose (26 frames, one subject). This is the one place where the avatar adds information
    keypoints lack. Swaps are rare, though (5.7 per 1,000 frames), so even a perfect fix is worth
    about 1 mm on average (estimate). It matters for visible glitches rather than for MPJPE.
  - **Upper body.** Silhouette, part labels and face features rank the correct upper body more
    reliably than DWPose (87–94% vs 80%), but fitting on them destroys the upper body. Usable at
    most as a verifier.
  - **Hands.** No: DWPose is better (83% vs at most 59%).
  - **Whole pose / flips.** No signal, and too few flips to test (2.3 per 1,000).
- **Sapiens.** Part labels were no better than chance on swap frames (50%), and were not better than
  the silhouette on legs or upper body (66% and 91%, vs 69% and 94%). Running Sapiens live is **not**
  worth it on this evidence (about
  100–200 ms per frame on the laptop, estimate).
- **If you want to keep a narrowed AvatarLoop alive**, the cheapest decisive next test is the
  *discrete* one from DESIGN.md §3: score "legs as predicted" vs "legs swapped" with the avatar
  features. Measure how often choosing the lower-loss hypothesis fixes a swap without creating
  new ones, on all 59 validation swap frames. That needs enrollment for `courtyard_dancing_00`,
  `outdoors_parcours_00` and `courtyard_hug_00`; two of these are two-person sequences, which need
  instance masks. Otherwise, I recommend closing method (c) and putting the effort into the
  student itself (data and supervision for the legs).

---

## Post-hoc follow-ups A1 and A2 (2026-10-11 evening; NOT part of the pre-registered test)

Both were requested after the A0 result and use the same student (c-vitpose, step 32k) and the
3DPW validation split. They are exploratory: thresholds and rules were chosen after seeing A0.

### A1 — upper-body-only fitting (same 3 enrolled tracks, same 30 fitting frames as A0)

- **What is fitted.** Only the spine (×3), neck, head, collars, shoulders and elbows. Root, legs,
  wrists, fingers and the camera translation stay at the student's prediction.
- **Procedure.** Same as A0: 60 Adam steps, anchor 0.1 on the squared axis-angle offsets,
  objective = loss / loss₀.
- **Losses.**
  - Silhouette.
  - Features: the upper-body and face DINOv2 windows of A0.
  - Control: DWPose upper-body keypoints (nose, eyes, ears, shoulders, elbows, wrists) plus face,
    as two equally weighted groups.
  - Features + keypoints: the mean of the two normalised losses.
- **Run.** GPU 6 (shared), 24.6 min (the budget was about 20). Output
  `/raid/ubx858/outputs/phase1_task3/a1/run1/`, with `run_info.json`.

Change vs the student, mean over 30 frames, with the share of frames that improve in brackets:

| | MPJPE | PA-MPJPE | PVE | Upper body | Hands | Lower body | Loss after / before |
|---|---|---|---|---|---|---|---|
| Student (before), mm | 73.3 | 48.1 | 78.4 | 49.7 | 90.3 | 103.3 | — |
| Silhouette | +25.7 (10%) | +19.9 (13%) | +41.9 (0%) | +21.3 (20%) | +87.0 (10%) | −0.1 (57%) | 0.84 |
| Features (upper body + face) | +45.7 (0%) | +33.7 (7%) | +67.5 (0%) | +38.6 (3%) | +131.6 (13%) | +0.1 (20%) | 1.27 |
| DWPose upper body + face (control) | +8.0 (23%) | +3.2 (40%) | +12.5 (20%) | +6.5 (27%) | +26.8 (27%) | −0.0 (53%) | 0.34 |
| Features + keypoints | +9.4 (17%) | +4.8 (30%) | +14.9 (20%) | +8.0 (33%) | +28.0 (27%) | +0.0 (53%) | 0.74 |

- Medians tell the same story. Upper-body change: silhouette +13.9, features +34.6, keypoints +5.7,
  combined +4.4 mm.
- By track, MPJPE / upper-body change:

  | Track | Silhouette | Features | Keypoints | Combined |
  |---|---|---|---|---|
  | jumpBench | +32.0 / +28.3 | +50.9 / +43.4 | +11.6 / +9.6 | +16.9 / +13.6 |
  | golf | +13.4 / +10.6 | +27.3 / +23.0 | +6.9 / +5.7 | +8.4 / +9.7 |
  | parcours | +31.6 / +25.0 | +58.7 / +49.4 | +5.3 / +4.2 | **+3.0 / +0.7** |

- Time per step: silhouette 127 ms, features 245 ms, keypoints 107 ms, combined 258 ms.

**What this shows.**
- **Restricting the fit to the upper body does not rescue the avatar losses.** Upper-body error
  still rises by 21 mm (silhouette) and 39 mm (features), and only 3–20% of frames improve.
- **The feature loss is not even minimised** (×1.27): its windows move with the pose and make the
  optimisation unstable.
- **Hands get much worse in every variant.** Moving the shoulders and elbows carries the wrists
  away, and nothing in these losses constrains the hands.
- **Keypoints are the least harmful, but still harmful** (+6.5 mm upper body). Adding features to
  keypoints does not beat keypoints alone: +8.0 vs +6.5 overall. The single exception is parcours
  (+0.7 vs +4.2), so it is not consistent.
- **So per-frame fitting from the student's prediction to image evidence lowers the loss but not
  the 3D error.** That holds for 2D keypoints too. The upper-body ranking advantage seen in A0
  (silhouette 94% vs keypoints 80%) does not turn into a better pose under optimisation.

### A2 — leg swaps without an avatar: a causal temporal rule (all 3DPW validation)

- **Two hypotheses per frame:**
  - H0: legs as predicted;
  - H1: legs swapped, with left/right hip, knee, ankle and foot rotations exchanged and mirrored
    (R → M R M, M = diag(−1, 1, 1)).
- **Rule.** In time order within each track, keep the hypothesis whose knees and ankles
  (pelvis-centred, camera frame) are closer to the previous frame's **output**. The first frame
  keeps H0. Variants: a margin (switch only if H1 is closer by more than m mm); comparing with the
  previous **raw** prediction; resetting at gaps (tracks have gaps of up to 106 frames).
- **Measures.** Swaps (the A0 definition, ≥ 50 mm) and standard metrics over all 10,413 frames.
  Before any rule, 59 swaps.
- **Run.** GPU 1 (shared), 84 s, output `/raid/ubx858/outputs/phase1_task3/a2/c_vitpose_val_gtcrop/`.
  The variants were recomputed on the CPU from `per_frame.npz`.

| Rule | Frames switched to H1 | Swaps fixed | Swaps created | Swaps after | Lower body, all frames (89.61 before) | MPJPE, all frames (80.94 before) | Lower body on the 59 swap frames (263.3 before) |
|---|---|---|---|---|---|---|---|
| Previous output, no margin | 3,680 | 21 | 1,207 | 1,245 | 123.84 | 91.41 | 228.8 |
| Previous output, margin 50 mm | 1,252 | 20 | 618 | 657 | 102.69 | 85.03 | 229.5 |
| Previous output, margin 100 mm | 465 | 20 | 265 | 304 | 94.75 | 82.52 | 230.1 |
| Previous output, reset at gaps | 2,984 | 19 | 960 | 1,000 | 116.32 | — | — |
| Previous raw prediction | 108 | 9 | 24 | 74 | 89.80 | — | — |
| *Oracle: per frame, the hypothesis with lower true lower-body error* | 1,618 | 23 | 2 | 38 | 87.60 | 80.36 | 223.2 |

**What this shows.**
- **No simple temporal rule helps; every variant creates more swaps than it fixes.**
  - Rules that compare with the previous output lock in: after one wrong switch, the swapped state
    is "consistent" from then on.
  - Comparing with the raw previous prediction avoids lock-in. It cannot fix runs of swapped frames,
    and it still creates more swaps (24) than it fixes (9).
- **Most "swaps" are not leg swaps.** 23 of the 59 swap frames are whole-body front/back flips
  (facing angle > 120°): all 21 in `courtyard_dancing_00` and 2 in `outdoors_parcours_00`.
  Exchanging the legs cannot fix those.
- **Even a perfect selector between H0 and H1 (the oracle) fixes only 23 of 59.** It is mostly
  `outdoors_parcours_01`, where the mirrored legs fix 19 of its 30 swap frames. The oracle lowers
  mean lower-body error by only 2.0 mm (MPJPE −0.6). So the ceiling for any leg-swap selector on
  this student is small, avatar-based or not.

### Updated conclusion (A0 + A1 + A2)

The A0 recommendation stands: **no-go for avatar losses as an adaptation signal.** The two
follow-ups remove the remaining hopes for a cheap fix.
- **A1:** the avatar's upper-body ranking advantage does not survive optimisation.
- **A2:** leg swaps are rare, a third of them are really whole-body flips, and even an oracle
  swap-selector gains about 2 mm of lower-body error.
- **A narrow, verifier-only use of avatar features for leg swaps** (88% vs 38% on 26 frames of one
  subject, §2) remains untested beyond that track. Its ceiling on this student is the oracle above.
