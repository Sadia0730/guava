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
