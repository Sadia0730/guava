# Phase 1 · Task 3 — A3 (geometry ranking) and A4 (learned render-and-compare refiner)

Started 2026-10-12. This is a new direction after A0 / A1 / A2 (see `A0_REPORT.md`).
Hand-designed avatar losses that are backpropagated to the pose fail for three reasons: they don't
see depth, their minimum is not the true pose, and appearance mismatch gets blamed on the pose.

**Two go/no-go checks.**
- **A3:** do *geometric* cues estimated from the live frame (surface normals, relative depth) rank
  the true pose above the student's better than appearance and 2D keypoints do?
- **A4:** does a small *learned* refiner that sees a render of the predicted body reduce joint error
  more than the same refiner without the render?

**Rules.** All numbers are *measured* unless marked as estimates. Any GPU job longer than 15 min is
proposed with its exact command, duration and memory, and waits for approval. The running L-data
job is not touched.

## Plan (written before running anything)

### A3 — geometry ranking test (≤ 15 min GPU, no approval needed)

- **Frames.** The same 150 frames as A0: 3 enrolled 3DPW-validation tracks, 50 frames each after
  the first 5 s, GT-keypoint crop, student (c-vitpose) step 32k.
- **Comparisons.** As in A0, both poses are rendered with the true 3DPW camera.
  - Overall: the GT pose vs the student's pose.
  - Per part: the GT pose vs the GT pose with one part (legs / upper body / hands = wrists) taken
    from the student.
- **Rendered geometry.** The subject's avatar body mesh: GUAVA's posed SMPL-X with the avatar's
  enrolled static offsets and joint offsets, and the student's shape, as in A0. It is rasterised
  with PyTorch3D at 256 px, giving camera-frame surface normals, depth (z) and a silhouette.
- **Estimated geometry.**
  - Sapiens-Normal 0.3B and Sapiens-Depth 0.3B, run on the same 1024×768 crop as the Sapiens
    segmentation in A0 and warped into the 256 crop.
  - Person mask: Sapiens-Seg's non-background classes, not StyleMatte, which failed on back views
    in A0.
  - Licence, speed on this server and a laptop estimate are reported.
- **Normal convention.** Fixed once for all frames: the axis-sign pattern (one of 8) that best
  matches the GT-pose render on the first 10 frames. This uses GT only to fix a global coordinate
  convention, never per frame.
- **Losses**, all over pixels where the rendered mesh and the person mask overlap:
  - normals: mean (1 − cos);
  - relative depth: the estimated depth is aligned to the rendered depth by a least-squares scale
    and shift, then the loss is the mean absolute residual in metres;
  - silhouette: 1 − IoU(rendered mask, person mask);
  - normals + silhouette: their sum, equal weights;
  - DWPose keypoint control, exactly as in A0.
- **Output.** Ranking accuracy (how often the GT pose scores better), overall and per part, with
  frame counts, in the layout of A0's table.

### A4 — learned render-and-compare refiner (prototype; training needs approval)

- **Frozen student.** (L-0.1) `best.pt` (step 102k). It supplies image tokens (ViTPose-S output
  after the 1×1 expansion, 1280 × 16 × 12) and the initial pose.
- **Render.** The plain SMPL-X body mesh at the current pose, rendered in the student's own crop
  camera (focal 3,072 px, 256 px). Channels: part-colour RGB (fixed left/right-distinct colour per
  body part), camera-frame normals, silhouette. It is centre-cropped to 256×192 like the student's
  input. No GUAVA. PyTorch3D, no gradient through the render.
- **Refiner** (about 4 M parameters, *estimate*):
  - image tokens projected to 256;
  - render → 4 strided convolutions → 256 × 16 × 12, so the two token grids are spatially aligned,
    concatenated per token and projected;
  - one token for the current pose (22 × 6D rotations + camera (tx, ty, s));
  - a 4-layer transformer (width 256, 8 heads);
  - output from the pose token: a residual rotation per joint (6D, root + 21 body joints) and a
    camera update.
  - The output layer starts at zero, so the untrained refiner is the identity.
- **Variants.**
  - (ii) No render: identical network, with the render input replaced by zeros. Same capacity; it
    still sees the image tokens and the current pose.
  - (iii) With render.
- **Training data.** BEDLAM2 train split (the Task 1 GT dataset and augmentation). Starting pose
  per sample:
  - 50%: the frozen student's own prediction on that crop;
  - 50%: the GT pose with real student errors applied, i.e. the per-joint error rotations and
    camera error of a randomly drawn BEDLAM2-*validation* frame (an "error bank" built once from
    (L-0.1) on 20,000 held-out validation crops). Joint correlations are preserved.
- **Supervision.** 3D GT through the Task 1 GT losses on the refined output: root and body
  geodesic, root-relative 3D joints, camera, 2D. Weights as in variant (c); shape, hands and face
  are not refined.
- **Training.** AdamW, LR 3e-4, warm-up then cosine, batch 64, bf16 autocast for the frozen
  student. (ii) and (iii) use the same seed, data order and starting poses.
  - Budget (*estimate*): about 3 h each, run in parallel on two GPUs not used by L-data. This is
    proposed for approval with exact commands once the smoke tests give measured speeds.
- **Evaluation.** 3DPW validation, all 10,413 frames, GT-keypoint crop, (L-0.1) predictions as
  input. One refinement step (two reported as a secondary result). Metrics come from the standard
  evaluator, imported unchanged: MPJPE, PA-MPJPE, PVE, lower body. Runtime per frame: render plus
  refiner, at batch 1 and batched.
- **Go criterion** (pre-registered; my proposal for "clear margin", please confirm or change):
  (iii) beats both (i) and (ii) by ≥ 2.0 mm MPJPE, with the paired sequence-bootstrap 95% CI of
  each difference excluding 0, and PA-MPJPE not worse than (i).

## A3 results (2026-10-12)

**Run.** GPU 0 (free at launch), 3.5 min wall. Output `/raid/ubx858/outputs/phase1_task3/a3/run1/`
(`run_info.json`: command, commits, hashes). Same 150 frames, student and true-camera setup as A0.
The DWPose control reproduces A0's per-frame values on all 150 frames.

**Normal convention.** Sign pattern (+x, −y, −z), i.e. Sapiens normals use x right, y up, z toward
the camera, while the renders use OpenCV axes. At the GT pose on the 10 calibration frames, the mean
(1 − cos) is 0.145 for this pattern and at least 0.25 for every other.

**Depth alignment.** The fitted depth scale is positive on 92% of frames at the GT pose. On the other
8% (3 frames listed in the summary), the alignment flips the depth profile, a sign of a noisy estimate
or a small overlap.

**Missing data.** On 9 of 50 `outdoors_parcours_01` frames, Sapiens-Seg returns an almost empty
person mask even though the subject is clearly visible (e.g. `run1/outdoors_parcours_01_00197_a3.jpg`).
Normals and depth are undefined there, so those rows use 141 frames. The mesh-silhouette rows keep
all 150, so on those 9 frames the silhouette comparison is essentially noise.

### Ranking accuracy (how often the GT pose scores better), same 150 frames as A0

| Loss | Overall: GT vs student | Legs | Upper body | Hands (wrists) |
|---|---|---|---|---|
| Normals (Sapiens-Normal vs mesh) | 46% (65/141) | 77% (109/141) | 79% (110/140) | 35% (49/141) |
| **Relative depth** (Sapiens-Depth vs mesh) | **64% (90/141)** | 69% (97/141) | 74% (103/140) | 49% (69/141) |
| Mesh silhouette (Sapiens-Seg mask) | 50% (75/150) | 73% (109/150) | 65% (98/150) | 71% (107/150) |
| **Normals + silhouette** | 48% (67/141) | **79% (111/141)** | 71% (99/140) | 58% (82/141) |
| DWPose keypoints (control) | 43% (64/150) | 69% (104/150) | **80% (120/150)** | **83% (124/150)** |
| *A0, for reference: GUAVA silhouette / part labels / features* | *35% / 39% / 32%* | *69% / 66% / 58%* | *94% / 91% / 79%* | *41% / 55% / 55%* |

**By track** (jumpBench / golf / parcours):

| Loss | Overall | Legs |
|---|---|---|
| Normals | 36% / 74% / 24% | 72% / 86% / 73% |
| Relative depth | 44% / **90%** / 56% | 56% / 78% / 73% |
| Mesh silhouette | 42% / 58% / 50% | 82% / 60% / 76% |
| Normals + silhouette | 32% / 70% / 39% | 82% / 70% / 85% |
| DWPose (control) | 26% / 50% / 52% | 82% / 44% / 82% |

**What this shows.**
- **Relative depth is the only cue so far that prefers the whole GT pose over the student's more
  often than not** (64%). Keypoints score 43%, and every A0 appearance loss 32–53%. The signal comes
  mostly from one track, though: golf 90%, against 44% and 56% on the other two.
- **Legs.** Normals + silhouette (79%) and normals (77%) beat keypoints (69%) by 8–10 points on
  average. Per track they are better on golf and parcours, and level with keypoints on jumpBench.
- **Upper body: no gain.** Normals (79%) equal keypoints (80%). A0's GUAVA silhouette (94%) remains
  the best upper-body ranker, but A1 showed it doesn't survive optimisation.
- **Hands (wrist rotation): keypoints win clearly.**
- **Verdict for A3: weak go for geometry as an *input*, not as a hand-designed loss.** Depth and
  normals carry some information about errors that are invisible in 2D (depth, leg configuration)
  that keypoints don't. The effect is modest and track-dependent. Given A0 and A1, the way to test
  whether it helps is a learned comparison (A4), not optimisation of these losses.

### Sapiens-Normal / Sapiens-Depth 0.3B

- **Licence:** CC-BY-NC-4.0 on both model cards; research use only.
- **Files:** in `/raid/ubx858/models/sapiens/`, outside the repo.
  - Normal: SHA-256 `aa2db29f0033e741…`, 1.36 GB.
  - Depth: SHA-256 `65054e6b6083171b…`, 1.28 GB.
- **Speed on this server** (*measured*; L40S GPU 1, shared with another user's job at about 69%
  utilisation; 1024×768 input, 10 timed runs):

  | Model | fp32 | bf16 | Peak memory (bf16) |
  |---|---|---|---|
  | Normal | 323 ms | 109 ms | 2.0 GiB |
  | Depth | 368 ms | 129 ms | 2.6 GiB |

  The segmentation model of the same size took 49 ms in bf16 on an idle GPU, so the shared GPU
  inflates these numbers.
- **RTX 3080 laptop** (*estimate, not measured*): about 100–200 ms per frame for each model in bf16,
  scaled from the idle-GPU segmentation timing as in DESIGN.md. Two of them come to about
  200–400 ms per frame. Not real-time; usable only on a few frames per second, e.g. on adaptation
  frames.

## A4 status: implemented and smoke-tested; full training NOT started (needs approval)

**Code.**
- PEAR `models/render_refiner.py`: `BodyRenderer` and `RenderRefiner`.
- PEAR `train_render_refiner.py`: trainer and error-bank builder.
- PEAR `configs/phase1_task3/a4_refiner_{render,norender}.yaml`: the two configs, identical except
  `use_render`.
- guava `phase1/task3/a4_eval_3dpw.py`: 3DPW evaluation with the standard evaluator's functions.

**Changes from the plan** (forced by measurements):
- **The render is now 128×128** (focal 1,536, crop 128×96), with PyTorch3D's coarse-to-fine
  binning. At 256 px the naive rasteriser took 1,739 ms per batch of 64 (*measured*), and
  1,739 / 709 / 454 / 252 ms for 256-naive / 256-binned / 128-naive / 128-binned. At 128×96 the
  render still matches the 16×12 token grid.
- **The render encoder has 3 strided convolutions** (128×96 → 16×12), the same in both variants.
- **Parameters:** 4,177,287 for the refiner (*measured*), identical for both variants. They also start
  from identical weights (init hash `b3d6064aa01f…`).

**Error bank.** `/raid/ubx858/outputs/phase1_task3/a4/error_bank_L01_bedlam2val.npz`.
- 20,000 evenly spaced BEDLAM2-validation crops, (L-0.1) step 102k, 31 s on GPU 1.
- The student's mean geodesic error there is 13.6° at the root and 15.6° over body joints. Per-joint
  values are in the `.json`.

**Smoke tests** (GPU 1, shared). These are not results; they show only that both variants train
and the evaluation runs.

| | No render (ii) | Render (iii) |
|---|---|---|
| Training speed (*measured*) | 228 img/s | 115 img/s |
| Peak memory (*measured*) | 2.1 GiB | 1.8 GiB |
| BEDLAM2-val root-relative MPJPE (22 joints), after 150 steps | 73.3 → 61.8 mm | 73.3 → 62.0 mm |

A 640-frame 3DPW evaluation with the 150-step checkpoints ran in 65 s.
- PA-MPJPE fell from 47.8 to 39.4 mm (no render) and 39.3 mm (render).
- MPJPE rose slightly (73.1 → 74.7 and 74.6), and upper-body error rose by 8 mm.
- Applying the refiner twice made MPJPE much worse (about 105 mm), so a second iteration needs
  training for it.

**Runtime per frame** (batch 1, *measured* on the shared GPU 1):

| Stage | Time |
|---|---|
| Student | 19.6 ms |
| Refiner without render | 3.2 ms |
| Body render (PyTorch3D) | 75.3 ms |
| Refiner with render | 3.6 ms |

The render dominates and would need a faster rasteriser (e.g. nvdiffrast or point splatting) for
real-time use.

**Proposed training runs (awaiting approval).** 20,000 steps each, batch 64, as in the configs.
Durations are *estimates* from the measured smoke speeds on a shared GPU. Both GPUs are shared with
another user's job; neither is used by L-data (GPU 5).

| Run | GPU | Duration | Memory |
|---|---|---|---|
| (ii) no render | 1 | about 1.6 h | about 2.1 GiB peak allocated, about 3–4 GB with the CUDA context |
| (iii) render | 6 | about 3.2 h | about 1.8 GiB peak allocated, about 3–4 GB with the CUDA context |

```
cd /home/ubx858/guava/third_party/PEAR
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib \
CUDA_VISIBLE_DEVICES=1 nohup python train_render_refiner.py --config configs/phase1_task3/a4_refiner_norender.yaml \
  --output-dir /raid/ubx858/outputs/phase1_task3/a4/norender --device cuda:0 > /raid/ubx858/outputs/phase1_task3/a4/norender.log 2>&1 &
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib \
CUDA_VISIBLE_DEVICES=6 nohup python train_render_refiner.py --config configs/phase1_task3/a4_refiner_render.yaml \
  --output-dir /raid/ubx858/outputs/phase1_task3/a4/render --device cuda:0 > /raid/ubx858/outputs/phase1_task3/a4/render.log 2>&1 &
```

**Evaluation afterwards** (≤ 15 min, *estimate*; all 10,413 3DPW-validation frames):

```
cd /home/ubx858/guava && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib \
CUDA_VISIBLE_DEVICES=1 python phase1/task3/a4_eval_3dpw.py \
  --refiner-norender /raid/ubx858/outputs/phase1_task3/a4/norender/checkpoints/latest.pt \
  --refiner-render /raid/ubx858/outputs/phase1_task3/a4/render/checkpoints/latest.pt \
  --out /raid/ubx858/outputs/phase1_task3/a4/eval_val --device cuda:0
```

**Criterion to confirm.** The go criterion in the plan uses MPJPE as the primary metric. The smoke
test suggests the refiner may improve PA-MPJPE and lower-body error but not MPJPE, i.e. it fixes
articulation but not global orientation. Please confirm MPJPE as primary, or switch to PA-MPJPE,
before the evaluation is run.
