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
