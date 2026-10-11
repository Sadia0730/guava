# Phase 1 · Task 3 — The decisive test: design

Status: design only, written 2026-10-11. No implementation beyond the feasibility check in §6.
Every number marked *measured* comes from a run in this repository, with its output path
given. Everything else is an estimate or a choice, and is labelled that way.

**Question.** Does avatar-based supervision give the student a signal that 2D keypoints don't?

**Methods.** The student, the adapter and the update schedule are identical in all three; only
the self-supervised loss differs.

| | Method | Self-supervised signal at test time |
|---|---|---|
| (a) | No adaptation | none (frozen student) |
| (b) | Online adaptation, 2D keypoints | off-the-shelf whole-body keypoints (DWPose), confidence-weighted reprojection loss |
| (c) | AvatarLoop | the user's GUAVA avatar rendered at the predicted pose, compared with the live frame: silhouette / part labels (legs), feature loss (upper body, hands, face) |

**Models.**
- Development: (c-vitpose) `best.pt`, step 32,000, single-expert ViTPose-S code (PEAR `4e58a33`).
  Path `/raid/ubx858/outputs/phase1_task1/screen/c_vitpose/checkpoints/best.pt`, SHA-256 `dc7337170ef4443c…`.
- Final experiment: the winning Task 1 long-run checkpoint, (L-0.1), (L-0.05) or (L-data), once that is decided. Hyperparameters are frozen on 3DPW validation with the development model, then re-checked on validation with the final model (no re-tuning unless validation regresses; any re-tuning is reported).
- GUAVA `assets/GUAVA/checkpoints/best_160000.pt` (SHA-256 `c2ffe92cc01314eb…`).
- DWPose and YOLOX from EHM-Tracker: `dw-ll_ucoco_384.onnx` (`724f4ff2439ed61a…`) and `yolox_l.onnx` (`7860ae79de6c89a3…`).
- StyleMatte `stylematte_synth.pt` (`5ce985571e909b66…`).

---

## 1. Causal protocol on 3DPW

**Splits.** Develop and tune on 3DPW **validation**; run the final comparison once on **test**.
3DPW is unseen by the student (Task 1). Track counts and lengths (*measured*, standard evaluator's `load_samples`):

| Split | Person-frames | Tracks | Track length | Scored frames if the first 5 s are excluded |
|---|---|---|---|---|
| validation | 10,413 | 16 (12 sequences) | 9–55 s, median 20 s | 8,039 (77.2%) |
| test | 35,515 | 37 (24 sequences, 13 with two people) | 13–73 s | 30,061 (84.6%) |

A 10 s window would keep only about 55% of validation frames (computed the same way), so the
window is **N = 5 s** (150 frames at 30 fps).

**Session = one track.** Each 3DPW track is one user session with its own avatar and its own
adapter, reset to the base weights at the start. Two-person sequences give two independent sessions.

**Enrollment (first N = 5 s of the track).**
1. EHM-Tracker runs on the window frames and gives the identity: SMPL-X shape, `joints_offset`,
   head and hand scale, and FLAME shape. Identity is tracked jointly over the window. Pose is not used.
2. GUAVA builds the avatar feed-forward from **one** source frame. The frame is chosen
   automatically: the largest person box whose face EHM-Tracker tracks successfully. GT is not
   used. Building took 1.04–1.21 s (*measured*, §6).
3. Window frames are **excluded from scoring**.

**Online loop (scored part).** For each frame t in time order:
1. Predict frame t with the current adapter. This is the prediction that gets scored.
2. Add frame t to the replay buffer, with its self-supervision targets and the frozen model's
   prediction.
3. Every k frames, update the adapter from the buffer (§2).

So the prediction for frame t only ever depends on adapter updates computed from frames before t.
At t = N the buffer is seeded with the enrollment-window frames for **both** (b) and (c); they are
past frames, and the avatar exists from that moment.

**Allowed per-frame precomputation.** The detector box, DWPose keypoints and person mask for frame
t depend only on frame t. They can therefore be computed offline and cached without breaking
causality.

**Crop.** The **detector crop** (YOLOX box from DWPose's detector, `training_square_box`, as in
the standard evaluator's `--crop detector`) is used for prediction, adaptation and scoring. GT is
used only to associate the detection with the track (the evaluator's IoU rule) and to score; it
never enters a loss. (a) is also reported with the GT-keypoint crop, to connect to the Task 1
tables.

**Sanity requirement.** On the scored frames, (a) must reproduce the standard evaluator's per-frame
values exactly (detector crop, same checkpoint). The online harness imports the evaluator's
functions (`load_samples`, `CropDataset`, `predicted_smplx_vertices`, `frame_metrics`) unchanged
from `tools/eval_3dpw_standard.py` (tag `eval-3dpw-standard-v1`, SHA recorded per run).

**3DPW has no turnaround.** AvatarLoop's intended enrollment has the user turn around in front of
the camera. In 3DPW the avatar comes from a single frame, and GUAVA hallucinates the back.
Consequences:
- Feature losses are gated to views near the enrollment view (§3).
- Front/back-flip results on 3DPW understate what a turnaround enrollment could do.
- On this point the protocol is biased **against** (c). This is stated now so it cannot be
  invoked after the fact.

Other 3DPW properties that work against (c):
- Subjects are often small in the frame (outdoor, full body), so face and hand crops are
  low-resolution.
- Some sequences involve objects (bench, bicycle, bags) that the person mask may include.

**Seeds.** Each method runs with 3 replay-sampling seeds. Everything else is deterministic, except
the rasteriser's backward pass (atomic adds). Its run-to-run spread is reported (§6 shows an
example).

## 2. Adapter (identical for (b) and (c))

| Item | Choice |
|---|---|
| Where | LoRA on the 42 `Linear` layers of the student's 6 decoder layers (7 per layer); backbone, decoder norms, regression heads frozen |
| Rank | r = 8, α = r (scale 1). 0.651 M trainable parameters (r = 4: 0.326 M; r = 16: 1.303 M) vs 40.5 M in the head |
| Init | A Kaiming-uniform, B = 0, so the adapted model is exactly the base model at t = N (unit-tested) |
| Optimizer | Adam, β = (0.9, 0.999), no weight decay, grad-norm clip 1.0, fp32. LR 1e-4, tuned on validation over {3e-5, 1e-4, 3e-4} |
| Update | every k = 8 frames (0.27 s at 30 fps), 1 optimizer step |
| Batch | B = 8: the 4 newest frames plus 4 drawn uniformly from the rest of the buffer |
| Replay buffer | FIFO, 512 frames (about 17 s). Stores the crop, the self-supervision targets, and the frozen model's outputs |
| Anchoring | (i) output anchor on every batch, against the frozen model's cached outputs: geodesic distance on the 22 body rotations, plus L1 on the camera parameters (tx, ty, s) and on shape; weight λ_o. (ii) weight anchor λ_w·Σ‖BA‖²_F, the distance from the initial weights. λ_o and λ_w are tuned on validation, the same values for (b) and (c) |
| Inference | LoRA merged into the weights after each update (ΔW = BA), so the per-frame student latency equals the base model's (to be verified, §4) |

**"Same compute"** means the same number of updates, optimizer steps, batch size and frames for (b)
and (c). Wall-clock time differs, because (c) renders. It is measured and reported (§4).

**Compute control:** if (c) wins, (b) is re-run with as many optimizer steps per update as fit
into (c)'s measured time per update. This tests whether (c) wins only because it costs more.

## 3. Losses

### (b) 2D keypoints

- **Detector.** DWPose as used by EHM-Tracker: YOLOX-L person detector plus `dw-ll_ucoco_384`,
  133 COCO-WholeBody keypoints, ONNX. onnxruntime 1.18 with the CUDA provider is installed in the
  guava environment (*checked*). Keypoints come from the full frame for the track's detector box
  and are mapped into the crop with the crop affine.
- **Correspondence.** EHM-Tracker's own `smplx_joints_to_dwpose` mapping
  (`EHM-Tracker/src/modules/refiner/smplx_utils.py:225`; PEAR's copy has the same index arrays).
  It covers 134 points:
  - body: 18, OpenPose order, with a neck;
  - feet: 2 × 3;
  - face: 17 contour points plus 51 others;
  - hands: 2 × 21.

  The per-keypoint weights start from EHM-Tracker's own fitting weights in that function. Group
  weights are tuned on validation, including the hips, since COCO hip keypoints are not SMPL-X hip
  joints.
- **Loss.** Σ_j c_j · ρ(‖π(J_j) − k_j‖ / 256), where:
  - c_j is the DWPose confidence, set to 0 below 0.3;
  - ρ is Geman–McClure with σ = 0.05 crop widths (tuned on validation);
  - π is the student's own crop camera (`pd_cam`, focal 24 on the normalised crop), the same
    projection as the Task 1 2D loss, with the same [−1, 2] clamp.
- **Gradient path.** loss → projected joints → PEAR `EHM_v2` (rotation-matrix input,
  differentiable) → student outputs (rotations, shape, camera) → decoder → LoRA.

### (c) AvatarLoop

**Rendering.**
- The user's GUAVA avatar is driven by the student's predicted body and hand rotations, shape and
  expression. Identity coefficients (`joints_offset`, FLAME shape) come from enrollment.
- The camera is the student's `pd_cam`. In GUAVA's convention this becomes an identity view with
  the translation (−tx, −ty, tz) added to the Gaussian means, because the rasteriser returns no
  gradient for the view matrix; tan(fov/2) = 1/24, 256 px.
- Alignment was checked: at the predicted pose, with no fitting, the silhouette soft-IoU with the
  matte is 0.70 (*measured*, §6).

**Is the render differentiable with respect to pose? Yes, measured.**
- Gradient path: loss → `diff-gaussian-rasterization-32` (means, rotations, scales) → GUAVA LBS
  (`ehm`, axis-angle) → `roma.rotmat_to_rotvec` → student rotation matrices → decoder → LoRA.
- The silhouette loss from the student's output gives a finite, non-zero gradient on the decoder
  weights on 8 of 8 frames (mean norm 5.31, §6).
- Planned change: a rotation-matrix input path in GUAVA's `ehm` (like PEAR's
  `EHM_v2(pose_type="rotmat")`), so the gradient does not go through rotmat→axis-angle, which is
  singular near π. This changes how GUAVA receives its input, not the student architecture.

**Legs: silhouette and part labels.** GUAVA's leg Gaussians are untrained. In the feasibility
renders they appear as sparse, semi-transparent speckle (§6), so the leg silhouette is **not**
taken from GUAVA's opacities.
- **Silhouette.** It is rendered from the posed body as a *solid* shape: the same Gaussians with
  opacity 1 and scale ≥ 1 cm. The fallback is a soft mesh rasteriser (PyTorch3D, GPU build
  available) on the posed SMPL-X mesh. Loss: 1 − soft-IoU with the person mask, plus L1 on the
  mask boundary band.
- **Part labels** (if a parser is approved, below). Per-Gaussian one-hot part channels, taken from
  the SMPL-X vertex each Gaussian is bound to, are rendered in the same pass; the rasteriser has
  32 channels, enough for ≤ 31 parts plus alpha. Loss: cross-entropy against the parser's part
  map, restricted to the lower body. Left/right part labels are the only part of (c) that
  directly constrains leg swaps, because a silhouette is identical under a left/right leg swap.

**Person mask.**
- StyleMatte (EHM-Tracker's matting, `stylematte_synth.pt`) gives the alpha. Its backbone comes from
  `facebook/mask2former-swin-tiny-coco-instance`, which it downloads on first use to
  `assets/StyleMatte/` (182 MB, untracked, **not to be committed**).
- In two-person frames, the full Mask2Former Swin-T COCO-instance model (same downloaded weights)
  gives instances. The one with the highest IoU with the track's detector box is selected and
  intersected with the matte.
- Alternative matte: BiRefNet-general, on disk at
  `/raid/ubx858/LHM-plusplus_data/pretrained_models/BiRefNet-general-epoch_244.pth`.

**Part parser: none on disk.** Candidates:
- Sapiens-Seg 0.3B: 28 classes, left/right limbs.
- SCHP on LIP: 20 classes, with left/right arm, leg and shoe.

The licences of both must be checked. **Downloading either needs your approval.** Without a parser,
(c) still runs with the silhouette and feature terms, but criterion B for leg swaps (§5) is then
not expected to be met. This is stated in advance.

**Upper body, hands and face: feature loss.**
- **Render.** The avatar's RGB is GUAVA's coarse RGB (the first 3 of the 32 rasterised channels)
  or the StyleUNet-refined image. The refiner uses `randomize_noise=False`, because the default
  `True` makes the loss stochastic. The choice between coarse and refined is made on validation by
  cost and benefit.
- **Comparison.** Frozen DINOv2 ViT-B/14 patch features (weights cached locally,
  `dinov2_vitb14_pretrain.pth`) are compared by cosine distance inside the person mask, on three
  crops: upper body, face (around the projected head), and hands (around the projected wrists).
- **Fallback.** LPIPS-VGG (GUAVA's `submodules/lpipsPyTorch`). Its linear weights download from
  GitHub on first use.
- **Gating.** Applied only when the frozen model's predicted facing is within 90° of the
  enrollment view, since the back is hallucinated. Face and hand terms also require the projected
  part to be at least 24 px.

**Discrete hypothesis tests (symmetric for (b) and (c)).** Gradient losses cannot cross a 180°
basin, so front/back flips and leg swaps need a discrete check. For each buffer frame two
alternatives are scored:
- the predicted root yawed by 180° about the body's vertical axis;
- the left and right leg rotations exchanged.

Each method uses its own score: keypoint reprojection error for (b), feature and part-label loss
for (c). If an alternative beats the prediction by a margin m (tuned on validation), its rotations
become a pseudo-target (geodesic loss) for that frame. Both methods get exactly the same mechanism,
so any difference comes from the signal.

**What (c) does not use.** No keypoint detector anywhere: no DWPose term, and no DWPose
gating. It shares only the person detector box with (a) and (b).

## 4. Metrics

All metrics are computed on the scored frames (after the 5 s window). Rows: (a), (b), (c), and PEAR
(released ViT-H, no adaptation) as a reference row on the same frames.

| Metric | Definition |
|---|---|
| MPJPE, PA-MPJPE, PVE (mm) | `frame_metrics` from `tools/eval_3dpw_standard.py`, unchanged: H36M regressor → J14, hip-midpoint centring, similarity Procrustes, SMPL-topology PVE via the fixed SMPLX→SMPL correspondence |
| Per part (mm) | the evaluator's `diag_lower_body`, `diag_upper_body` and `diag_hands` (SMPL kinematic joints, non-standard, labelled as such) |
| Error vs time | per scored frame, the time since the end of enrollment; MPJPE / PA-MPJPE in bins 0–5, 5–10, 10–20, 20–40, > 40 s; per-track running-mean curves; slope of a fit of error against time |
| Leg-swap rate | per frame, pelvis-centred camera-frame SMPL kinematic joints (neutral regressor, as `diag_*`). Swap if exchanging the predicted left and right knee and ankle lowers their summed error vs GT by ≥ 50 mm. Per 1,000 frames; sensitivity at 25 and 100 mm |
| Front/back-flip rate | facing vector f = normalise((L_hip − R_hip) × (neck − pelvis)) for prediction and GT in the camera frame, without rotation alignment. Flip if the angle is > 120°. Per 1,000 frames; sensitivity at 90° and 150° |
| Adaptation cost | GPU ms per update (data already on the GPU: forward, loss, backward and optimizer step, CUDA-synchronised; median and p90); amortised ms per frame (÷ k); per-frame student latency, adapted vs base; peak memory. Per-frame precomputation (DWPose, matting, instance masks) and enrollment time are reported separately, because a live system pays them too |

**Uncertainty.** Paired track-level bootstrap (1,000 resamples over tracks, the same method as the
evaluator's `bootstrap_ci`) for every difference between methods. Per-seed values are given next
to the 3-seed mean.

**Timing caveat.** All 8 GPUs here are shared: our training runs use GPUs 0 and 1, and another user
runs jobs on GPUs 2–7. Timings record the GPU, its load at run time (from nvidia-smi), and the
median. The feasibility check already shows a spread: 59.2 to 92.7 ms per step for the same work,
depending on the other user's load (§6).

## 5. Pre-registered success criterion

Fixed now, before any (b) or (c) result exists.

**Primary metric.** MPJPE (mm): mean over all scored frames of the 3DPW **test** split, detector
crop, final model, mean over 3 seeds. MPJPE is chosen over PA-MPJPE because it keeps global
orientation errors, including front/back flips, which Procrustes alignment would remove.

Define Δ_X = MPJPE(a) − MPJPE(X), the improvement of method X over no adaptation.

**AvatarLoop wins if (A) or (B) holds.**

- **(A) Accuracy.** Δ_c ≥ 1.5 · Δ_b, **and** Δ_c > 0 with its paired 95% CI excluding 0. If
  Δ_b ≤ 0, (A) reduces to Δ_c > 0 with its CI excluding 0.
- **(B) Failure modes.** For the leg-swap rate or for the front/back-flip rate:
  - (c) lowers the rate vs (a) by ≥ 25% (relative), with its paired 95% CI excluding 0;
  - (b) does not do the same on that rate: its relative reduction is < 10%, or its CI includes 0;
  - guard: (c)'s PA-MPJPE is not worse than (a)'s by more than 1.0 mm.

  (B) is assessable for a given rate only if (a)'s rate is ≥ 5 per 1,000 scored frames. This is
  decided from (a) on **validation** before any test run, and recorded.

Otherwise AvatarLoop does not win on 3DPW.

**Discipline.**
- (b) and (c) get the same tuning budget on validation: at most 8 loss-specific configurations
  each, with all shared adapter settings identical.
- Every hyperparameter is frozen before the test run, and the test split is run once per seed.
- Secondary metrics (PA-MPJPE, PVE, per part, time curves, GT-keypoint crop, cost) are reported
  but do not decide the outcome.

## 6. Risks and unknowns

Ranked from most to least risky:

1. **Avatar-render losses may not point toward the correct pose (riskiest; checked below).** A
   render loss only helps if the true pose scores better than wrong ones. Possible failures:
   - wrong avatar shape or clothing;
   - matte errors;
   - GUAVA's untrained legs;
   - the depth/scale ambiguity of a silhouette.
2. **Leg swaps are invisible to silhouettes.** A silhouette is identical under a left/right leg
   swap. Fixing swaps in (c) needs left/right part labels (parser not on disk; licence check and
   approval needed). The labels themselves confuse left and right in back views.
3. **Avatar quality on 3DPW.**
   - EHM-Tracker may fail on small or distant faces; the per-track enrollment success rate is
     reported at Checkpoint A.
   - GUAVA uses a single image and hallucinates the back.
   - There is no turnaround (§1).
4. **Masks.** Held objects (bench, bicycle, bags), occlusion, and the second person in 13 test
   sequences.
5. **Adapter drift on long tracks** (up to 73 s). Mitigated by the anchors and the replay buffer.
   Error vs time (§4) would show drift.
6. **Statistical power.** 37 test tracks; per-track bootstrap CIs may be wide for a few-mm
   effect.
7. **Cost.** (c) renders B frames per update; the rasteriser loops over the batch. (c) will be
   several times slower than (b); this is reported, not hidden.
8. **Engineering.**
   - The GUAVA rasteriser (`/tmp/guava-cuda-torch22`) and the GPU build of PyTorch3D
     (`/tmp/pytorch3d-torch22`) exist only under `/tmp`. They need a reproducible install with
     recorded hashes; the pylib copy of PyTorch3D is CPU-only.
   - Three assets EHM-Tracker needs are missing from `EHM-Tracker/assets` but exist in PEAR's
     assets. They are to be linked, not committed.
   - The live script runs the student under `inference_mode` and discards `pd_cam`; online
     adaptation needs both changed.
   - GUAVA's `ehm` accepts axis-angle only (§3).
   - The refiner adds noise by default.
   - The rasteriser's backward pass is not deterministic (atomic adds); see the run-to-run spread
     below.

### Feasibility check for risk 1 (read-only, done)

Script: `phase1/task3/feasibility_guava_gradient.py`. It writes only to its `--out` directory.
Command (from the repo root; `--sil-mode solid` for the variant):

```
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib \
CUDA_VISIBLE_DEVICES=<free GPU> python phase1/task3/feasibility_guava_gradient.py \
  --out /raid/ubx858/outputs/phase1_task3/<dir> --device cuda:0 [--sil-mode solid]
```

Commits: guava `2a104140` plus the commit that adds this script and DESIGN.md; PEAR `4e58a33`. Seed: `torch.manual_seed(0)`.
The script needs no training data.

**Setup.**
- Data: 3DPW validation `courtyard_jumpBench_01` (single person), 8 frames evenly spaced from 10%
  of the sequence to its end, GT-keypoint crop.
- Student: (c-vitpose) `best.pt`.
- Avatar: an existing GUAVA example person (`assets/example/tracked_image/vicox_14_right`), **not**
  the 3DPW subject (no 3DPW enrollment exists yet). Its identity and clothing are therefore wrong
  by construction; the student's predicted shape is used.
- Mask: StyleMatte.
- Pose fitting: from the student's prediction, axis-angle offsets on the root and 21 body joints
  plus the camera translation; Adam, 60 steps, silhouette L1 loss, anchor λ·‖offsets‖².
- Reference: the GT body pose (3DPW SMPL joints 1–21 → SMPL-X body joints 1–21, root rotated into
  the camera frame), with the student's shape and hands, and only the camera translation fitted.
- Run on GPU 6 (shared). Output `/raid/ubx858/outputs/phase1_task3/feasibility_v2/`.

**Results** (*measured*; means over the 8 frames; errors in mm from the standard evaluator's
`frame_metrics`):

| Silhouette from GUAVA's own Gaussians | Soft IoU | MPJPE | PA-MPJPE | PVE | Lower body | Upper body |
|---|---|---|---|---|---|---|
| Student prediction = (a) | 0.704 | 69.2 | 44.5 | 85.0 | 64.9 | 53.4 |
| Fitted, λ = 0.001 | 0.784 | 137.9 | 93.3 | 160.2 | 207.7 | 84.8 |
| Fitted, λ = 0.1 | 0.739 | 72.2 | 47.9 | 88.8 | 70.4 | 54.9 |
| Fitted, λ = 1 | 0.728 | 69.5 | 44.9 | 85.3 | 65.8 | 53.4 |
| GT pose (camera fitted) | **0.678** | 29.4 | 24.6 | 38.3 | 10.6 | 23.3 |

**What it shows.**
- **The gradient reaches the student.** The silhouette loss from the student's output gives
  finite, non-zero gradients on the decoder weights on 8 of 8 frames (mean norm 5.31).
- **Speed.** The avatar builds in 1.04–1.21 s (three runs). It has 203,911 Gaussians. One
  render-plus-backward step at 256 px for one frame took 59.2 ms (GPU 5, first run), 91.4 ms
  (GPU 6, this run) and 92.7 ms (GPU 5, solid variant). All GPUs were shared, so the spread
  reflects the other user's load.
- **The loss is minimised at the wrong pose.** With this avatar and matte, the silhouette scores
  the true pose **lower** than the student's wrong pose: IoU 0.678 vs 0.704, and lower in 6 of 8
  frames. Fitting therefore improves IoU while making every 3D metric worse:
  - weak anchor: MPJPE 69.2 → 137.9, lower body 64.9 → 207.7;
  - λ = 1: IoU still rises (0.704 → 0.728), and the errors stay within 0.9 mm of baseline.

  No anchor setting improved any 3D metric.
- **Run-to-run spread.** The first run of the same configuration (λ = 0.001,
  `/raid/ubx858/outputs/phase1_task3/feasibility/`) gave MPJPE 136.8 and PA-MPJPE 90.7, vs 137.9 and
  93.3 here. That is the rasteriser's non-determinism, amplified by a nearly unconstrained fit.
- **Overlays** (`frame_*.jpg`: image | matte | fitted silhouettes | GT-pose silhouette): GUAVA's
  legs render as sparse, semi-transparent speckle. The upper body is solid. A silhouette loss on
  speckled legs rewards spreading and moving them, not posing them correctly.

**Follow-up variant: solid silhouette.** Same frames and settings, but every Gaussian is made
opaque with scale ≥ 1 cm, which approximates the posed body's solid silhouette (`--sil-mode
solid`). It tests whether the speckle is the cause. Run on GPU 5 (shared); output
`/raid/ubx858/outputs/phase1_task3/feasibility_v3_solid/`.

| Solid silhouette | Soft IoU | MPJPE | PA-MPJPE | PVE | Lower body | Upper body |
|---|---|---|---|---|---|---|
| Student prediction = (a) | 0.783 | 69.2 | 44.5 | 85.0 | 64.9 | 53.4 |
| Fitted, λ = 0.001 | 0.882 | 93.9 | 61.8 | 118.0 | 81.1 | 70.0 |
| Fitted, λ = 0.1 | 0.847 | 72.4 | 47.1 | 89.3 | 65.7 | 55.8 |
| Fitted, λ = 1 | 0.828 | 69.8 | 45.0 | 85.6 | 65.8 | 53.8 |
| GT pose (camera fitted) | 0.790 | 29.4 | 24.6 | 38.3 | 10.6 | 23.3 |

(*measured*; head-gradient norm 5.78, finite on 8 of 8 frames; 92.7 ms per step on the shared GPU)

- **The speckle was a large part of the failure.** With a solid silhouette the collapse is much
  smaller (weak anchor: MPJPE 93.9 instead of 137.9, lower body 81.1 instead of 207.7). The GT pose
  now scores slightly **higher** than the prediction: IoU 0.790 vs 0.783, and higher in 5 of 8
  frames.
- **But the silhouette barely tells right from wrong.** The GT pose has 40 mm less MPJPE, yet scores
  only 0.007 higher in IoU. Fitted wrong poses reach 0.83–0.88, above the GT pose. With this
  borrowed avatar, **no anchor setting improved any 3D metric**: the best (λ = 1) is
  0.4–0.9 mm worse than baseline on every metric.
- **Remaining confounds.** The avatar's shape and clothing are not the subject's (the silhouettes
  are visibly wider at the torso). The matte cuts the feet at the crop edge. The feature loss was
  not tested here.

**Consequences for the design** (already reflected in §2–§3):
- the leg silhouette comes from the posed body, not from GUAVA's opacities;
- strong output and weight anchors;
- amortised LoRA updates over a replay buffer instead of per-frame pose optimisation;
- appearance features only where the avatar is reliable (upper body, gated by view);
- discrete hypothesis tests for flips and swaps.

**Verdict on risk 1: unresolved, and the main threat to (c).** Silhouette supervision with a
mismatched avatar gives a gradient but no useful direction. Whether the **subject's own** avatar
(which the protocol in §1 provides) together with the feature loss gives a direction is still
unknown. The plan therefore starts with a go/no-go gate on exactly that (§7, step A0) before the
harness is built.

## 7. Implementation plan

Time estimates are working days and GPU time; all are **estimates**. Any GPU job longer than
15 minutes is proposed with its exact command, expected duration and memory, and waits for
approval. Runs use GPUs not used by the Task 1 training jobs.

| Checkpoint | Work | Estimate |
|---|---|---|
| **A0 — go/no-go on risk 1** | Enroll the **subject's own** avatar (EHM-Tracker + GUAVA, first 5 s) on 3 single-person validation tracks. Repeat the feasibility check with the solid silhouette **and** the DINOv2 feature loss. Pass if, on most frames, the GT pose scores better than the student's prediction under the combined loss, **and** anchored fitting lowers lower-body or MPJPE error on average. **Stop and report either way.** If it fails, the harness is not built until we decide. | 1 day; GPU under 15 min |
| **A — data and enrollment** | Reproducible install of the GUAVA rasteriser and GPU PyTorch3D (hashes recorded); EHM-Tracker assets linked; GUAVA `ehm` rotation-matrix input; refiner noise off. YOLOX detections for validation (test detections exist). Per-frame DWPose keypoints, StyleMatte mattes and instance masks for both splits. EHM-Tracker + GUAVA enrollment for 16 + 37 tracks, with a QC sheet (source frame next to avatar render, per track). (a) on the scored frames reproduces the standard evaluator per frame, exactly. (a)'s swap and flip base rates on validation, which decide whether criterion B is assessable. **Stop and report.** | 2 days; GPU about 1–2 h in total for preprocessing (to be measured on a 500-frame slice first) |
| **B — harness and (b)** | Causal loop, LoRA, buffer, anchors, and the metrics module (swap, flip, time bins, cost). Unit tests: (1) zero-init LoRA reproduces (a) bit-for-bit; (2) causality: changing frames > t leaves the prediction at t unchanged; (3) the metric definitions on synthetic swaps and flips. (b) on validation, with tuning (≤ 8 configurations). **Stop and report.** | 2 days; GPU about 10 min per validation run of (b) |
| **C — (c)** | Solid/mesh silhouette, part labels (if a parser is approved), the DINOv2 feature loss with gating, and the hypothesis tests. (c) on validation with the same tuning budget. Cost measurement. Hyperparameters frozen; the frozen config is committed. **Stop and report, with the final-run commands for approval.** | 3 days; GPU about 30–60 min per validation run of (c), mainly rendering |
| **D — final** | Winning long-run checkpoint: re-check on validation, then 3DPW test for (a), (b), (c) × 3 seeds (plus the compute control if (c) wins). Verdict against §5, written to `phase1/task3/REPORT.md`. | 1 day; GPU about 1.5 h per (c) seed and about 10 min per (b) seed on test. Seeds can run in parallel on free GPUs |

Every run logs its config, git commits of both repositories, seed, data split, and checkpoint path
and SHA-256, as in Task 1.

**Open decisions for you** (they do not block A0 or A):
1. **Part parser.** Approve downloading Sapiens-Seg 0.3B or SCHP-LIP, after a licence check.
   Without one, (c) has no direct leg-swap signal.
2. **An optional (b + c) arm.** It would answer "does the avatar add anything **on top of**
   keypoints?". The pre-registered criterion compares (c) with (b) only, as specified, so this arm
   would be reported outside the criterion.
