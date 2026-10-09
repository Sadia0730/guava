# Phase 1 / Task 1: BEDLAM2 ground-truth supervision for the AvatarLoop student

Status legend used throughout: **measured** = read from a file or produced by code on
this server; **computed** = derived arithmetically from code and saved arguments, not
observed during a run; **reported** = a number given to me that I could not reproduce.

---

## Checkpoint 1: current setup (read only)

Snapshot: guava `avatar_loop` @ `c0ffe55`, PEAR submodule `avatar_loop` @ `c00a32d`.
Student checkpoint under study: `/raid/ubx858/outputs/pear_student_v2_phase2/checkpoints/step_0235000.pt`
(SHA-256 `b62f4027…c061c495`). Teacher: PEAR `pear_model.pt` (SHA-256 `be82dfa0…abeb18`).
No code was changed for this checkpoint.

### 1. Training pipeline

**Data preparation** (`third_party/PEAR/tools/prepare_student_distill_data.py`)

| Item | Value |
|---|---|
| Sources | BEDLAM2 MP4s, **single-person render jobs only** (`*_1_*`); UBody videos |
| Person box | YOLOX-L (`EHM-Tracker/pretrained/dwpose/yolox_l.onnx`) every 5th frame, largest box first, then highest-IoU tracking; boxes linearly interpolated for the frames in between |
| Crop | Square, side = 1.25 × max(box w, box h), centred on the box, `cv2.warpAffine` to 256×256 (out-of-image area black), JPEG q95 |
| Crop box saved? | **No** — only the 256×256 JPEGs and `_sequence.json` (frame count, fps, frame step) |
| Split | BEDLAM2: SHA-1 hash of the render-job name < 0.05 → val (whole render jobs held out). UBody: official inter/intra-scene test lists → val |
| Manifest | `train.jsonl`: 21,512 sequences (18,429 BEDLAM2, 3,083 UBody); frames: BEDLAM2 5,872,942, UBody 504,402 (measured) |

**Loader** (`third_party/PEAR/dataset/student_distill_dataset.py`)

- Clips of `clip_length=2` frames, `temporal_stride=8` apart, starting every `clip_step=8` frames.
- `balance_sources=True`: every batch alternates BEDLAM2 / UBody, so the epoch length is
  2 × (larger source's clip count). UBody clips are therefore repeated ~11× per epoch.
  Train clips per epoch: 1,446,920 (measured, run logs).
- **Only augmentation: horizontal flip, p = 0.5**, applied to the whole clip. The teacher
  sees the same flipped image, so labels stay consistent. No scale, translation, rotation,
  colour, blur, or occlusion augmentation.

**Model** — matches your description (verified in `student_backbone.py`, `smplx_head.py`):
ImageNet normalisation, centre crop 256→192 columns, stem stride 2 (96×128×96), four
depthwise stages, 1×1 projection to 512, learned 16×12 positional embedding, 4 transformer
blocks (8 heads), 1×1 expansion to 1280, GroupNorm. Decoder: one zero query, 6 cross-attention
layers at width 1024, linear heads for SMPL-X pose (6D rotations, initialised from the SMPL mean
pose), 200-D body shape, 50-D SMPL-X expression, hand/head scale, FLAME pose/eyes/eyelids/jaw,
300-D FLAME shape, 50-D FLAME expression, and camera.
Camera: `[tx, ty, s] + [0, 0, 1.5]`, `tz = 24 / s`, fixed rotation `diag(-1,-1,1)`, focal 24,
projection to normalised `[0,1]²` over the full **256×256 square** (not the 192-wide centre).
Trainable parameters: **65.0 M** total (backbone 24.5 M, measured in run logs). The comment in
`configs/student_l70_v2.yaml` ("~34M params") is out of date.

**Losses** (`distillation_loss`; every target is the PEAR teacher output on the same image)

| Term | Definition | Phase 1 weight | Phase 2 weight |
|---|---|---|---|
| camera | L1 on teacher `pd_cam` translation (3 values) | 1.0 | 1.0 |
| body | Σ wᵢ·L1 on rotation matrices / params: global 5, body 10, hands 1 each, hand/head scale 0.5, SMPL-X exp 0.5, shape 0.1 | 1.0 | 1.0 |
| flame | Σ wᵢ·L1: eye 0.5, head pose 1, jaw 1, eyelid 0.5, expression 0.5, shape 0.1 | 0.5 | 2.0 |
| rotation | Geodesic angle on global, body, both hands | 1.0 | 1.0 |
| joint3d | L1 on 22 SMPL-X body joints in body-model space (no camera translation, no root centring) | 1.0 | 1.0 |
| joint2d | **Old, camera-insensitive version** (joints divided by their own depth; see issue T1) | 1.0 | 1.0 |
| feature | 1 − cosine between student 1280×16×12 map and teacher backbone tokens | 2.0 | 1.0 |
| velocity | L1 on frame-to-frame parameter differences | 0.0 | 0.1 |
| per-part delta, motion sampling | Not present in the code at the time (absent from saved args) | — | — |

The 3D/2D joint terms use `SMPLXV2` with `SMPLX_NEUTRAL_2020` (body model only, no FLAME head).

**Optimiser and schedule** (from arguments saved in the checkpoints)

| | Phase 1 (`pear_student_v2_phase1`) | Phase 2 (`pear_student_v2_phase2`, gives 235k) |
|---|---|---|
| Init | Random backbone; **head copied from PEAR and frozen** | Resume phase-1 step 150,000; head unfrozen |
| Steps | 0 → 150,000 | 150,000 → 250,000 (235,000 selected) |
| Batch | 8 clips × 2 frames = 16 images | 24 clips × 2 frames = 48 images |
| Optimiser | AdamW, wd 0.05 on all parameters, grad-norm clip 1.0, bf16 autocast | same; **optimiser state reset** at resume (logged) |
| LR | 3e-4, warm-up 2,000, cosine to 5% at step 150,000 | nominal 5e-5, warm-up 500, cosine to 5% at step 250,000 |
| Effective LR (computed) | 3e-4 → 1.5e-5 | the cosine runs over absolute steps 0–250k, so phase 2 starts at **1.90e-5**, reaches 2.9e-6 at 235k; the nominal 5e-5 and the warm-up are never applied |
| Epochs (computed from logged steps/epoch) | 0.83 | 1.41 up to step 235k |
| Images seen (computed) | 2.40 M | 4.08 M up to step 235k |

Checkpoint 235000 was chosen by a teacher-agreement sweep over steps 155k–250k
(`outputs/domain_investigation/v2_checkpoint_selection.md`), not by any ground-truth metric.
A later run (`pear_student_v2_temporal_refine`, 235k → 285k, motion sampling + delta losses)
exists but is not the model behind the pilot numbers.

### 2. Evaluation

**Your pilot numbers could not be reproduced on this server, and three different evaluators exist.**

| Evaluator | Joints compared | Root | PA set | Crop | Frames | Teacher MPJPE / PA | Student MPJPE / PA |
|---|---|---|---|---|---|---|---|
| `tools/eval_pear_3dpw.py` (laptop commit; source of your numbers) | Pred: SMPL-X mesh → SMPL mesh (`SMPLX2SMPL`) → **neutral SMPL J-regressor, 24 joints incl. hands/feet**. GT: 3DPW `jointPositions` (24 SMPL joints) | joint 0 (pelvis) | all 24 | GT OpenPose-18 box, 0.75 aspect, black side padding | 111 = 3 per person-track × 37 tracks | **62.6 / 45.1 (reported)** | **101.5 / 68.8 (reported)** |
| `tools/evaluate_pear_3dpw.py` (server reconstruction of the pilot) | Pred: **SMPL-X kinematic joints 0–21 directly** vs GT SMPL kinematic joints | joint 0 | J14 subset of SMPL kinematic joints | GT OpenPose-18 box, square, scale 1.25 | 111 | 95.9 / 67.3 (measured) | 137.2 / 93.2 (measured) |
| `tools/evaluate_pear_3dpw_stream.py` (server full split) | same as above | joint 0 | same | same | 35,515 (full test) | 97.2 / 69.7 (measured) | 138.1 / 95.5 (measured) |
| PEAR paper, Table 3 | not published as code | — | — | — | — | 71.3 / 45.3 (paper) | — |

Sources of measured rows: `outputs/pear_3dpw_235000/annotated/summary.json`,
`outputs/pear_3dpw_full_protocol/step_0235000/summary.json` (`train_test_mode`, J14).
The laptop's output files are not on this server; that row is as reported by you. That it
came from `eval_pear_3dpw.py` is inferred: its default of 3 samples per person-track over the
37 test tracks gives exactly 111 frames.

Common to all three: millimetres, test split, both people in two-person sequences, GT joints
rotated into the camera frame with `cam_poses`, PEAR camera axes converted to OpenCV.
None computes PVE, hand, or face error. The 3DPW annotation files label all 37 test tracks
as male (`genders == 'm'`).

**Standard 3DPW protocol** (SPIN, VIBE, HMR2.0, CameraHMR; SMPL-X methods such as OSX and
SMPLer-X convert to SMPL first): 35,515 test person-frames with valid camera pose; ground
truth = gendered SMPL mesh from `poses`/`betas`; prediction = SMPL mesh (SMPL-X converted
with the SMPL-X→SMPL vertex mapping); **14 LSP joints via the Human3.6M joint regressor**;
MPJPE after aligning the **hip midpoint**; PA-MPJPE after per-frame Procrustes on the 14 joints;
PVE on the 6,890 SMPL vertices after pelvis alignment.

**Explicitly: no evaluator in this repository matches the standard 3DPW test protocol.**
- The server evaluators compare SMPL-X kinematic joints against SMPL kinematic joints.
  The two skeletons place hips, spine, and neck differently, so a perfect pose still has
  non-zero error. This plausibly explains PEAR's 69.7 mm PA-MPJPE here versus 45.3 in its paper.
- The laptop evaluator uses the right mesh conversion but the wrong joint set (24 kinematic
  joints, neutral regressor, pelvis-joint root) and a 111-frame sample.
- The frame count of the server stream evaluator (35,515) is the standard one.

### 3. Wrong, fragile, or undocumented

Training

- **T1. The 2D joint loss used for checkpoint 235000 never responded to the camera.**
  Fixed in code on 2026-10-08 (`project_joints`), but no model has been trained with the fix.
  Its target is still the teacher, and joints outside the crop are not masked.
- **T2. Phase 2 never used its stated learning rate.** Because the cosine is computed over
  absolute steps, phase 2 ran at 1.9e-5 → 2.9e-6 instead of 5e-5, with no warm-up, after an
  optimiser reset. "Retrain (a) with the same schedule" has to reproduce this two-phase
  behaviour deliberately, or we change the schedule for all three variants.
- **T3. Not reproducible.** No global seed (`torch.manual_seed` absent; only the DataLoader
  generator is seeded with 42); no git commit, data hash, or code diff is saved with runs.
- **T4. Model selection used teacher agreement only.** No ground-truth metric was used to
  pick 235k.
- **T5. Weak augmentation.** Flip only. No scale, translation, rotation, or occlusion
  augmentation, which matters for lower-body truncation.
- **T6. Crop boxes were never stored**, so ground truth cannot currently be mapped into the
  training crops without recomputing them.
- **T7. Split is not motion-independent.** BEDLAM2 bodies and motions recur across render
  jobs, so held-out render jobs share motions with training. BEDLAM2's official `testset`
  flags are ignored (relevant only if we evaluate on BEDLAM2).
- **T8. Teacher weights load with `strict=False`** in training and in every evaluator; a
  partial load would be silent (mitigated only by the recorded teacher hash).
- **T9. Source balancing oversamples UBody about 11×.** Not documented in the run configs.
- **T10. Known latent bug:** `--motion_pool_factor > 1` together with `--teacher_cache` pairs
  cached teacher outputs with the wrong images (paths are not filtered with the batch). Not
  triggered by any existing run.
- **T11. Parameter-count comment is wrong** (65.0 M measured vs "~34M" in the config).

Evaluation

- **E1. Non-standard 3DPW protocol in every evaluator** (section 2). Any number so far is
  comparable only within one evaluator.
- **E2. The H36M joint regressor needed for the standard protocol is not on the server.**
  Gendered SMPL models are available (`/home/ubx858/ml-hugs/data/smpl/`), and the SMPL-X→SMPL
  mapping is in `third_party/PEAR/assets/SMPLX2SMPL.zip`.
- **E3. EHF is not on the server at all.**
- **E4. Three different crop policies.** Training: detector box, square, ×1.25.
  Evaluators: ground-truth OpenPose keypoint box (tighter, excludes the head top and feet).
  Live inference (`main/live_pear_guava.py`, `input_framing: centered`): **no detector**, the
  central square of the webcam frame resized to 256×256, so the person's size depends on
  distance from the camera.
- **E5. 3DPW has no finger or face ground truth.** "Hand" error on 3DPW can only mean wrist
  joints or the rigid SMPL hand vertices. Hand and face accuracy must come from EHF.
- **E6. No PVE, per-part, or hand/face metrics are implemented**, and no evaluator has tests
  that check it against a known reference value.

### Decisions needed before Checkpoint 2

1. **Canonical crop.** Checkpoint 3 asks for BEDLAM2 crops "exactly like our inference
   preprocessing", but live inference has no detector. Options: (i) detector box, square,
   ×1.25, matching the existing training data, and also used for evaluation; (ii) match the
   live centre-square framing everywhere. I recommend (i) for training and benchmarks, and
   treating live framing as a separate deployment issue.
2. **Standard evaluator.** May I fetch `J_regressor_h36m.npy` (public, from the SPIN/VIBE data
   release) and write a new evaluator implementing the standard protocol for all four rows?
   Per your rules I will not modify the existing evaluators; the new one would be the only
   one used for the comparison table.
3. **EHF** must be downloaded by you (SMPL-X website, licence acceptance).
4. **Variant (a) schedule.** Reproduce the two-phase recipe exactly (frozen teacher head,
   then unfrozen; effective LR as in T2), or give all three variants one clean single-phase
   schedule? I recommend one clean schedule for all three, with (a) as its distillation-only
   member. Copying PEAR's head into (b) would leak teacher knowledge into the "ground truth
   only" variant, so a fair comparison needs the same initialisation for all three.

### Decisions received (2026-10-08)

1. Crop: training = detector box, square, ×1.25, with ±15% scale and ±10% shift jitter.
   Benchmarks report (i) the standard ground-truth-keypoint crop as the main number and
   (ii) the detector crop as a secondary number; same crop for every model in a row.
2. A new standard 3DPW evaluator; existing evaluators untouched, their numbers kept for
   reference. PEAR must land within ~2 mm of its published 3DPW numbers before anything
   else is evaluated. Non-standard per-part errors go in a separately labelled table.
3. EHF: path to be supplied; continue without it.
4. One clean single-phase schedule (linear warm-up, cosine decay), same initialisation for
   (a), (b), (c), no PEAR head copied; camera-aware 2D loss in all; global seed and git
   commit/config recorded; checkpoint selection on 3DPW **validation** with the new
   evaluator; standard augmentations for all variants; short (~25%) screening runs first,
   then only the winner on the full schedule.

### Known issue for architecture v2 (out of scope for Task 1)

**Live framing.** `main/live_pear_guava.py` (`input_framing: centered`) feeds the central
square of the webcam frame with no person detector, so the person's scale and position in the
crop depend on their distance from the camera. Training and benchmarks use person-centred
crops. v2 should use tracked person crops at inference.

---

## Standard 3DPW evaluator (validated; gate passed)

`tools/eval_3dpw_standard.py`, tests in `tests/test_eval_3dpw_standard.py` (9/9 pass).
Existing evaluators were not modified.

**Protocol implemented** (identical for every model): all `campose_valid` person-frames of
the split (35,515 on test); ground truth = gendered SMPL mesh from `poses` and 10 `betas`,
rotated into the camera frame with `cam_poses`; prediction = EHM-s mesh (the mesh GUAVA
animates: SMPL-X body with the FLAME head) mapped to SMPL topology; 14 LSP joints from the
Human3.6M regressor on both meshes; MPJPE after hip-midpoint centring; PA-MPJPE after
per-frame similarity Procrustes on the 14 joints; PVE over 6,890 SMPL vertices after
hip-midpoint centring. Also logged: SPIN-style centring on the regressed H36M pelvis
(diagnostic), and the **non-standard** per-part table (lower body, upper body, hands) using
SMPL kinematic joints from the neutral SMPL regressor applied to both SMPL-topology meshes.
3DPW has no SMPL-X ground truth, so "SMPL-X joints" cannot be compared on 3DPW without a
skeleton mismatch; hand error on 3DPW covers only wrists and the rigid SMPL hand joints.

**Crops.** Main: PEAR's own box code on the ground-truth OpenPose keypoints
(`get_bbox` ×1.2, then `process_bbox` square ×1.25, 256×256), copied from
`third_party/PEAR/inference_images.py` and tested to match it. Secondary: YOLOX-L box
associated by highest IoU with the visible-keypoint box (fallback to the main crop if IoU < 0.1,
counted), square ×1.25 exactly as the training data were prepared.

**Assets** (all hashes are written into every `summary.json`):
- `J_regressor_h36m.npy` from SPIN's official `data.tar.gz` (SHA-256 `c655cd70…6444a0`,
  17×6,890); byte-identical to both copies bundled in `third_party/PEAR/assets/SMPLX2SMPL.zip`.
- SMPL male/female/neutral from `/home/ubx858/ml-hugs/data/smpl/` converted once from
  chumpy pickles to plain `.npz` (`/raid/ubx858/datasets/eval_assets/smpl/`, source and output
  hashes in `conversion_manifest.json`).
- Isolated Python packages in `/raid/ubx858/datasets/eval_assets/pylib` (the `guava` env is
  unchanged): `smplx` 0.1.28; PyTorch3D 0.7.6 built from the official tag (CPU extension
  only; PEAR's EHM-s code uses only its pure-PyTorch parts), `iopath`, `portalocker`.

**Validation so far (measured)**
- Unit tests: Procrustes recovers a known similarity transform and never reflects; identical
  meshes give 0 on every metric; a 30° rotation raises MPJPE but not PA-MPJPE; crop code
  matches PEAR's; training square box matches data preparation; the SMPL-X→SMPL matrix
  rows sum to 1. Translation invariance holds within the H36M regressor's own tolerance (its
  rows sum to 0.9996–1.0; 0.02 mm at a 0.3 m offset; both meshes are built at the origin).
- The PEAR teacher loads with **no missing or unexpected keys** (despite `strict=False`).
- 2D overlays (12 frames): the predicted mesh projects onto the correct person in
  multi-person frames.
- Smoke subsets (**not the gate**; PEAR, ground-truth-keypoint crop):

  | Subset | Frames | MPJPE | PA-MPJPE | PVE |
  |---|---|---|---|---|
  | every 100th test frame | 356 | 74.9 | 46.4 | 84.6 |
  | every 10th test frame | 3,552 | 73.5 | 45.6 | 82.1 |
  | PEAR paper, Table 3 | 35,515 | 71.3 | 45.3 | — |

**SMPL-X → SMPL conversion: method A approved (2026-10-09); B kept as a diagnostic only.** Measured on the 356 PEAR predictions:

| | A: fixed vertex correspondence | B: per-frame SMPL fit to A (600 Adam steps) |
|---|---|---|
| Method | `smplx2smpl.pkl`: each SMPL vertex is a barycentric blend of 1–3 SMPL-X vertices (rows sum to 1) | neutral SMPL pose, 10 betas, translation optimised to A's vertices |
| Cost | 0.13 ms/frame | 33 ms/frame (≈ 20 min extra per full test run) |
| MPJPE / PA-MPJPE / PVE | 74.93 / 46.42 / 84.58 | 74.35 / 46.23 / 83.92 |
| Effect | — | fit residual 5.1 mm (p95 6.9); J14 joints move 5.3 mm (p95 11.3) but metrics change < 1 mm |

Recommendation: **A**. It is deterministic and essentially free, and it changes the metrics by
under 1 mm relative to B. B's result also depends on optimiser settings and convergence.

**Other findings while building it**
- The released PEAR checkpoint is a **ViT-H** backbone (32 layers, width 1280) while the paper
  describes ViT-B. Our "PEAR" row may not be the exact model in Table 3.
- The `onnxruntime` CUDA provider cannot load in the `guava` env (needs CUDA 11 cuBLAS) and
  silently falls back to CPU. YOLOX therefore runs on CPU (105 ms/image single process); the
  training crops were almost certainly produced on CPU as well. The evaluator now requests the
  CPU provider explicitly and detects in 32 parallel processes (2,455 images in 56 s).

### Pre-gate checks (2026-10-09)

**Gender labels.** In every raw `sequenceFiles/*.pkl`, `genders` is a per-sequence Python
list with one NumPy string (`'m'` or `'f'`) per person-track; the evaluator maps it per
person-track. All 37 test tracks are `'m'`; validation has 9 `'f'`, train 16 `'f'`.
Rebuilding the shipped `jointPositions` from `poses`/`betas`/`trans` (20 frames per track):
the **male** model reproduces them with 0.00 mm error on **all 37 test tracks** (female
16–71 mm, neutral 25–60 mm); as a control, the female model reproduces all 9 female
validation tracks with 0.00 mm. The labels are correct and correctly parsed.
PEAR PVE against meshes built with each model (`downtown_arguing_00`, every 5th frame):
p0 male/female/neutral 80.7 / 91.0 / 87.2 mm; p1 79.9 / 78.1 / 88.2 mm. Prediction error is
not a reliable gender test (the prediction is gender-agnostic), but using the wrong model
would move MPJPE by up to 13.6 mm on p0.

**PEAR model identity.** The paper states a ViT-B backbone. The only released checkpoint
(`BestWJH/PEAR_models`, revision `513a74e7`, single file `pear_model.pt`, 2.69 GB) is ViT-H
(32 layers, width 1280), as is the released `configs/train.yaml`; no ViT-B config or
checkpoint exists in the repository or on Hugging Face. The model card says: *"This is the
initial version of our PEAR model, rather than the final version presented in our paper. While
it may slightly underperform the final model in certain complex poses ... The final version is
scheduled for release within the next few months."* Upstream (`Pixel-Talk/PEAR`) lists no
later model release and no 3DPW evaluation code. Our row is therefore labelled
**"PEAR (released ViT-H checkpoint)"**; it is not the Table 3 model.

**Parameter-count comment** in `third_party/PEAR/configs/student_l70_v2.yaml` fixed to
"65.0M parameters in total: backbone 24.5M, head 40.5M" (verified by instantiating the model).

### Gate (full test set, 35,515 person-frames, ground-truth-keypoint crop): PASSED

| | Ours (measured) | Paper Table 3 | Difference | Criterion |
|---|---|---|---|---|
| PA-MPJPE | 45.65 | 45.3 | +0.35 mm | within ~1 mm |
| MPJPE, hip-midpoint centring (main) | 73.56 | 71.3 | +2.26 mm | within ~3 mm |
| MPJPE, SPIN-style H36M-pelvis centring (diagnostic) | 73.44 | — | | |

### 3DPW test results (standard protocol, 35,515 person-frames)

Mean, with a 95% confidence interval from a 1,000-sample bootstrap over the 24 test
sequences. All in millimetres.

| Model | Crop | MPJPE | PA-MPJPE | PVE |
|---|---|---|---|---|
| PEAR (released ViT-H checkpoint) | GT keypoints (main) | **73.6** [69.5, 78.1] | **45.7** [43.1, 48.7] | **82.3** [77.9, 86.9] |
| PEAR (released ViT-H checkpoint) | Detector | 73.7 [69.7, 78.1] | 45.9 [43.2, 49.0] | 82.5 [78.1, 87.2] |
| Student, checkpoint 235000 | GT keypoints (main) | **113.0** [104.5, 122.3] | **71.6** [67.2, 76.3] | **129.1** [119.4, 139.8] |
| Student, checkpoint 235000 | Detector | 111.2 [103.3, 119.8] | 71.3 [66.9, 75.9] | 127.6 [118.4, 138.1] |

Detector-crop fallbacks: **41 of 35,515** person-frames (0.12%) for both models (no
detection with IoU ≥ 0.1; YOLOX found no person in 35 of 24,547 test images). These
frames use the ground-truth-keypoint crop instead.

**Non-standard diagnostic** (NOT comparable to published numbers): SMPL kinematic joints
from the neutral SMPL regressor applied to both SMPL-topology meshes, root (pelvis joint)
centred, no alignment. Lower body = hips, knees, ankles, feet; upper body = spine, neck,
head, collars, shoulders, elbows; hands = wrists and the two rigid SMPL hand joints
(3DPW has no finger ground truth).

| Model | Crop | Lower body | Upper body | Hands (wrist level) |
|---|---|---|---|---|
| PEAR (released ViT-H checkpoint) | GT keypoints | 96.9 [90.0, 104.0] | 49.2 [45.7, 53.1] | 104.5 [95.1, 116.0] |
| PEAR (released ViT-H checkpoint) | Detector | 96.8 [90.3, 103.9] | 49.3 [45.8, 53.1] | 105.1 [95.4, 117.0] |
| Student, checkpoint 235000 | GT keypoints | 133.5 [120.9, 147.1] | 74.1 [68.3, 80.8] | 187.3 [166.9, 210.7] |
| Student, checkpoint 235000 | Detector | 130.6 [119.5, 142.5] | 72.7 [67.4, 79.1] | 187.1 [166.8, 210.3] |

**Reading the tables**
- On the standard protocol the student trails PEAR by **39.4 mm MPJPE** and **26.0 mm
  PA-MPJPE** (main crop). Under the old laptop protocol the same models measured
  101.5 vs 62.6; those numbers are not comparable with these.
- The crop choice changes results by under 2 mm for both models; the student is slightly
  better on detector crops, which match its training crops.
- In the diagnostic table the student's largest relative gap is at the **wrists/hands**
  (187 vs 105 mm), followed by the lower body (134 vs 97 mm).
- 2D overlays (24 per run) show both models' meshes projected onto the correct person.

### Run log

| Run | Output directory (`/raid/ubx858/outputs/phase1_task1/eval_3dpw/…`) | Time | Peak GPU memory |
|---|---|---|---|
| PEAR, GT-keypoint crop | `pear/test_gt_keypoints` | 392 s | 5.21 GB |
| Student 235000, GT-keypoint crop | `student_235000/test_gt_keypoints` | 165 s | 6.18 GB |
| YOLOX detections, 24,547 test images | `detections/detections_test.json` | 419 s (CPU, 32 processes) | — |
| PEAR, detector crop | `pear/test_detector` | 393 s | 5.21 GB |
| Student 235000, detector crop | `student_235000/test_detector` | 135 s | 6.18 GB |

Each `summary.json` records the command, checkpoint SHA-256 (PEAR `be82dfa0…`, student
`b62f4027…`), asset hashes, evaluator SHA-256 (`d28f6efd…`), and git state: guava
`c0ffe55` and PEAR `c00a32d`, both with **uncommitted changes** (the new evaluator, tests and
the config comment are not committed yet). Per-frame results are in each run's `per_frame.csv`.
Seed: none needed (evaluation is deterministic apart from the bootstrap, which uses seed 0).

### Commits and clean reruns (2026-10-09)

| Repo | Commit | Content |
|---|---|---|
| guava `avatar_loop` | `f7281f25`, tag **`eval-3dpw-standard-v1`** | Commit A: `tools/eval_3dpw_standard.py`, `tests/test_eval_3dpw_standard.py`, and the method-B diagnostic `tools/analyze_smplx2smpl_conversion.py` |
| PEAR `avatar_loop` | `8fa18f4` | Commit B: parameter-count comment in `configs/student_l70_v2.yaml` |
| guava `avatar_loop` | `7dd0b5e4` | submodule pointer to `8fa18f4` |

Not pushed. No licensed asset is in either repo: SMPL `.npz`, the H36M regressor and the isolated
packages live under `/raid/ubx858/datasets/eval_assets/` and are referenced by path and
SHA-256; `.gitignore` already excludes `datasets/`, `assets/SMPLX/lockedhead/` and `third_party/`.

The two ground-truth-keypoint rows were rerun from the clean tree (`*_clean` directories,
`PYTHONDONTWRITEBYTECODE=1` so tracked `.pyc` files in PEAR stay clean). Their `summary.json`
records guava `7dd0b5e4` and PEAR `8fa18f4`, both `dirty: false`, evaluator SHA-256 `d28f6efd…`.
**All 35,515 per-frame values are bit-identical** to the uncommitted runs for both models
(maximum difference 0 mm): PEAR 73.56 / 45.65 / 82.28, student 112.98 / 71.61 / 129.08.

## Crop-visibility diagnostic (no training)

**How the two models crop.** Identical at inference: both take a 256×256 patch, normalise
it, and drop 32 columns on each side (`x[:, :, :, 32:-32]` in `Ehm_Pipeline.forward_features`
and `PearStudentPipeline.prepare_input`), so the backbone sees the central 256×192; both
cameras project onto the full 256×256 square. In this evaluation both receive the same
patch. Their *training* crops differ: PEAR's loader (`dataset/webdata_loader.py`) uses a square
box of side `scale.max()` around an annotated centre (how `scale` was computed is not
recoverable; its training annotations are not released); the student used the YOLOX box,
square, ×1.25. PEAR's own `inference_images.py` uses a YOLOv8x box squared ×1.25.

**Method** (`tools/diag_crop_visibility.py`, output `/raid/ubx858/outputs/phase1_task1/diag_crop_visibility/`):
GT SMPL joints (gendered, with translation) projected with 3DPW intrinsics/extrinsics into the
256×256 ground-truth-keypoint crop; each wrist/ankle classified as centre (x in [32, 224)), side
strip, or outside the crop. Error = per-joint error from the non-standard diagnostic (neutral SMPL
regressor on both meshes, pelvis-joint centred). Geometry check: projected joints lie a median
6.8 px (p90 20.0 px) from the annotated OpenPose keypoints over 125,857 visible wrists/ankles.

| Joints | Region | Instances | PEAR (mm) | Student (mm) | Gap (mm) |
|---|---|---|---|---|---|
| Wrists | Visible centre | 71,004 | 97.6 | 173.0 | +75.4 |
| Wrists | Side strips (discarded) | 20 | 470.4 | 388.7 | −81.7 |
| Wrists | Outside the crop | 6 | 335.4 | 403.3 | +67.8 |
| Ankles | Visible centre | 67,473 | 141.8 | 192.1 | +50.4 |
| Ankles | Side strips (discarded) | 35 | 168.1 | 495.2 | +327.1 |
| Ankles | Outside the crop | 3,522 | 202.6 | 244.7 | +42.0 |

Of the 3,522 ankles outside the crop, 1,410 are also outside the original image (truncated
in the photo); 2,112 are in the image but outside the keypoint box, mostly where the ankle
keypoint is not annotated. The side-strip groups are too small for reliable means.
Share of the student's total deficit coming from joints outside the visible centre:
**wrists −0.02%, ankles 4.5%**.

**Conclusion.** The crop is not a meaningful cause of the student's wrist or lower-body error:
99.96% of wrists and 95% of ankles fall in the 192-wide centre both models see, and the
student's gap there (+75 mm wrists, +50 mm ankles) accounts for essentially the whole deficit;
for out-of-crop ankles the student's gap (+42 mm) is no larger than for visible ones. No crop
change is proposed for accuracy reasons. One check carries over to Checkpoint 2: measure the
same visibility statistics on the BEDLAM2 training crops (detector box, square ×1.25 with
jitter), because training crops, not these evaluation crops, decide what the student learns from.

---

## Checkpoint 2: BEDLAM2 compatibility (read only, no training code)

Diagnostic scripts: `phase1/task1/scripts/c2_*.py`. Outputs:
`/raid/ubx858/outputs/phase1_task1/checkpoint2/`. All numbers below are measured unless marked.

**Correction to an earlier statement.** I previously said 6 render jobs in our manifest had no
labels and that one label file was corrupt. Both came from reading a partially copied zip.
The complete `bedlam2_labels_processed.zip` holds **68** label files; all 68 are now
extracted and verified against the archive (size and CRC). **Every one of our 40 render jobs
has labels.**

### 1. Annotation format

| Source on disk | Content | Granularity |
|---|---|---|
| `bedlam2_labels_processed/<job>.npz` (68 files, 20 GB; CameraHMR's processed labels) | one row per **person per labelled frame** | per frame, every 5th frame (6 fps of 30 fps; all 68 files) |
| `motions_npz_training/*.npz` (9,952) | AMASS-format SMPL-X motions per body (`poses` 165, `trans`, `betas`), 30 fps | per motion |
| `raw/bedlam2.0/*_gt_*csv.tar.gz` | `be_seq.csv` (bodies, motion names, placement), per-frame camera CSVs (Unreal units) | per sequence / per frame |
| `render_db/bedlam2.sqlite` | `gt_camera` (8.0 M rows), `sequence` (`testset`, `affected`, `num_bodies`), `renderjob` | per frame / per sequence |
| `B2RenderStatus.csv` | known issues per render job (free text) | per job |

Label rows: **2,099,549** (single-person jobs 1,105,733 in 41 jobs; multi-person 993,816 rows
over 279,769 images in 27 jobs).

Fields (all verified by reconstruction, see §3):

| Field | Shape | Meaning / units |
|---|---|---|
| `imgname` | str | `seq_XXXXXX/seq_XXXXXX_FFFF.png`; `FFFF` = frame index in the 30 fps video |
| `pose_cam` | 165 | SMPL-X axis-angle: root (camera frame) 3, body 63, jaw 3, eyes 6, hands 90 (radians) |
| `pose_world` | 165 | same, root in world frame; body/hands identical to `pose_cam` |
| `shape` | 16 | betas of the **locked-head neutral SMPL-X** |
| `trans_cam`, `trans_world` | 3 | metres; camera-frame vertices = `SMPLX(pose_cam, shape) + trans_cam + cam_ext[:3,3]` |
| `cam_int` | 3×3 | per-frame pinhole intrinsics (zoom varies), principal point at image centre |
| `cam_ext` | 4×4 | world→camera extrinsic, OpenCV axes (x right, y down, z forward), metres |
| `proj_verts` | 437×3 | projected positions of a fixed set of 437 SMPL-X vertices; 3rd column always 1 |
| `gtkps` | 171×3 | projected keypoints; 138 equal SMPL-X joints exactly, 33 (indices 1–14, 25–43) match no joint or vertex (unknown regressor); 3rd column always 1 |
| `center`, `scale` | 2, 1 | CameraHMR crop box (not used; definition not verified) |
| `lh_c`, `rh_c`, `lh_s`, `rh_s`, `hand_det_conf` | — | hand boxes and detection flags |
| `gender` | str | always `neutral` |

Linking: a label file is one render job; a row is identified by `imgname` only. **There is no
person or body ID**: in single-person jobs each image has one row; in multi-person jobs people
in the same image are not identified, and people are not linked across frames.

Differences from the BEDLAM2 documentation / expectations:
- No face ground truth: jaw and eye rotations are exactly 0 in every row; no expression field.
- No visibility information (`gtkps` / `proj_verts` confidence is always 1, including
  off-image points).
- 33 of the 171 `gtkps` points have an unknown definition.
- 2 label files have no entry in the render database (`20240621_1_250_archmodelsvol8_tracking`,
  same row and sequence counts as `20241107_…_tracking`, probably a re-render; and
  `20240805_5-10_250_busstation_orbit_zoom`). Neither is in our manifest.
- "Test set motions will not be released", yet the labels contain rows in **test-flagged
  sequences**: 1.2% of single-person rows (all in the six MOYO jobs, where the 17 labelled
  test-flagged sequences per job differ from the 17 sequences missing from the labels) and
  37.8% of multi-person rows. With no body ID, test bodies cannot be identified within a
  sequence.

### 2. Body model compatibility

| Item | BEDLAM2 | Our student (EHM-s) | Compatible? |
|---|---|---|---|
| Body model | SMPL-X **locked head**, neutral | SMPL-X 2020 neutral (`SMPLX_NEUTRAL_2020.npz`) body, FLAME head | same topology and skeleton; different shape space |
| Gender | neutral (all rows) | neutral | yes |
| Shape | 16 betas | 200 betas (decoder), padded to 300 | **yes via a fixed linear map** (below) |
| Hand pose | full axis-angle, 15 joints per hand, **flat hand mean** (absolute) | 6D → rotation matrices, 15 joints per hand, no mean added | **yes**, convert axis-angle → rotation matrix |
| Face | none (jaw = eyes = 0, no expression) | FLAME pose / jaw / eyes / eyelids / 50 expr / 300 shape, head scale | **no ground truth** |

**Shape conversion** (`betas_locked16_to_2020_200.npz`): least-squares affine map
β₂₀₀ = M β₁₆ + c fitted on the 5,452 non-head vertices (EHM-s replaces the 5,023 head vertices
with FLAME). On 1,020 real BEDLAM2 rows (15 per label file):

| Measure | Converted (M, c) | Naive copy of 16 betas |
|---|---|---|
| T-pose body vertices | 0.0002 mm mean, 0.004 mm max | 0.31 mm mean, 1.99 mm max |
| Posed body vertices, root-relative | 0.0003 mm mean, 0.008 mm max | — |
| Posed body joints (22) | 0.03 mm mean, 0.86 mm max (head/neck joints) | — |
| Posed hand joints (30) | 0.0002 mm mean, 0.001 mm max | — |
| Head vertices (replaced by FLAME in EHM-s) | 0.22 mm mean, 29.4 mm max | — |

The conversion is lossless for the body. Hand convention confirmed numerically: with
flat-hand-mean the reconstruction matches the stored points to 0.0001 px; without it, up to
20 px. PEAR's EHM-s body path matches the reference `smplx` SMPL-X 2020 implementation on body
vertices (0.025 mm mean, 0.36 mm max; head region up to 5.9 mm).

**Face.** BEDLAM2 provides no face ground truth. The rendered faces are neutral (zero jaw,
zero expression), but supervising our FLAME expression to zero on BEDLAM2 would only teach
"always neutral". Proposal: no face loss from BEDLAM2 in any variant. In (a) and (c) the face
comes from PEAR distillation; in (b) the face head receives **no supervision at all**, so (b)
is a body-accuracy arm only and its face and EHF face numbers are not meaningful.

**Joint sets proposed for the losses.**
- 3D: the 55 SMPL-X joints of SMPL-X 2020 with converted betas (22 body + jaw/eyes + 30 hand),
  pelvis-relative, camera-frame orientation; computed with `SMPLXV2` for both prediction and
  ground truth, so skeletons match exactly. Face joints excluded.
- 2D: the same body and hand joints projected with the **true BEDLAM2 camera** into crop
  coordinates; only joints inside the 256×256 crop contribute.
- Rotations: root (camera frame), 21 body, 30 hand joints (geodesic).
- Shape: L1 on the converted 200-D betas.

### 3. Camera and crop geometry

BEDLAM2: per-frame pinhole `cam_int` (focal varies with zoom, principal point at the image
centre), OpenCV axes, metres; person in camera frame as in §1. Ours: rotation
`diag(-1, -1, 1)`, translation T = (tx, ty, tz) with tz = 24/s, focal 24 over the normalised
[0,1]² square, i.e. **3,072 px for a 256 px crop**, principal point at the crop centre. The
body-model frame equals OpenCV camera axes (verified in the 3DPW evaluation).

**Conversion.** Ground-truth body mesh X = SMPL-X(pose_cam, β) without translation;
t = trans_cam + cam_ext[:3, 3]; square crop with top-left (x0, y0) and side L, a = 256/L;
k = a·f / t_z. Then
T_z = 3072 / k, T_x = −t_x − (a(c_x − x0) − 128)/k, T_y = −t_y − (a(c_y − y0) − 128)/k.
2D targets: true BEDLAM2 projection mapped into the crop.

**Verification** (50 random frames, single-person jobs; `camera_check/`, 10 overlays):

| Projection compared with BEDLAM2's own camera | Mean per frame (median) | Worst frame mean | Worst single vertex |
|---|---|---|---|
| Our crop camera, analytic T | 4.9 px (3.5) | 15.0 px | 55.5 px |
| Our crop camera, best possible T (per-frame least squares) | 3.5 px (2.8) | 11.5 px | 50.4 px |

**It is not ~0, and cannot be with our camera model.** Our camera has a fixed ~4.8° field of
view; BEDLAM2 crops typically span ~30°, so perspective (near/far body parts, off-axis viewing)
cannot be reproduced by any translation. Distant on-axis shots match to 0.1–0.3 px; close-ups
and off-axis people reach 10–15 px mean (correlation of error with off-axis angle 0.49).
Consequences: 2D targets must be the true projections (not re-projections through our camera);
the 2D loss has an irreducible floor of a few pixels; camera translation is best learned through
the 2D loss, with the analytic T as an optional weak target. Fixing this properly needs a camera
model with focal/principal-point input (out of scope; architecture v2).

### 4. Crop boxes

On 400 labelled frames (10 per job, 40 single-person jobs; `crop_boxes/`):

| YOLOX-L vs ground-truth-mesh box (mesh box clipped to the image) | Value |
|---|---|
| IoU, mean / median / p10 | 0.918 / 0.937 / 0.856 |
| Centre offset / box size, mean / p90 | 1.7% / 3.4% |
| Scale ratio YOLOX / GT, mean (p10–p90) | 1.017 (0.996–1.048) |
| Detector misses (IoU < 0.1) | 13 / 400 (3.3%) |

| Option | Cost | Notes |
|---|---|---|
| (i) Regenerate YOLOX boxes | ~30 frames/s with 32 CPU processes (measured on the sample) → ~10 h for the 1.1 M single-person labelled frames, ~13 h with multi-person images (estimate); boxes < 0.2 GB | only visible-extent boxes; 3% misses; no person identity in multi-person frames |
| (ii) Ground-truth mesh box, clipped to the image, + jitter | seconds | exact per person; detector deviation (1.7% centre, 1.7% scale) is far inside the planned ±10% shift / ±15% scale jitter |

**Recommendation: (ii)**, with the mesh box **clipped to the image** before squaring. An
unclipped box puts truncated body parts inside the crop as black padding, which a detector
crop at inference never shows. Keep YOLOX for evaluation (secondary crop) and inference.

Joint visibility on BEDLAM2 crops (square ×1.25; 800 wrist and 800 ankle instances):

| Crop | Wrists: centre / strips / outside | Ankles: centre / strips / outside | Joints off the image |
|---|---|---|---|
| GT mesh box (unclipped) | 800 / 0 / 0 | 798 / 2 / 0 | 27 wrists, 129 ankles drawn as padding |
| YOLOX box | 754 / 5 / 15 | 666 / 1 / 107 | 24 wrists, 125 ankles; 105 of the 107 ankles outside the crop are off the image |
| GT mesh box + jitter | 795 / 5 / 0 | 790 / 8 / 2 | as unclipped |

With ground-truth-box crops (plain or jittered) ≥98.75% of wrists and ankles land in the visible
centre. With YOLOX crops 94% of wrists and 83% of ankles do; nearly all of the rest are body
parts outside the photo itself, which no crop can show. As on 3DPW, the crop is not a
meaningful loss of signal.

Storage: training needs crops with jitter, so frames cannot be the existing pre-cut 256×256
JPEGs. A full 1280×720 frame is ~249 KB (JPEG q95), ~275 GB for all single-person labelled
frames (too large for the 519 GB free on `/raid`). Proposed: store a context crop of 2× the
clipped box at 384×384 per labelled person (estimate ~50 KB → ~55 GB single-person,
~85 GB with multi-person), and apply jitter, rotation and the final 256×256 crop on the fly.

### 5. Data scope

**Multi-person jobs** add 993,816 labelled person-rows (279,769 images), +90% over
single-person; after removing test-flagged sequences 617,863 rows. Per-person ground-truth
boxes make them usable without a detector or tracker. Risks: crowding (up to 10 people per
image, other people inside the crop), heavy occlusion with no visibility flag, and no body ID
for temporal clips. Their MP4s are not extracted yet. Recommendation: single-person jobs only
for the screening runs; add multi-person data to the winner as a separate step.

**BEDLAM2 official test flags.** Our existing manifests contain test-flagged sequences:
1,075 in train and 162 in val (our val split is a hash of the render-job name and ignores the
flags). Proposed fix: drop every test-flagged sequence (database `sequence.testset = 1`) from
all splits, for distillation and ground truth alike, and also drop the labelled-but-test-
flagged MOYO sequences. Cost: 13,410 single-person label rows (1.2%). "Affected" sequences
(known issues per the database) occur only in multi-person jobs (18.4% of their rows); exclude
them when multi-person data is added. Checkpoint selection uses 3DPW validation, so the BEDLAM2
val split is for loss monitoring only.

**UBody.** `annotations/<scene>/smplx_annotation.json` holds per-person SMPL-X **pseudo
ground truth** (10 betas, root/body/hands/jaw axis-angle, 10 expression, translation) with a
virtual camera of focal ~35,558 px (near-orthographic; depth not meaningful);
`keypoint_annotation.json` holds COCO-WholeBody 2D keypoints with validity flags
(`face_valid`, `lefthand_valid`, `full_body`, ...). Most frames are upper-body framings, so the
SMPL-X lower body is largely unconstrained. Assessment: not reliable enough to be ground truth
for body accuracy (our main metric); its 2D keypoints are usable where flagged valid.
Recommendation: **(2) PEAR distillation only**, and only in the variants that use PEAR. For a
clean comparison the screening runs should use **BEDLAM2 single-person frames only** in all
three variants (same images), so that "GT vs distillation" is not confounded with "with vs
without UBody". UBody distillation (at a fixed share, e.g. 20% of each batch instead of the
current ~11× oversampled 50/50 balance, issue T9) is then a follow-up for the winner, which
matters for the webcam/upper-body deployment.

### 6. Mismatch table

| Item | Mismatch found | Proposed fix | Effort | Risk |
|---|---|---|---|---|
| Shape space | locked-head 16 betas vs SMPL-X 2020 200 betas | fixed affine map M, c (saved); lossless on body | low | low |
| Hand pose | none (both absolute, flat hand) | axis-angle → rotation matrices | low | low |
| Face | no face GT in BEDLAM2 | no face loss from BEDLAM2; face from PEAR in (a)/(c); (b) face untrained | low | (b) unusable for face metrics |
| Root translation | `trans_cam` is not the camera-frame translation | t = trans_cam + cam_ext[:3,3] (verified to 1e-4 px) | low | low |
| Camera model | fixed 3,072 px crop focal vs real perspective | 2D targets from true projection; analytic T as weak target; accept 3.5–4.9 px floor | low | 2D loss cannot reach 0; close-ups noisier |
| Crop boxes | never stored (T6) | GT mesh box clipped to image, square ×1.25, ±15% scale / ±10% shift jitter | low | slight train/inference box mismatch (measured small) |
| Frame storage | pre-cut 256 crops prevent jitter | 2× context crops at 384 px for labelled frames (~55 GB) | medium (one-off build, hours) | disk |
| Label rate | GT at 6 fps (every 5th frame) vs 30 fps clips, temporal stride 8 | single-frame training (clip length 1) for all screening variants; temporal losses later on 5-frame strides | low | loses current velocity/delta losses in screening |
| Test split | 1,237 test-flagged sequences in our manifests; MOYO labels disagree with DB flags | drop all DB test-flagged sequences everywhere | low | 1.2% fewer single-person rows |
| Visibility | no visibility/occlusion flags | 2D loss only for joints inside the crop; 3D for all | low | occluded joints still supervised in 3D (correct GT, harder) |
| Multi-person | not used; no body ID | later, with per-person GT boxes, after excluding test/affected | medium | crowding/occlusion |
| UBody | pseudo-GT, upper-body framing, 11× oversampling | distillation only, fixed share, after screening | low | deployment-relevant data delayed |
| `gtkps` | 33 of 171 points undefined | do not use; compute targets from the GT mesh | none | none |

### Proposed plan for Checkpoint 3 (no code written yet)

1. **Data build** (one-off, read from the 40 single-person MP4 jobs): for every labelled row in
   non-test-flagged sequences (~1.08 M rows), decode the frame, compute the image-clipped GT mesh
   box, save a 2× context crop (384×384) and a per-row target record: root/body/hand rotations,
   converted β₂₀₀, 55 pelvis-relative 3D joints, true 2D joints in context-crop coordinates,
   t, K. Keep the existing job-level train/val split. Report the measured build time before running it.
2. **Loader**: on-the-fly crop with ±15% scale, ±10% shift, rotation, colour jitter, synthetic
   occlusion, horizontal flip (with SMPL-X left/right joint swap and pose mirroring), producing
   the 256×256 input and transforming 2D targets and the root orientation consistently.
3. **Losses**, each with its own weight: rotation (geodesic), shape (L1 on β₂₀₀), 3D joints, 2D
   joints (camera-aware, in-crop only), optional camera-T, and the PEAR distillation terms as a
   separate group; PEAR targets computed on the same augmented crop.
4. **Configs** (a), (b), (c): single phase, linear warm-up + cosine, identical seed, batch,
   images and augmentation; same initialisation, no PEAR head copied; selection on 3DPW
   validation with the standard evaluator.
5. **Smoke test** on a small subset: ~200 iterations per config, loss curves, and 10 BEDLAM2
   crops with GT and predictions overlaid; then a full training-time proposal for approval.

Decisions needed: (1) single-frame screening (no temporal losses) — yes/no; (2) BEDLAM2
single-person only for screening, UBody/multi-person as follow-ups — yes/no; (3) ~55 GB context
crop build on `/raid` — yes/no.
