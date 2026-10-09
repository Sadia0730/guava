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
