# G0 — What GUAVA releases, for runtime Gaussian-update experiments (read-only check, 2026-10-13)

**Scope.** Upstream GUAVA (github.com/Pixel-Talk/GUAVA, 3 commits, last on 2025-10-08) and EHM-Tracker
(github.com/Pixel-Talk/EHM-Tracker), compared with our fork. Nothing was downloaded, changed or run.
Marks: **V** = verified in local files, the GitHub API or fetched pages; **I** = inferred. Four claims were
re-checked by hand and are marked ✔:
- the `bg_color` bug;
- the checkpoint being present;
- the `EHM.py:27` bug;
- the truncated HaMeR file.

**Summary.**
- **Released:** inference code, training code, weights and the tracker.
- **Not released:** the paper's dataset, its 58-video test list, and any per-subject optimisation or
  multi-frame fusion code.
- **Consequence:** runtime Gaussian-update experiments are possible with what is released, but the
  update and fusion code has to be written, and our own test split has to be defined.

## 1. Inference — available

**Entry points (V):**
- `main/test.py` runs self-reenactment (default), cross-reenactment (`--render_cross_act`) and novel
  views.
- Wrappers are in `main/run_shells/run_test_*.sh`; `app.py` is a Gradio demo.
- The config is `assets/GUAVA/config.yaml`.

**Input (V):** an EHM-Tracker output directory: `img_lmdb/`, `optim_tracking_ehm.pkl`,
`id_share_params.pkl` and `videos_info.json`.
- The source is frame 0.
- Every driving frame needs tracked parameters.

**Weights:**
- **GUAVA:** `assets/GUAVA/checkpoints/best_160000.pt`, 1.30 GB, **present locally ✔**. It comes from
  Google Drive; there is nothing on Hugging Face.
- **Parametric models:** FLAME 2020 and SMPL-X 2020 are present.

**Missing:**
- `assets/example/tracked_video/`: `tracked_video.zip`, 1.08 GB, Google Drive. It is needed for the
  README demos.
- The rasteriser and the GPU build of PyTorch3D exist only under `/tmp`, not in the conda env.

## 2. Training — code complete, data not released

**Code (V):**
- `main/train.py` and `main/trainer.py` (Lightning Fabric).
- Losses: L1 + LPIPS on the refined and raw renders, head and hand crop losses, regularisers.
- `configs/train/ubody_512.yaml`: batch 6, 200k iterations.
- The README asks for 48 GB of VRAM; the paper reports ~156 A6000-hours.

**Bug ✔:**
- `main/trainer.py:143` reads `MODEL.bg_color`, which no config defines.
- It would fail at the first validation (iteration 10k).
- The fix is a one-line config entry.

**Training data (V):** the paper's set (YouTube + OSX/UBody + How2Sign; >26k clips, 620,511 frames) is
**not released**. GitHub issue #3, which asks for it, is unanswered.
- **UBody:** our raw copy is at `/raid/ubx858/datasets/raw/UBody`.
- **How2Sign:** public, CC BY-NC 4.0.
- **YouTube part:** unavailable.
- **Data preparation (V):** EHM-Tracker `tracking_video.py --tracking_with_interval`, then
  `build_lmdb_dataset` and `split_dataset`.

**Leakage (I):** the released weights saw UBody and How2Sign videos, so these are not clean held-out
data for them.

## 3. Test split and metrics — metrics released, split not

**Test split (V):** the paper's test set (58 IDs, ~15 s each, 28,287 frames; cross-reenactment with 10
sources × 8 ViCo-X drivers) is not released. There is no ID list, no videos and no tracking results.

**Split mechanism (V):** `TrackedData_infer` can read a `testing_split.json` (the last N frames per
video), but no script creates one.

**Metrics (V):**
- `main/metrics.py`: PSNR, SSIM, LPIPS-VGG and MAE on masked 512² renders.
- `main/metrics_face.py`: ArcFace identity similarity.
- Paper self-reenactment: PSNR 25.87, SSIM 0.900, LPIPS 0.0813.

## 4. EHM-Tracker (offline) — available, with local defects

**Code (V):** Apache-2.0. It covers video and single-image tracking and the dataset builder.
- **Upstream fix missing locally ✔:** commit `176f4d88` (2026-01-21) fixes
  `register_buffer('right_hand_center', left_hand_center)`. Our copy still has the bug at
  `EHM-Tracker/src/modules/ehm/EHM.py:27`.

**Models:** `pretrained.zip` (7.08 GB: DWPose, PIXIE, HaMeR-ONNX, TEASER, FAN, MediaPipe, StyleMatte,
VPoser) is extracted locally.
- **Truncated file ✔:** `pretrained/hamer/onnx__MatMul_4769` is 6.6 MB on disk against 26.2 MB in the
  zip. Re-extract it from the local zip.
- **Missing links:** `EHM-Tracker/assets` lacks links to `SMPLX_NEUTRAL_2020.npz`,
  `flame_generic_model.pkl` and `FLAME/FLAME2020/`.

**Output (V):**
- **Per frame:** SMPL-X, FLAME and MANO coefficients, the camera and 2D keypoints.
- **Per video:** shape, scales and joint offsets, averaged over the whole video.
- **Images:** masked 1024² images stored in LMDB.

**Speed (V):** offline only. Upstream warns that "a 10-second video can take over 6 minutes". Frames
with low hand confidence are dropped, so tracked videos can have gaps. Runtime driving parameters
must therefore come from PEAR or our student.

## 5. What matters for runtime Gaussian updates (V, from code)

**Build step:** `Ubody_Gaussian_inferer.forward` runs once per source image. A frozen DINOv2 encoder
produces:
- ~10.5k template Gaussians, one per EHM vertex;
- ~190k UV Gaussians, one per valid texel of the 512² UV map, bound to mesh triangles;
- a 32-channel colour feature per Gaussian.

**Per frame:** no learned network. EHM skinning, then the 32-channel rasteriser, then the StyleUNet
refiner.
- There is no pose-dependent or clothing deformation.
- The refiner adds random noise unless called with `randomize_noise=False`.

**What can be optimised:**
- The avatar attributes are plain tensors (not `nn.Parameter`).
- The rasteriser backpropagates to means, colours, opacities, scales and rotations, but not to the
  camera.
- The UV indexing gives fixed correspondence between reconstructions from different frames, but only
  without pruning: pruning drops Gaussians with opacity ≤ 0.001.
- Per-frame face visibility is already computed (`mesh_renderer.render_fragments`).

**Not in the release:** no test-time optimisation, per-subject fine-tuning or multi-frame fusion code.
The authors only suggested a LoRA on the decoders (GitHub issue #10).

## 6. Licences

| Component | Licence |
|---|---|
| GUAVA code | Apache-2.0 (V) |
| Bundled Gaussian rasteriser and simple-knn | Inria licence, research / non-commercial (V) |
| GUAVA weights | no statement (V); treat as research-only (I) |
| EHM-Tracker code | Apache-2.0 (V) |
| SMPL-X, FLAME, MANO | non-commercial, no redistribution (V) |
| How2Sign | CC BY-NC 4.0 (V) |
| UBody | no licence stated (V) |

## To download or request (nothing done)

1. `tracked_video.zip`, 1.08 GB: GUAVA example tracked videos, for the demos. Optional.
2. Optional: EHM-Tracker `examples.zip` (size not shown).
3. Request from the authors: the GUAVA dataset or video list, and the 58-video test list.
4. Local fixes, no download needed:
   - re-extract the truncated HaMeR file;
   - link the SMPL-X and FLAME files into `EHM-Tracker/assets`;
   - apply upstream commit `176f4d88`;
   - add `MODEL.bg_color` to the training config;
   - install the rasteriser and PyTorch3D outside `/tmp`.
