# GaussianAvatar baseline on AvatarTest-v1 (PREREG "AvatarTest-v1 addendum 1")

GaussianAvatar (Hu et al., CVPR 2024), official code https://github.com/huliangxiao/GaussianAvatar
@ `d981c62238ef64e89dcc04719d2ebbb4758b080a` (2024-03-26, "readme update"). The repository tree is used unmodified
(`git status` shows only the unzipped assets / SMPL copies as untracked files); every adaptation lives in this folder.

## Files
| file | role |
|---|---|
| `setup.sh` | rebuilds everything under `/dev/shm/ubx858_baselines/gaussianavatar/` (code, conda env, CUDA ext., assets, SMPL copies, caches) |
| `env_conda_list.txt` | `conda list` of the env built by `setup.sh` (2026-10-31) |
| `run_clip.py` | one clip: convert -> stage 1 -> export -> posmaps -> stage 2 -> test render -> `run.json`; deletes the work dir |
| `ga_convert.py` | input package -> GaussianAvatar data folder (train = recon keys only, test = test keys) |
| `ga_official.py` | runs an official preprocessing script with only its hard-coded path lines replaced (README: "remember to change the path") |
| `ga_train.py` | runs the official `train.py` (black background, compat shim, stage-2 TensorBoard off) |
| `ga_render.py` | test render (official `render_free_stage2` with the test frame's GT pose) |
| `ga_compat.py` | pytorch-2.0.1 CPU/CUDA indexing shim (no numerical effect) |
| `cam_sanity.py` | camera-convention check (method's own template + loader camera + rasterizer vs masks) |

## Environment (setup.sh)
- `environment.yml` of the repository (python 3.10.13, pytorch 2.0.1 / torchvision 0.15.2, pytorch-cuda 11.8, tqdm,
  tensorboard, scipy, pip opencv-python) with three pins it leaves open: numpy 1.23.5, mkl 2023.1.0, opencv-python
  4.8.1.78 (resolved today the yml gives numpy 2 / mkl 2025, and pytorch 2.0.1 then fails to import:
  `undefined symbol: iJIT_NotifyEvent`; numpy < 1.24 is also what chumpy needs to unpickle the official SMPL .pkl).
- Packages the code imports but the yml omits: lpips 0.1.4, open3d 0.18.0, trimesh 4.0.10, torchmetrics 1.2.1,
  PyOpenGL 3.1.7, smplx 0.1.28 (pip, imported by `scripts/gen_pose_map_our_smpl.py`), chumpy 0.70, pytorch3d 0.7.4
  (official prebuilt wheel py310_cu118_pyt201; only `pytorch3d.transforms` is used), plyfile, imageio, pillow.
- README: "compile diff-gaussian-rasterization and simple-knn as in 3DGS": graphdeco-inria/diff-gaussian-rasterization
  @ `59f5f77e` (the commit gaussian-splatting pinned before its Oct-2024 update; returns `(color, radii)`, the 2-output
  API `gaussian_renderer/__init__.py` unpacks) and simple-knn @ `44f76429`, built with CUDA 11.8 nvcc from
  `nvidia/label/cuda-11.8.0`, `TORCH_CUDA_ARCH_LIST=8.9` (L40S).
- Official assets: OneDrive `assets.zip` (17 MB, sha256 `caec1c55...7588`) unzipped into `assets/` (uv masks, LBS maps,
  barycentric maps, faces, UV template). SMPL: copies of `/home/ubx858/ml-hugs/data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`
  (v1.1.0 python release, unmodified) in `assets/smpl_files/smpl/`; only NEUTRAL is used (`smpl_gender='neutral'`).
- freeglut (conda-forge) + the system Xvfb (`xvfb-run`) for the official GLUT position-map renderer
  (`scripts/posmap_generator/lib/renderer/gl`), which needs a window.
- All caches (PIP_CACHE_DIR, CONDA_PKGS_DIRS, TORCH_HOME, HF_HOME, XDG_CACHE_HOME, TMPDIR) under the /dev/shm folder.

## Recipe (official "Run on Your Own Video", README; all hyper-parameters = `arguments/__init__.py` defaults)
1. `scripts/gen_pose_map_cano_smpl.py` on `data/train` -> canonical query posmap 512 (Gaussian anchors) + canonical joint mats.
2. Stage 1: `train.py --train_stage 1 --pose_op_start_iter 10` -- 200 epochs, batch 2, Adam lr_net 3e-3 /
   geo-feature 5e-4, MultiStep milestones [66, 133] (stepped per iteration, as coded), L1 + 0.2 D-SSIM + scale / offset /
   feature regularisers, LPIPS (alex, 0.2) after epoch 30; **training-frame pose + translation refinement** (SparseAdam
   5e-3 on per-frame SMPL pose / transl embeddings) from epoch 11 on. This is the README's own-video command; the paper's
   "joint optimisation of motion and appearance".
3. `scripts/export_stage_1_smpl.py` -> refined training poses `smpl_parms_pred.pth`.
4. `scripts/gen_pose_map_our_smpl.py` (`parms_name = smpl_parms_pred.pth`) -> 128 posmaps of the posed training meshes.
5. Stage 2: `train.py --train_stage 2 --stage1_out_path <stage1>/net/iteration_200` -- 200 more epochs, pose encoder
   (UNet on the posed-body posmap) + decoder fine-tune (lr 3e-4), poses frozen. = the paper's dynamic-appearance stage.
- People-Snapshot's released model is stage 1 only on pre-refined poses (README "Run on People Snapshot"); the
  own-video recipe (both stages, pose refinement on) is the one for a monocular video with given SMPL.
- Iterations per stage = floor(n_recon / 2) x 200 (epochs are fixed, so the step count scales with the clip length).

Test: `data/test` gets the same canonical posmap (same betas), `smpl_parms_pred.pth` = the GT SMPL of the test frames
(no optimisation) and their 128 posmaps from the same official script; `ga_render.py` renders each test frame with its
GT pose and its K into the 512 crop, background black, 8-bit PNG.

**Test-time poses.** The code has no test-time pose optimisation. The paper says it "follow[s] the same evaluation
protocol outlined in InstantAvatar" for People-Snapshot, whose released poses (`poses_optimized.npz`, Anim-NeRF) were
refined on all frames including the test frames, i.e. its published numbers use test poses fitted to the test images
by another method. Not reproduced here (main protocol = GT test poses); a separate variant would need test-pose fitting
code GaussianAvatar does not ship.

## Every change relative to the official pipeline
1. **Per-frame intrinsics** -- none in code: the loader already has a per-frame camera path (`--cam_static 0` reads
   `cam_parms/<name>.npz` with `extrinsic`, `intrinsic`); `ga_convert.py` writes extrinsic = I, intrinsic = K[key].
   Projection check: `getProjectionMatrix(K)` + the 3DGS rasterizer give u = fx x/z + cx in the contract's continuous
   pixel convention (3DGS pixel index i = continuous i + 0.5).
2. **Black background** (registered GT = image x mask on black; renders on black). The official loader composites the
   masked image on white (`image * mask + (1 - mask) * 255`, mask >= 128) and `white_background` is True and cannot be
   switched off from the CLI (store_true, default True). Instead: `ga_convert.py` composites on black with the same
   threshold (alpha > 127), the loader runs with `--no_mask 1` (reads the image as is), and `ga_train.py` sets
   `white_background=False` (rasteriser background black). Training and test background are therefore the same, as in
   the official eval (which uses white for both).
3. **Test rendering** -- the official `render_free_stage2` takes the pose from the *training* pose embedding indexed by
   the test frame index (`self.pose(idx)`), so it cannot drive novel poses; `eval.py` only loads stage-1 checkpoints.
   `ga_render.py` copies `render_free_stage2` verbatim except that pose / translation come from the test frame's GT SMPL
   (`batch['pose_data']`, `batch['transl_data']`, exactly as `render_free_stage1` does), loads the stage-2 checkpoint
   with `stage2_load`, and otherwise mirrors `eval.py` (batch 1, modules left in train mode, so BatchNorm uses
   per-batch statistics as in the official evaluation).
4. **Checkpointing** -- `--save_epoch 200`: the official save rule (`epoch > save_epochs[0]=100 and epoch % save_epoch
   == 0`) with the default 30 never writes epoch 200, whereas the released People-Snapshot model is `iteration_200`.
   Only the final model is saved; training is unaffected.
5. **Compatibility shims** (no numerical effect): (a) `AvatarModel.__init__` indexes CPU tensors with the CUDA mask
   `valid_idx` -> pytorch 2.0.1 raises "indices should be either on cpu or on the same device"; `ga_compat.py` gives
   `__init__` the mask on the CPU and moves it back to CUDA afterwards. (b) Stage 2 crashes at its first iteration when
   TensorBoard is installed (`train.py` logs `scale_loss`, defined only in stage 1); `ga_train.py` makes the
   tensorboard import fail for stage 2 so `train.py` takes its own "not logging progress" path. Stage 1 logs normally.
6. **Script paths** -- `gen_pose_map_cano_smpl.py`, `export_stage_1_smpl.py`, `gen_pose_map_our_smpl.py` have
   placeholder paths the README tells the user to edit; `ga_official.py` replaces exactly those lines (and
   `parms_name` -> `smpl_parms_pred.pth` for the "predicted position map") and runs them under `xvfb-run`.

## Camera sanity (cam_sanity.py, all recon frames)
GaussianAvatar's own initial template (canonical query points, no offsets, posed by its LBS with the GT SMPL, its loader
camera with our K, its 3DGS rasteriser, 4 mm white splats) vs `masks/<key>.png > 127`; reference = SMPL mesh filled with
cv2 at pixel centres using K.

| clip | frames | IoU template vs mask (mean / min) | IoU mesh fill vs mask | IoU template vs mesh fill |
|---|---|---|---|---|
| ubody/ConductMusic_S21_Trim8 | 154 | 0.789 / 0.750 | 0.766 | 0.930 |
| bedlam2/20240731_1_1827_hdri__seq_000095 | 48 | 0.888 / 0.858 | 0.925 | 0.886 |

UBody: residual = loose jacket, hair, violin and the SMPL lower body below the mask's bottom edge; BEDLAM2: tight
overlap (the template IoU is lower than the mesh fill only because 4 mm splats leave gaps). Overlays: `logs/cam_sanity_*.jpg`
next to the clips' run.json.

## Timing (budget rule: the 2 registered timing clips; nothing else was run)
Run 2026-10-31 with `run_clip.py`, one clip at a time, NVIDIA L40S. train_s = wall time from the start of the
conversion to the end of stage 2 (all preprocessing included); renders + run.json + logs in
`E/renders/gaussianavatar/<dataset>/<clip>/`, work dirs (checkpoints, posmaps) deleted from /dev/shm afterwards.

| | bedlam2/20240731_1_1827_hdri__seq_000095 | ubody/ConductMusic_S21_Trim8 |
|---|---|---|
| GPU | 0 (only process on it when checked; steady rate) | 2 (shared server; rate fluctuated 3.2-5.8 it/s from ~11 min on -> probably contended) |
| recon / test frames | 48 / 48 | 154 / 154 |
| iterations (stage 1 + stage 2, batch 2) | 4,800 + 4,800 = 9,600 | 15,400 + 15,400 = 30,800 |
| convert + canonical posmap | 5.2 + 4.8 s | 17.6 + 6.6 s |
| stage 1 (incl. pose refinement) | 870.2 s (0.181 s/it) | 3,673.2 s (0.239 s/it) |
| export + training posmaps | 2.1 + 6.6 s | 2.2 + 11.9 s |
| stage 2 | 990.6 s (0.206 s/it) | 3,320.8 s (0.216 s/it) |
| **train_s** | **1,879.4 s (31.3 min)** | **7,032.3 s (117.2 min)** |
| test prep: convert / canonical posmap / pose posmaps | 7.9 / 7.5 / 20.3 s | 23.7 / 4.4 / 12.7 s |
| render: model load / loop / pure GPU per frame | 2.9 s / 0.154 s/frame / 40.1 ms | 2.0 s / 0.089 s/frame / 25.9 ms |
| run.json render_s_per_frame (posmaps + loop) | 0.734 s | 0.201 s |
| GPU memory (training) | ~20 GB (nvidia-smi) | not recorded |

Projection, all 70 clips on 1 GPU:
- registered rule (mean measured time per clip x 70): (1,929.6 + 7,091.4 s) / 2 x 70 = **87.7 h** -> <= 48 h needs >= 2
  free GPUs at launch (2 GPUs: 43.9 h). The two timing clips are longer than average (UBody mean 117.5 recon frames,
  BEDLAM2 mean 36.5), so this overestimates.
- iteration-weighted (time is ~linear in iterations = 200 x floor(n_recon/2) per stage; split totals 351,000 UBody +
  144,600 BEDLAM2 per stage) + ~80 s fixed per clip: 54.7 h at the BEDLAM2 (uncontended) rates, 61.2 h with each
  dataset's own measured rates. subset24 (152,600 + 45,800 per stage): 22-25 h (rule-literal 30.1 h).
