# GART baseline (AvatarTest-v1, PREREG addendum 1)

Method: GART, Gaussian Articulated Template Models (Lei et al., CVPR 2024). Official code
https://github.com/JiahuiLei/GART at commit `16c11f8a5bb3ae249a9d04dc9d98c316e10f1126` (2023-11-27, "add arxiv id",
the repository HEAD; no later commits). GART's own files are not modified (`git -C code status` clean apart from the
copied SMPL file under the git-ignored `data/`).

## Files (this folder)
- `setup.sh`: rebuilds code + env from scratch under `/dev/shm/ubx858_baselines/gart/` (`code/`, `env/`, `cache/`).
- `avatartest_loader.py`: the only loader code; a dataset class with GART's dataset interface reading the contract package.
- `run_gart.py`: train on `recon`, render `test`, write `run.json`; `--sanity N` = camera check only.
- `run_clip.sh <dataset> <clip> [args]`: sets the cache variables, clears PYTHONPATH, runs `run_gart.py` in the env.

## Environment
GART `install.sh`: python 3.9, `pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8` (pytorch + nvidia channels),
fvcore / iopath, `pytorch3d` (pytorch3d channel), `requirements.txt`, `lib_render/simple-knn`,
`lib_render/diff_gaussian_rasterization-alphadep`, lpips. Differences: prefix env; caches (pip, conda pkgs, torch, HF, XDG,
TMPDIR, matplotlib, torch extensions) under `/dev/shm/ubx858_baselines/gart/cache`; CUDA 11.8 nvcc / headers installed
into the env to build the two CUDA extensions against torch 2.0.0+cu118 (machine nvcc is 12.8), `TORCH_CUDA_ARCH_LIST=8.9`
(L40S); MVDream (Text2GART generation guidance only) not installed. Body model: `/home/ubx858/ml-hugs/data/smpl/SMPL_NEUTRAL.pkl`
(SMPL v1.1.0 python release, 10 betas; the same pkl the input package's SMPL fit used, via its npz conversion) copied to
`code/data/smpl_model/SMPL_NEUTRAL.pkl`.

## Configuration: `profiles/people/people_1m.yaml`, unchanged
Candidates shipped for real monocular human video: People-Snapshot (`people_30s / people_1m / people_2m`, loader
`instant_avatar_people_snapshot`) and UBC-Fashion (`ubc_mlp / ubc_list`, loader `instant_avatar_wild`).
People-Snapshot is chosen because it is the setting of our input: one monocular video, known camera (calibrated, identity
extrinsics), given SMPL per frame, and a temporal split (train = first part of the video, test = the later part with
novel poses), exactly GUAVA's Table 2 protocol. The UBC profiles target in-the-wild fashion videos with estimated poses
(ReFit) and interleaved train/test frames (train 0::4, test 2::4), add 32 extra bones and view balancing to absorb
loose clothing / pose error, and `ubc_list` drives the extra bones by a per-training-frame list (undefined for novel
test poses). Within People-Snapshot, `people_1m` is the repository default (`solver.py --profile` default and
`example.ipynb`); the paper's People-Snapshot table reports the shorter `people_30s` schedule ("GART (30s)"),
`people_2m` is a longer variant. people_1m: 2000 steps, 1 frame / step, SH order 1, DA-pose canonical, voxel skinning
field 64, 4 extra pose-MLP bones, random background, SSIM 0.1, training-frame SMPL pose + translation refinement from
step 500 (lr 1e-4) - kept on (training frames only). Run as `script/fit_people_*.sh` does (`--fast`, i.e. no viz).

## Loader changes (all in avatartest_loader.py / run_gart.py; also in each run.json "loader_changes")
1. File layout: `E/base/<dataset>/<clip>/` (frames.json, images/<key>.png, masks/<key>.png, cameras.npz, smpl.npz).
2. Frames: frames.json `recon` keys for training, `test` keys for rendering (instead of START_END slices).
3. Per-frame intrinsics `K[key]`: GART's `RealDataOptimizablePoseProviderPose` already keeps one K per frame and its
   renderer takes K per call, so only the loader fills it per frame (the PeopleSnapshot loader repeats one K).
   Our K has fx = fy and principal point (256, 256) = W/2, the centred case of GART's renderer (no padding path).
4. Masks: StyleMatte alpha > 127; image composited on white with the mask, as GART's loaders do.
5. No resize: GART's IMAGE_ZOOM_RATIO (0.5) exists to bring PeopleSnapshot's 1080 px frames to 540 px; our crops are
   512 px and the registered protocol trains / renders at 512, so the loader does not downscale (this is the profile's
   only key not used; the method hyper-parameters are untouched).
6. Body model / params: neutral SMPL, smpl.npz betas10 (one per clip), per-frame global_orient + body_pose (23x3) +
   transl in the camera frame (GART's PeopleSnapshot runs use gendered SMPL and InstantAvatar-refined poses).
7. Test rendering: the lines of GART's `_save_render_image_from_pose` (model forward with the GT pose, `render_cam_pcl`,
   pixels with alpha == 0 set to the background) with background black, saving only the prediction as `<key>.png`
   (8 bit, 512x512). No test-time pose optimisation (GART's published People-Snapshot evaluation runs TTO on test poses;
   that would be a separately named extra variant and is not run).
8. Not run: `viz_human_all`, `eval_fps` (visualisation / FPS benchmark after training; no effect on the model).
9. Import shim (environment, not method): `lib_gart/templates.py` imports the dog model `smal.smal_tpg`
   unconditionally and that module reads BITE's `smal_data/mean_dog_bone_lengths.txt` at import time (data the README
   asks for only for dogs). `run_gart.py` registers a placeholder `smal.smal_tpg` in `sys.modules` (raises if a SMAL
   template were ever built); the human path never touches it.
10. GART's `RealDataOptimizablePoseProviderPose.pose_diff` property references a missing attribute (`pose_list`); the
   training-pose refinement magnitude reported in run.json is computed in `run_gart.py` from the provider's parameters
   (diagnostic only).

## Timing definition (run.json)
`train_s` = wall time from python start of `run_gart.py` (imports, reading the recon frames from /vault, model init,
2000 steps, checkpoint save) to the end of training. `render_s_per_frame` = CUDA-synchronised forward + rasterisation +
device-to-host per test frame (mean); PNG encoding and the copy to /vault are excluded (reported in notes).
Checkpoints / work dirs live in `/dev/shm/ubx858_baselines/gart/work/` and are deleted after the renders are copied.

## Results of the setup task (2026-10-31, GPU 0 = NVIDIA L40S, sequential, nothing else on that GPU)

Camera-convention check (`run_gart.py --sanity 8`, 8 recon frames each; untrained GART template = Gaussians on the
posed SMPL mesh rendered by GART's renderer with K[key]; also the plain SMPL mesh silhouette; IoU vs alpha > 127;
files `camera_sanity.{json,jpg}` next to each run.json):

| clip | IoU GART template | IoU SMPL mesh | IoU template vs mesh |
|---|---|---|---|
| bedlam2/20240731_1_1827_hdri__seq_000095 (exact GT) | 0.900 | 0.927 | 0.900 |
| ubody/ConductMusic_S21_Trim8 (pseudo-GT, loose jacket / hair / violin) | 0.769 | 0.758 | 0.923 |

Budget-rule timing (full train + test render; `train_s` as defined above):

| clip | recon / test | iterations | train_s | of which GART fit loop | render s / frame | Gaussians |
|---|---|---|---|---|---|---|
| ubody/ConductMusic_S21_Trim8 | 154 / 154 | 2000 | 134.1 | 112.6 | 0.0060 | 21,253 |
| bedlam2/20240731_1_1827_hdri__seq_000095 | 48 / 48 | 2000 | 162.0 | 143.9 | 0.0057 | 26,673 |

Whole script wall (train + render + PNG + copy to /vault) about 145 s (UBody) and 171 s (BEDLAM2); the fixed
2000-step schedule does not scale with the number of recon frames (observed 17.8 it/s on the UBody clip, 13.9 it/s
on the BEDLAM2 clip, which ended with more Gaussians; per-clip times will vary). Projection for all 70 clips on one L40S, sequential: 30 x 145 s + 40 x 171 s = 3.1 h.
