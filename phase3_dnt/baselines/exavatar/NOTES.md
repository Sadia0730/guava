# ExAvatar baseline (AvatarTest-v1, PREREG addendum 1)

Official code: `mks0601/ExAvatar_RELEASE` @ `518f5647c8fa75cb648315c1677fdc105101fe11` (main branch = custom-video /
NeuMan code). Everything heavy lives in `/dev/shm/ubx858_baselines/exavatar/` (rebuilt by `setup.sh`); nothing on /raid,
only renders + `run.json` + small logs go to `E/renders/exavatar/<dataset>/<clip>/`.

## Files
| file | role |
|---|---|
| `env.sh` | paths + cache variables (PIP_CACHE_DIR, CONDA_PKGS_DIRS, TORCH_HOME, HF_HOME, XDG_CACHE_HOME, TMPDIR ... all under /dev/shm) |
| `setup.sh` | rebuilds code (pinned commits), the 2 conda envs, weights (SHA-256 checked) and `human_model_files` |
| `run_clip.sh <dataset> <clip> [gpu\|auto]` | whole pipeline for one clip, timing marks, run.json, copy to E, cleanup |
| `exa_common.py` | GT SMPL-X 2020 -> ExAvatar SMPL-X layer conversion |
| `prep_clip.py` | builds ExAvatar `Custom` subject folders from the input package |
| `fit_face_gt.py` | ExAvatar `fitting/main/fit.py` with the body frozen at the GT (face fitting only) |
| `finalize_params.py` | GT body + smoothed ExAvatar face params, ID files, meshes for Depth-Anything |
| `render_test.py` | test renders (animate.py path, refined Gaussians, black bg, our K) |
| `sanity_camera.py` | camera / pose-convention check through ExAvatar's own renderer (IoU vs masks) |
| `write_run_json.py` | run.json (contract keys + timing breakdown) |

## Environments
* `envs/exavatar` (python 3.10, torch 2.6.0+cu124 / torchvision 0.21.0 as ExAvatar's `environment.yml`; pytorch3d V0.7.8
  and `leo-frank/diff-gaussian-rasterization-depth` @03f0b7d built from source with the system CUDA 12.8 nvcc for sm_89;
  chumpy from git master (0.71, as environment.yml); pinned pip versions from environment.yml for the packages used).
  The yml's kitchen-sink packages (isaacsim, genesis, sam-2, ...) are not installed: the steps run here do not import them.
* `envs/exavatar_pose` (python 3.10, torch 2.1.2+cu121, mmcv 2.1.0 built from the PyPI sdist, mmdet 3.3.0, mmpose v1.3.2):
  mmcv 2.1 has no wheel for torch 2.6 and download.openmmlab.com (prebuilt wheels, checkpoints) is unreachable from this host.
* Tools cloned into `fitting/tools` as the README says: DECA @a11554ae, Depth-Anything-V2 @a561b849, mmpose v1.3.2;
  ExAvatar's `copy_code.py` run unchanged (its Hand4Whole / segment-anything copies fail harmlessly: not installed).
* `avatar/main/config.py`: `dataset = 'Custom'` (the README's "Set dataset in main/config.py"); nothing else edited.

## Weights / body models
| item | source | size |
|---|---|---|
| deca_model.tar | Google Drive id from DECA's fetch_data.sh | 434 MB |
| dw-ll_ucoco_384.pth (DWPose-l 384) | HF yzd-v/DWPose (official) | 407 MB |
| rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth | HF facebook/sapiens-pose-bbox-detector (openmmlab host unreachable; SHA-256 b66b2707..., i.e. a re-saved copy, not the 235e8209-hashed original) | 99 MB |
| depth_anything_v2_vitl.pth | local copy (VACE annotators), SHA-256 a7ea19fa... = HF depth-anything/Depth-Anything-V2-Large | 1.34 GB |
| FAN s3fd + 2DFAN4 (DECA face detector) | face_alignment auto-download (adrianbulat.com) | 86 + 92 MB |
| VGG16 (LPIPS loss) | torchvision auto-download | 553 MB |
| pip / conda packages | PyPI, download.pytorch.org, conda-forge | ~5 GB |

Body models (copies): SMPL-X v1.1 neutral/male/female (`/raid/.../smplx_v1_1`), FLAME 2020 = `assets/FLAME/FLAME2020/generic_model.pkl`,
vertex ids / UV obj from `assets/SMPLX`. Files the pipeline needs that are not in our assets (FLAME 2019 generic model,
FLAME 2020 male/female, FLAME static/dynamic landmark embeddings, FLAME_texture.npz, smplx_flip_correspondences.npz) are
copied from LHM++'s `human_model_files` (`/raid/ubx858/LHM-plusplus_data/.../human_model_files`, ExAvatar's exact layout;
its SMPL-X v1.1, FLAME 2020, vertex-id and UV files are byte-identical to our official copies). No MPI download was needed.

## Pipeline per clip (what ran, what was replaced)
ExAvatar custom-video pipeline (`fitting/tools/run.py` + avatar README) vs. here:

| ExAvatar step | here |
|---|---|
| extract_frames / frame lists | our 512 crops; `frame_list_train` = recon keys; test keys only for the face fit and the render |
| make_virtual_cam_params (focal 2000) / COLMAP | **replaced**: per-frame K of the crop (`cam_params/<idx>.json`, R = I, t = 0) |
| DECA (initial FLAME) | **ran** (run_deca.py unchanged), separately on the recon frames and on the test frames (its mean FLAME shape comes from recon frames only) |
| Hand4Whole (initial SMPL-X) | **replaced by the GT** (smplx_init = converted GT incl. GT jaw / expression as the init) |
| mmpose whole-body 2D keypoints (RTMDet + DWPose) | **ran** (run_mmpose.py unchanged) on recon and test frames (raw crops) |
| fit.py (SMPL-X + FLAME joint fit) | **face part ran, body replaced by the GT** (`fit_face_gt.py`): same losses / schedule; body poses, trans, betas, joint/locator offsets frozen at the GT; FLAME params + flame_shape + face_offset fitted on recon frames; test frames: per-frame FLAME only, ID frozen |
| unwrap.py (face texture) | **ran unchanged**, recon frames only (it uses frame_list_train) |
| smooth_smplx_params.py | its smoother applied to the **face params only** (jaw, eyes, expr), recon / test separately; GT body not smoothed |
| SAM masks | **replaced** by the given StyleMatte alpha |
| Depth-Anything-V2 background point cloud (static-camera route) | **ran unchanged** on the avatar folder (black-composited recon frames -> black background points) |
| train.py | **ran unchanged** (default config: 5 epochs x 15 repeats of the train list, batch 1, scene + human Gaussians, per-frame SMPL-X refinement of training frames at lr 1e-4, shape / joint-offset refinement) |
| test.py / fit_pose_to_test (NeuMan protocol: test-pose optimisation) | **not run** (main protocol: no test-time pose fitting). ExAvatar's published NeuMan evaluation does fit test poses -> possible extra variant |
| animate.py | basis of `render_test.py` |

### GT conversion (SMPL-X 2020 neutral -> ExAvatar's SMPL-X layer)
ExAvatar hard-codes the SMPL-X v1.1 **male** model (`smplx_gender = 'male'`, also in the fitting code), 100 betas, with
FLAME-2020 expression dirs on the face and `flat_hand_mean=False`. Kept. Conversion: global_orient / body_pose copied;
hands = absolute GT - v1.1 hand mean (verified: posed-vertex error on the hands 5.6 mm vs 26-29 mm with absolute poses);
betas100 = least squares onto the GT rest shape (all vertices; residual 1.2 / 1.5 mm mean, 10 / 12 mm max on the 2 clips);
joint_offset (an ExAvatar ID parameter) = GT rest joints - ExAvatar rest joints (male regressor, incl. ExAvatar's fitted
face_offset) so the kinematic chain equals the GT's (before it: 3.0-3.1 mm mean, 16-17 mm max joint error); trans =
transl_gt + pelvis_gt - pelvis_ex; at render time re-anchored to the trained shape's pelvis (ExAvatar refines betas
during training; shift ~1 mm). Face: jaw / eyes / expression are ExAvatar's own per-frame face fit (GT face params are
only the fitting init / prior, as Hand4Whole's would be). The male skinning weights / regressor and ExAvatar's lack of
body pose-correctives give a posed-mesh difference to the GT of 5.4 / 5.7 mm mean (sanity check below).

### Background (decision, stated)
Our crops follow the person with a per-frame camera (identity extrinsics), so the scene is not static in ExAvatar's world
frame; both ExAvatar background routes assume it is (static camera + Depth-Anything, or COLMAP-registered cameras). The
avatar is therefore trained on the recon crops composited on black with the given alpha (image * alpha / 255, the same
construction as the scoring GT); ExAvatar's scene Gaussians (initialised by its Depth-Anything route) then model a black
background and all its losses are kept. DECA / mmpose / face fitting / unwrap see the raw crops. Masks: the given alpha
(ExAvatar binarises at 0.5).

### Every loader / code-path change (also in run.json "loader_changes")
1. cam_params from our per-frame K (ExAvatar's virtual-camera loader path, unchanged code reads them).
2. Frames: raw crops for the fitting folders; black-composited recon crops for the avatar folder.
3. Masks: given alpha instead of SAM.
4. smplx_init from the GT instead of Hand4Whole; bbox-based translation guess disabled (GT root position used).
5. `fit_face_gt.py` instead of `fit.py` (body / body-ID frozen at the GT; face fitted as in fit.py; test frames: ID frozen;
   visualisation renders and videos skipped).
6. Smoothing restricted to face params, per split.
7. `avatar/main/config.py` dataset = 'Custom' (README instruction).
8. Test render: animate.py's path with refined Gaussians, black bg, our K, GT body + its face fit; checkpoint loaded with
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch 2.6 default `weights_only=True` rejects ExAvatar's own checkpoint).
9. Environment only: `OMP_NUM_THREADS=MKL_NUM_THREADS=8` (shared 224-core host; DECA and the face fit run ~2x faster than
   with all-core thread pools), env `python` first on PATH (ExAvatar's tools spawn `python ...`).

Hyper-parameters: all defaults (fitting: 3 epochs, batch 64, 500/250/250 iterations per batch; avatar: config.py as shipped).
All recon frames are used (the README's optional 5-fps subsampling of the train list is not applied).

## Camera / convention sanity (sanity_camera.py, all frames of the 2 timing clips)
GT SMPL-X 2020 mesh vs ExAvatar's posed template (HumanGaussian.forward with zeroed offset nets) driven by the converted GT:
5.4 mm (UBody) / 5.7 mm (BEDLAM2) mean vertex error. Silhouette of that template rendered with ExAvatar's own
GaussianRenderer (get_fov / view / projection matrices, our K, identity extrinsics) vs `masks/<key>.png > 127`:
IoU 0.79 (UBody, 1.5 px splats) / 0.92 (BEDLAM2, 0.6 px splats); reference: GT mesh rasterised with our K vs mask
0.77 / 0.94 (masks include hair, clothing, the violin; UBody legs fall outside the original image). Visual overlays aligned.

## Timing / budget
See the report in RUNLOG / the coordinator's summary; per-clip breakdown in each run.json `timing_s`.
