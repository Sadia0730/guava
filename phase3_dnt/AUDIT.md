<!-- B0 audit, read-only, 2026-10-15. Written by the audit agent; (I) = inferred. Saved at phase3_dnt/AUDIT.md and /raid/ubx858/outputs/dnt/AUDIT.md -->

# GUAVA audit (fork at `/home/ubx858/guava`, branch `driven_not_tracked`, HEAD `785e3922`)

Marks: (I) = inferred, not confirmed by running code. Everything else was read in the code.

## 0. Environment
- The GUAVA rasteriser and the GPU build of PyTorch3D exist only under `/tmp`. This is documented in `/home/ubx858/guava/phase1/task3/DESIGN.md:310-311` and repeated in `/home/ubx858/guava/phase2/G0_GUAVA_RELEASE.md:40,148`.
- Command template (`DESIGN.md:328-329`): `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib`.
- `/home/ubx858/guava/LIVE_PEAR_SETUP.md:52-74` gives build instructions for PyTorch3D v0.7.7 and `submodules/diff-gaussian-rasterization-32`. It contains no `/tmp` PYTHONPATH.
- Weights: `assets/GUAVA/checkpoints/best_160000.pt`. Config: `assets/GUAVA/config.yaml`, identical to `configs/train/ubody_512.yaml`.

## 1. EHM forward
**File:** `/home/ubx858/guava/models/modules/ehm/EHM.py`

**`EHM.__init__(flame_assets_dir, smplx_assets_dir, n_shape=300, n_exp=50, with_texture=False, uv_size=512, check_pose=True, add_teeth=False)` (lines 14-33)**
- Builds SMPL-X and FLAME (MANO is commented out).
- Builds a combined `v_template`: the FLAME template is placed into the SMPL-X head vertices (`smplx2flame_ind`), aligned on the eye joints (FLAME joints 3:5 to SMPL-X joints 23:25).
- Builds a Laplacian.

**`EHM.forward(body_param_dict, flame_param_dict=None, mano_param_dict=None, static_offset=None, zero_expression=False, zero_jaw=False, zero_shape=False, pose_type='rotmat')` (lines 35-154)**
- **FLAME branch (40-73):**
  - Inputs: `shape_params` (B,300), `expression_params` (B,50), `jaw_params` (B,3), `eye_pose_params` (B,6), `eyelid_params` (B,2).
  - Global pose and neck pose are forced to zero (58-59).
  - Runs FLAME LBS, adds eyelid blendshapes, then multiplies by `body_param_dict['head_scale']` (B,3).
- **Body branch (79-135):** consumes these `body_param_dict` keys:
  - `shape` (B,≤300): the tracker gives 200, padded to 300.
  - `exp` (B,50).
  - `global_pose` (B,1,3), `body_pose` (B,21,3), `left_hand_pose` / `right_hand_pose` (B,15,3).
  - `joints_offset` (B,55,3).
  - `hand_scale` (B,3).
- **Poses are axis-angle.** `lbs_wobeta` defaults to `pose2rot=True` (`flame/lbs.py:255`). The `[1,21,3,3]` comments at lines 81-86 are out of date, and `pose_type` is unused.
- **Body jaw and eye are zeroed (96-97).** Jaw, eye and eyelid act only through the FLAME head, which replaces the SMPL-X head vertices in the shaped template *before* body LBS (119-122). Line 120 is dead code.
- `joints_offset` is added twice by design: to the rest joints (line 116, for head placement) and inside `lbs_wobeta` (`flame/lbs.py:295`).
- `hand_scale` scales the MANO-indexed vertices about the hand centres (124-130).
- **Returns:**
  - `vertices` (B,V,3).
  - `joints`: rest joints including the offset.
  - `joints_transform`: posed joints.
  - `ver_transform_mat`: T = W·A, (B,V,4,4), the per-vertex blended LBS matrix (`flame/lbs.py:323-324`).
  - `joint_transform_mat`: A, (B,55,4,4).
  - Head, left-hand and right-hand vertices with their reference joints.
- V = 10,475 SMPL-X vertices plus the teeth vertices (`add_teeth=True`, `SMPLX.py:429`), about 10.5k (I).
- There is no global translation. Translation lives in the camera (`camera_RT_params`).

**`get_transform_mat(...)` (156-207):** an alternative that returns the LBS transforms only. It reads MANO rotation matrices. The forward pass never calls it.

**How Gaussian deformation uses EHM** (`ubody_gaussian.py:222-243`):
- Template Gaussians use `ver_transform_mat[:,:,:3,:3]`, converted to a quaternion with `roma` (225-226). T is a blended matrix, not a pure rotation, so `rotmat_to_unitquat` projects it (I).
- UV Gaussians use triangle frames from `compute_face_orientation` (`/home/ubx858/guava/utils/graphics_utils.py:61-80`): edge-based orthonormal axes plus scale = (|v1−v0| + height)/2.

**Hook here:** replace or perturb `smplx_coeffs`/`flame_coeffs`, `static_offset` (per-vertex canonical offsets), `joints_offset`, `head_scale`/`hand_scale`. All of these are differentiable through LBS.

**Bug:** `Ubody_Gaussian.init_ehm(ehm=None)` (`ubody_gaussian.py:196`) passes `mano_assets_dir` as the third positional argument, which is `n_shape`. Every caller passes `infer_model.ehm`, so the bug is never triggered.

## 2. Template and UV Gaussian decoders (`Ubody_Gaussian_inferer`)
**File:** `/home/ubx858/guava/models/UbodyAvatar/ubody_gaussian.py`. Class at line 20, `forward(self, batch)` at 116-159.

**DINOv2 path** (`/home/ubx858/guava/models/modules/net_module/dino_encoder.py:7-113`, `DINO_Enocder.forward(images, output_size=None, head_images=None)`):
- Uses a frozen `dinov2_vitb14`. Input: the masked source image at 518² (`feature_img_size`), giving a 37×37 patch grid.
- A DPT-style fusion of 4 intermediate layers plus a low-level layer produces:
  - `f_map1`: 32 channels (`dino_out_dim`) at 512².
  - `f_map2`: 128 channels (`prj_out_dim`) at 512².
  - `f_global`: the 768-d CLS token.

**Template branch (lines 124-134):**
- EHM is run with the source parameters (128). Posed vertices are projected with `w2c_cam` and `invtanfov=24` (`sample_prj_feature`, 75-83) to sample `f_map2`.
- The input per vertex is the concatenation of the sampled feature (128), a learned per-vertex `vertex_base_feature` (128) and the MLP-mapped global feature (256), 512 in total.
- `Vertex_GS_Decoder.forward(input_features, cam_dirs)` (`/home/ubx858/guava/models/modules/net_module/feature_decoder.py:9-62`) is an MLP with a 27-d harmonic camera-direction embedding. Outputs:
  - `colors` (B,V,32);
  - `opacities`: sigmoid;
  - `scales`: sigmoid×0.05, an absolute world scale;
  - `rotations`: normalised quaternion;
  - `static_offsets=None`.
- `positions` is set to the canonical `v_template` (134).

**UV branch (lines 137-157):**
- RGB (3) is concatenated with `f_map1` (32), giving 35 channels, then inverse-texture-mapped to UV (§3).
- `uv_feature_decoder` is a `StyleUNet(in_size=512, out_size=512, activation=False, in_dim=35, out_dim=96, extra_style_dim=512)` (line 55). Its `extra_style` input is the CLS token through `uv_style_mapping`.
- The result is concatenated with a learned `uv_base_feature` (32,512,512).
- `UV_Point_GS_Decoder.forward(input_features, cam_dirs)` (`feature_decoder.py:64-137`) is a conv head. Outputs (B,512,512,·):
  - `colors` (32);
  - `opacities`: sigmoid;
  - `scales`: exp, relative to the triangle scale;
  - `rotations`;
  - `local_pos` (3).
- Outputs are flattened and masked by `smplx.uvmap_mask` (150-152). Each output gets `binding_face` and `face_bary`.
- `extra_dict['uvmap_texture']` = sigmoid of the first 3 UV channels.

**Counts:** about 10.5k template Gaussians and about 190k UV Gaussians. These numbers come from `G0_GUAVA_RELEASE.md:106-107`; I did not recompute them.

**Pruning** (`Ubody_Gaussian.prune_gaussians`, 201-215):
- Applies only to UV Gaussians, keeping those with opacity > `opacity_threshold` = 0.001. It requires B=1.
- `test.py` uses `pruning=True`. Training uses `pruning=False` (`trainer.py:84`).

**Side effect:** `Ubody_Gaussian.__init__` applies sigmoid in place to colour channels 0:3 (186-187). This mutates the inferer's output tensors, so building two `Ubody_Gaussian` objects from the same dicts applies the sigmoid twice (I).

**Hook here:** the returned dicts (`colors`/`opacities`/`scales`/`rotations`/`local_pos`), or feature injection at `vertex_sample_feature` / `uvmap_features`.

## 3. Inverse texture mapping and visibility
**`convert_pixel_feature_to_uv(img_features, deformed_vertices, w2c_cam, visble_faces=None, uv_features=None)`** (`ubody_gaussian.py:85-114`):
- Each 512² UV texel has a precomputed face index and barycentric coordinates:
  - `smplx.uvmap_f_idx`, `uvmap_f_bary` and `uvmap_mask = (f_idx != -1)` are built at `/home/ubx858/guava/models/modules/smplx/SMPLX.py:239-245`;
  - via `get_uvmap_faces_index` (line 795) and `get_uvmap_faces_barycoord` (line 803).
- The 3D point of each texel on the posed source mesh is projected (x,y·24/z, already in NDC) and the image features are `grid_sample`d with zeros padding.
- Output: (B,35,512,512), multiplied by the mask.

**Visibility** (lines 140-143, under `no_grad`):
- `BaseMeshRenderer.render_fragments(vertices, cameras=None, transform_matrix=None, faces=None, reverse_camera=True)` (`/home/ubx858/guava/utils/graphics_utils.py:471-492`).
- It runs a PyTorch3D rasteriser at 512² with `faces_per_pixel=1`. It reapplies diag(−1,−1,1,1) to return to the PyTorch3D convention and returns `pix_to_face` (B,512,512,1, packed indices).
- `torch.isin` turns this into a per-face visibility flag, which is ANDed with the UV mask.
- Visibility is per face, not per texel (I).

**Hook here:** `visble_faces` / mask for multi-view fusion (accumulate UV features and masks across frames), and `uv_features` (an initial UV map argument exists but is overwritten at line 100).

## 4. Per-frame deformation: `Ubody_Gaussian.forward(self, batch)`
**Location:** `ubody_gaussian.py:217-262`.

**Inputs:** only `batch['smplx_coeffs']` and `batch['flame_coeffs']`. `static_offset` is the template `static_offsets`, which is None.

**Template Gaussians:**
- xyz = EHM posed vertices. Note that `_smplx_xyz` is not used for posing.
- rot = q(T) ⊗ q.
- Scale is unchanged.

**UV Gaussians:**
- xyz = R_face · local_pos · s_face + bary-interpolated point.
- rot = q_face ⊗ q.
- scale = scale · s_face.

**Output keys:** `xyz`, `rotation` (wxyz), `scaling`, `opacity`, `features_color` (32), `sh_degree`, `smplx_xyz_deform`, `joints`.

There is no learned network per frame. `get_canoical_gaussians` (264-286) poses the UV Gaussians on the canonical template.

**Hook here:** the per-frame pose and expression dicts, and the stored tensors `_uv_local_xyz`, `_smplx_features_color`, etc. These are plain tensors, not `nn.Parameter`.

## 5. StyleUNet neural refiner
**Wrapper:** `/home/ubx858/guava/models/UbodyAvatar/gaussian_render.py`
- `GaussianRenderer.__init__(cfg)` (9-14) builds `Nueral_Refiner_Model['styleunet'](**cfg.styleunet)`, stored as `self.nerual_refiner`.
- `forward(*args, **kwargs)` calls `forward_nueral_refine(gaussian_assets, cam_params, bg=None, scaling_modifier=1.0, antialiasing=False)` (93-104). This runs `forward_raw` then `refine_raw`, puts the result in `renders` and deletes `raw_features`.
- `refine_raw(raw_results)` (90-91) is `self.nerual_refiner(raw_results["raw_features"])`. It does not pass `randomize_noise` or `extra_style`.

**Module:** `/home/ubx858/guava/models/modules/net_module/styleunet/styleunet.py:86-207`
- `StyleUNet.__init__(in_size, out_size, in_dim, out_dim, num_style_feat=512, num_mlp=8, activation=True, channel_scale=1, small=False, extra_style_dim=-1)`.
- `forward(x, randomize_noise=True, extra_style=None)`.
- Config used: in_size=out_size=512, in_dim=32, out_dim=3, `small=True`, no extra_style.
- `activation=True` means the output passes through a sigmoid.
- Input: the (B,32,512,512) raster. Output: RGB (B,3,512,512).
- UNet encoder, then a style code, then `StyleGAN2GeneratorCSFT_small` with SFT conditions.

**Noise:**
- The default `randomize_noise=True` draws fresh noise on every call (`StyleConv.forward`, 504-517), so the output is not deterministic from frame to frame.
- `randomize_noise=False` reads buffers of size `2**((i+5)//2)` (lines 353-356). This does not match the small generator, which has one conv per resolution. The failure is noted in `phase2/g1_live_gap.py:12,103-106`, which works around it by seeding.

**Hook here:** `forward_raw` / `refine_raw` split; the refiner can be swapped or fine-tuned.

## 6. Rasteriser call
**Location:** `gaussian_render.py:19-88`, `forward_raw(gaussian_assets, cam_params, bg=None, scaling_modifier=1.0, antialiasing=False)`.

**Loop:** one rasteriser call per batch item.
- Settings: `GaussianRasterizationSettings` with H/W, tanfovx/y = 1/24, `bg` = a 32-vector filled with the scalar `bg`, `viewmatrix=world_view_transform`, `projmatrix=full_proj_transform`, `sh_degree=0`, `campos=camera_center`, `antialiasing`.
- Call: `GaussianRasterizer_32(means3D, means2D=zeros (retain_grad), shs=None, colors_precomp=features_color (32), opacities, scales, rotations, cov3D_precomp=None)`.

**Returns:**
- `raw_renders` = channels 0:3;
- `raw_features` = all 32 channels;
- `viewspace_points`, `radiis`;
- `depths`: inverse depth, from `invdepths`;
- `extra_renders` = channel 3.

**Matrices** come from `get_full_proj_matrix(w2c, tanfov)` (`graphics_utils.py:44-50`, transposed in the 3DGS convention, z-near 0.01, z-far 100).

**Gradients** (`/home/ubx858/guava/submodules/diff-gaussian-rasterization-32/diff_gaussian_rasterization_32/__init__.py:44-139`):
- Gradients flow to means3D, means2D, colors, opacities, scales and rotations.
- Camera matrices get no gradient.
- `NUM_CHANNELS 32` (`cuda_rasterizer/config.h:15`).
- `DESIGN.md` states that the backward pass is non-deterministic because of atomic adds.

## 7. Evaluation scripts and data format
**`/home/ubx858/guava/main/test.py`**
- `render_set` (37-116), self-reenactment:
  - The source is frame 0 (`_load_source_info`, key_idx=0).
  - Targets are `frames[-test_num:]`, with bg = 0.
  - GT is written as image·mask.
  - Outputs go to `render/`, `gt/` and optionally `raw_render/` (`--save_coarse_render`), plus `speed_info.json`.
- `render_cross_set` (118-210): identity is swapped via `change_id_info` (shape, joints_offset, head/hand scale, FLAME shape). Options `--keep_source_cam` / `--canonical_camera`. No GT is written.
- `render_novel_views` (212-250) and `render_static_novel_views` (252-291) use `generate_novel_view_poses` (`utils/camera_utils.py:72`).
- `test()` (294-353): checkpoints are loaded with `strict=False`.
- **Default `test_full=True`:** all frames are evaluated, including the source frame 0, unless `--non_test_full` is passed and a `testing_split.json` exists.

**`/home/ubx858/guava/main/metrics.py`**
- `evaluate(model_paths)` (84-183) runs over `<test_dir>/<scene>/{render,gt}`. Inputs are 8-bit PNGs.
- Metrics:
  - PSNR over the whole image;
  - SSIM with an 11-px Gaussian window;
  - LPIPS-VGG;
  - MAE.
- **No mask is applied in the metric itself.** Masking only enters through the black-background GT composite.
- Scores are averaged per scene, then frame-weighted across scenes.

**`/home/ubx858/guava/main/metrics_face.py`:** insightface ArcFace cosine similarity between each source image and every render under `<render_folder>/<source_name>/*/render`.

**`/home/ubx858/guava/dataset/data_loader.py`**
- `TrackedData_infer` (203-304) reads `testing_split.json` if present; otherwise it uses `frames_num`.
- Files read:
  - `optim_tracking_ehm.pkl` (`[video][frame]` → `smplx_coeffs`, `flame_coeffs`, `left/right_mano_coeffs`, `body_crop`, `head_crop`, `*_hand_crop`, `dwpose_*`, landmarks);
  - `id_share_params.pkl` (`smplx_shape` (1,200), `flame_shape` (1,300), `head_scale` (1,3), `hand_scale` (1,3), `joints_offset` (1,55,3), `*_mano_shape`);
  - `videos_info.json`.
- LMDB keys: `{video}/{frame}/body_image` and `body_mask`, as JPEG/PNG (`utils/lmdb.py:33-39`).
  - Single-image example sets (`assets/example/tracked_image/*`) have no video level, so the keys are `{frame}/body_image`. Lines 253-293 handle this case.
- Camera (124-138): `w2c = diag(-1,-1,1,1) @ [camera_RT_params; 0 0 0 1]`, `c2w = inv(w2c)`, tanfov = 1/24.
- Sizes:
  - feature source image: masked, resized to 518;
  - target image and mask: 512;
  - LMDB originals: 1024 (`origin_image_size`).
- `load_canonical_render_prams` (377-395) defines a fixed camera with t = (0, 0.6, 22).

## 8. Released test split
**None is released.**
- No `testing_split.json` and no list of test identities exist anywhere in the repo, configs, assets or docs.
- `dataset_frames.json` (train/valid) is produced by EHM-Tracker's `src/split_dataset.py` / `build_lmdb_dataset.py:67-70`, not shipped.
- The README covers only demos and training on your own data.
- `phase2/G0_GUAVA_RELEASE.md:66-72` confirms that the paper's 58-ID test set and the 10×8 cross-reenactment set are not released.
- Only 3 example images ship (`assets/example/tracked_image/{NTFbJBzjlts__047, vicox_14_right, vicox_23_right}`, 1 frame each). `tracked_video/` is not present.
- `phase2/build_ubody_intra_test.py` is a fork-made UBody split for mesh recovery (OSX protocol). It is not a GUAVA rendering split.

## 9. Training (reference only)
**`/home/ubx858/guava/main/train.py:17-77`:** `train(config_name, base_model, devices, debug)` builds the dataset with `build_dataset` and uses Adam with betas (0, 0.99), lr 1e-4, a LinearLR schedule to 0.03× over 200k iterations (`ubody_gaussian.py:395-430`), batch 6.

**`/home/ubx858/guava/main/trainer.py`**
- `run_fit` (60-137): the source frame is chosen at random from the same video; the model is not pruned.

**`/home/ubx858/guava/utils/loss_utils.py`**
- `Optimization_Loss.forward` (89-138) applies these losses to both `renders` and `raw_renders`:
  - L1 (weight 1);
  - LPIPS-Alex (0.025, rising to 0.05 after 10k iterations);
  - head crop L1+LPIPS at 256² (×0.25);
  - each hand crop (×0.1).
- It also adds:
  - the `local_pos` norm hinge at 3.0 (×0.01);
  - the UV scale hinge at 0.6 (×1).
- For the first 1000 iterations the render is masked by the GT mask.

**Bug:** `trainer.py:143` reads `self._meta_cfg.MODEL.bg_color`, which no config defines. `add_extra_cfgs` (`utils/general_utils.py:65-73`) only adds `with_smplx_gaussian`. The config is read-only, so `run_val` fails at the first check (iteration 10k). The fix is to add `bg_color: 0.0` under MODEL.
