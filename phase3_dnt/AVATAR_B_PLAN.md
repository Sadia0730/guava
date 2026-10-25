# Avatar task B: GUAVA full-body pilot plan (SMPL-X 2020 head, as in GUAVA (a20))

Status: PLAN AND DATA CHECKS ONLY (2026-10-25). Nothing was trained, downloaded, moved or deleted. GPU use: three inference / matting
checks on GPU 5, each under 5 min. All new outputs are under `/raid/ubx858/outputs/dnt/avatar_B/` (scripts in `scripts/`).

## 0. Summary
- **GUAVA's code is not tied to the upper body in its geometry.** The SMPL-X UV map and the vertex Gaussians cover the legs and feet.
  Four things do assume a narrow-FOV upper-body setup: (i) a fixed near-orthographic camera (invtanfov 24, principal point at the
  crop centre); (ii) the training distribution (a20's leg / foot Gaussians are almost transparent); (iii) face and hand resolution at
  512 px; (iv) crop boxes that come from the tracker.
- **BEDLAM2 is the main data source.** It has exact SMPL-X and cameras on disk (raw tars + processed npz). Its masks are not on disk,
  but StyleMatte works on it (checked). After filters there are **10,130 single-subject full-body clips from 1,402 subjects (677k
  labelled frames at 6 fps)**. An identity-disjoint split is proposed: 40 test subjects / clips, a training pool of 1,278-1,310
  subjects, and a pilot subset of 1,500 clips from 933 subjects.
- **Exact BEDLAM2 cameras need a code change.** With GUAVA's fixed camera, the best rigid placement leaves a median 7 px mean vertex
  error (p90 20 px) in a 512 crop. Only about 20 % of rows (camera depth >= 5 m) are below 2 px. Proposed fix: a perspective crop
  plus a per-sample focal length, applied as a monkeypatch in a new fine-tune script.
- **Motion-X labels are not good enough to supervise pixels.** Of 20 random full-body clips, 5 are well aligned (25 %), 9 fair and 6
  poor. Neither silhouette IoU nor DWPose keypoint error separates them. Proposal: no Motion-X in training; filtered Motion-X clips
  only as an extra real-video test set.
- **Estimates:** data prep about 3-5 h (mostly StyleMatte); fine-tune about 4.7 h for 15k iterations (6.3 h for 20k); evaluation
  under 1 h. Disk: about 13-21 GB of data plus 1.3 GB (best.pt), against 585 GB free on /raid.

## 1. Upper-body assumptions in GUAVA (file:line) and what full-body framing needs
Classes: **OK** = works as is · **CFG** = config / data-conversion change only · **CODE** = code change (patched in a new fine-tune
script, like a20's `smplx_only_forward`; existing files untouched).

| # | Where | Assumption | Full-body impact | Class |
|---|---|---|---|---|
| 1 | EHM-Tracker `src/data_prepare_pipeline.py:113-114`, `src/utils/crop.py:281` (scale 1.1), `:247` union box; `src/configs/data_prepare_config.py` (`body_hd_size` 1024, `tanfov` 1/24) | Body crop = square around the DWPose person box x1.1, 1024 px | Already "whatever is visible": full-body videos give full-body crops. We do not run the tracker on BEDLAM; a new converter writes the same square framing from the GT mesh box (x1.1). | OK / CFG |
| 2 | `configs/train/ubody_512.yaml:9-10, 62-64`; `dataset/data_loader.py:84, 88-89` | Render 512², DINO input 518², stored images 1024 (`origin_image_size`) | Works. The face drops to about 55-70 px tall in a full-body 512 render (upper body: about 150-200 px). This is the main quality risk (§6). `origin_image_size` must equal the stored size. | CFG |
| 3 | `ubody_512.yaml:10` invtanfov 24; `data_loader.py:24, 91-97`; `utils/graphics_utils.py:44-50` (centred principal point); `models/UbodyAvatar/ubody_gaussian.py:29, 60-61` (mesh renderer, fixed focal), `:80` (`sample_prj_feature`), `:99` (`convert_pixel_feature_to_uv`); EHM camera `src/modules/renderer/util.py:117` (`cam2persp_cam_fov_body`: fixed FOV, scale through z) | One fixed, almost orthographic camera (tan FOV 1/24, camera about 32 m away for UBody) | Exact for EHM-tracked or Motion-X (OSX, f 6.5-19k px) data. **Not exact for BEDLAM2** (median f 725 px, depth 2.4 m). Measured on 1,000 pilot rows (`bedlam_camera_error.json`): plain crop + best translation, mean vertex error p50 / p90 = 7.5 / 18.9 px (max 45 / 97); perspective crop + best translation 7.1 / 20.0 px. Only 20 % of rows are under 2 px (rows at depth >= 5 m: median 0.8 px). Fix: perspective crop (homography warp, rotation folded into the root, principal point back at the centre) + per-sample `invtanfov` = 2f/L in the 4 places listed. The rasterizer already takes tanfov per sample (`gaussian_render.py:47-48`). | **CODE** (or a depth >= 5 m filter as a no-code fallback) |
| 4 | `data_loader.py:124-138` | Camera = diag(-1,-1,1)·[R\|T] from `camera_RT_params`; body orientation in `global_pose` | The converter writes R = diag(-1,-1,1), T = diag(-1,-1,1)·t', with the vertices in the (rotated) OpenCV camera frame. t' must absorb the pelvis pivot when the look-at rotation is folded into the root. | CFG |
| 5 | `utils/loss_utils.py:98, 101-104` | L1 + LPIPS on GT x StyleMatte mask, black background | OK | OK |
| 6 | `loss_utils.py:106-115, 140-159`; boxes `data_loader.py:143-185` (from tracker `head_crop` / `left_hand_crop` / `right_hand_crop` matrices x `body_crop['M_o2c-hd']`, scaled by image_size / origin_image_size) | Head / hand crop losses use tracker crops (face landmarks x1.75, hand keypoints); fallback to the full image only for the head (`:182-183`) | The converter must synthesise these crop matrices from the projected GT head / hand vertices. A full-body head box is about 60 px, upsampled to 256 for the loss (weak). `cal_box_loss` crashes when every box in a batch is empty (`torch.cat([])`, `loss_utils.py:154`); this is possible for hands out of view, so add a guard. Optional foot-crop loss (feet are about 30 px). | CFG + small **CODE** (guard) |
| 7 | `models/modules/smplx/SMPLX.py:219` (`uv_masks/uv_mask512_with_faceid_smplx.npy`, position map only), `:222-224` (`lbs_map_smplx_512` commented out, unused), `:239-245` (UV map mask from all SMPL-X UV faces); `ubody_gaussian.py:150-156` | UV Gaussians on every textured SMPL-X UV texel | **The legs and feet are represented**: 31,720 leg + 7,067 foot UV Gaussians (19.5 % of 199k) plus 1,378 vertex Gaussians. No code change. | OK |
| 8 | Measured a20 behaviour (`a20_legs/a20_legs_check.json`, renders `a20_legs/*.jpg`) | Trained on UBody, where legs are rarely visible | Leg / foot UV opacity mean 0.02-0.05 (torso 0.16-0.31) when the legs are unseen in the source. Legs reach 0.165 when 14.5 % of leg faces are visible (Olympic clip). Feet are never seen (opacity about 0.02-0.04). Zoomed-out renders lose the legs and the refiner produces blob artifacts. **Needs full-body fine-tuning.** | training |
| 9 | `ubody_gaussian.py:141-143` | Image features reach the UV map only through faces visible in the source frame | Unseen parts (back, soles) rely on `uv_base_feature` / vertex features, as for the back in upper-body use. | OK |
| 10 | Refiner StyleUNet 512 (`ubody_512.yaml:22-31`, `gaussian_render.py:90-91`) | Image-space refiner sized to the 512 render | Works; it is retrained by fine-tuning. A 1024 render would change the StyleUNet depth (no weight reuse), so it is out of scope for the pilot. | OK (CFG for 1024 = not pilot) |
| 11 | `data_loader.py:63-74` / `guava_finetune.py:84-88` | Source = random other frame of the clip (train), frame 0 (val / test) | Works with BEDLAM clips (>= 30 labelled frames). The labels are 6 fps, so every pair is at least 0.17 s apart. | OK |
| 12 | `data_loader.py:105-118`, `guava_finetune.py:97-121` | `flame_coeffs` / `id_share_params` keys, LMDB `<frame>/body_image`, `body_mask` | The converter writes the a20 fields (`smplx_coeffs`: shape 200, exp 50, jaw, eye_pose, body / hand poses axis-angle with flat hand mean = BEDLAM's convention, global_pose, camera_RT_params) plus dummy `flame_coeffs` (eye_pose_params from BEDLAM's eyes, zero expression). | CFG |
| 13 | `guava_finetune.py:163-165` (a20: EHM's FLAME rows reset to the SMPL-X 2020 template, `smplx_only_forward`) | Pure SMPL-X 2020 head | Unchanged. | OK |

## 2. Data feasibility (no downloads)
### 2a BEDLAM2 (primary)
- On disk: 66 `*_mp4.tar` (the expanded copy was deleted; see DISK.md). Members can be read straight from the tar: index 0.4 s and
  extract under 0.1 s for a 5 GB tar. Also on disk: `*_gt_centersubframe_exr_meta(_csv).tar.gz` (camera json/csv and `be_seq.csv`,
  **no image data**), processed per-row labels in `processed/bedlam2_gt{,_multi}/<job>.npz` (pose_cam, shape16, t_cam, K,
  joints2d_img, box, seq, frame; official test sequences already removed) and 512 px context crops. Consecutive labelled frames of
  one person exist (seq + frame, 6 fps).
- **The 512 context crops cannot be reused.** The body takes about 40 % of the crop (about 205 px), so the converter must re-decode
  the video.
- **Masks:** BEDLAM2 segmentation / depth EXRs are listed in `b2_data_depth.txt` / `b2_data_png.txt` but were never downloaded. Only
  the `exr_depth_meta` json is on disk. So we use StyleMatte (`EHM-Tracker/pretrained/matting/stylematte_synth.pt`, the EHM / GUAVA
  matting model). A sample on a snow HDRI clip gives a clean full-body matte (`prep_sample/`).
- **Identity:** `be_seq.csv` → `Body` = `<country>_<subject>_<size>_<motion>` (e.g. `us_0228_M_2000`). Subject = `us_0228`. Verified:
  1 SMPL-X shape per subject id (1,514 subjects, 0 conflicts). All 822 MOYO bodies share 1 shape, so they count as one subject
  "moyo". Appearance = texture_body (150) + texture_clothing (4,101 outfit textures) + hair, randomised per sequence.
- **Counts** (census `bedlam_census.json`, script `scripts/bedlam_census.py`). "Full-body" row = all body, face and foot joints inside
  the image (1 % margin) and >= 90 % of hand joints inside.

| filter | clips | subjects | labelled frames |
|---|---|---|---|
| single-person sequences (all) | 16,859 | 1,514 | 1,081,932 |
| >= 30 full-body rows | 12,049 | 1,478 | 856,304 |
| >= 60 full-body rows | 6,815 | 1,293 | 640,503 |
| >= 30 full-body rows + median body height >= 350 px (**pool**) | **10,130** | **1,402** | **677,134** |
| same, >= 400 px | 8,057 | 1,379 | 535,345 |

  Body height (px) p10 / p50 / p90 = 307 / 428 / 569 (1280x720 landscape 89 %, 720x1280 portrait 11 %). Multi-person jobs add 8,779
  person tracks (linkable by shape16), but they are not used: people occlude each other and there is no mask per person.
- Of the pilot clips, 760 are rendered with shoes and 740 barefoot (`renderjob.shoes`). **All test clips are barefoot** (the val jobs
  are barefoot), so feet results reflect bare feet.

### 2b Motion-X (secondary; 20-clip audit)
- 20 random clips (seed 0; 5 each from idea400 / humman / kungfu / perform) among clips with >= 80 % full-body rows and >= 3 stride-12
  rows. Full-body clips per subset: idea400 11,471 / 11,555, humman 751 / 908, kungfu 919 / 1,029, perform 578 / 914.
- The overlays use exactly the builder's mapping: label n → decoded frame n-1 (`frame_map`), OSX crop camera, label hand poses
  absolute. 4 frames per clip, full-body square crop, top raw / bottom mesh plus outline. Location:
  **`/raid/ubx858/outputs/dnt/avatar_B/motionx_overlays/`** (20 jpg + `overlay_scores.json` + `autoscore.json`).
- **Visual grades** (well aligned = limbs on the body within a few px at 512, usable for pixel losses):
  - good, 5: humman Reverse_the_rowing, Standing_of_knee, The_leg_up_L; idea400 Rub_your_hands; perform after_getting_drunk_v2.
  - fair, 9 (one limb or one frame off, or the silhouette too thin or shifted): Sma_vertical_stretch (lower legs), Trapezius_stretch,
    walking_and_Crying, Buying_Tickets, Straight_Dragon_Sword, Tsunami_Kick, clothing_clip22, stand_on_bus (best offset -2 frames:
    the re-indexed clip still lags), waiter.
  - poor, 6: Crawling_and_standing, sitting_and_Cutting, Butterfly_Kick, Flying_Kicks, Tai_Chi step27 (legs wrong),
    take_off_shoes.
  - **Well aligned: 5 / 20 = 25 % (Wilson 95 % CI 11-47 %); usable with tolerance (good + fair): 14 / 20 = 70 %.**
- **Automatic scores tested** (`scripts/motionx_autoscore.py`; StyleMatte on GPU 5; DWPose on CPU because onnxruntime-gpu lacks
  cuBLAS 11):
  - Silhouette IoU (mesh vs matte) is 0.39-0.74 for every clip. Clothing and hair dominate, so it does not separate the grades. The
    best 2D shift search found |dx|, |dy| <= 0.02 of body height, so the errors are limb / pose errors, not global offsets.
  - DWPose-vs-SMPL-X keypoint median error is 0.7-4.5 % of body height. It does not track the grades either: the cleanest humman
    floor clips score worst because DWPose drops side-view keypoints. Neither metric works as a filter alone.
- **Proposed automatic filter (not yet validated):**
  - per frame: (a) mesh-outside-matte fraction (silhouette precision, robust to loose clothing) < 5 %; (b) per-limb **max**
    DWPose-to-joint error < 3 % of body height on keypoints with score > 0.5, needing >= 15 such keypoints;
  - per clip: >= 80 % of frames pass;
  - calibrate the thresholds on these 20 graded clips plus 30 more, and require precision >= 0.9 for "good".
- **Recommendation:** no Motion-X in training for the pilot. Use about 20 clips that pass the filter and a visual check as a real-video
  test set; they are identity-disjoint from the BEDLAM-only training data by construction. Note that all 5 humman clips show the
  same performer. The other route, EHM-Tracker on Motion-X (GUAVA-native labels), costs about 11 min per clip (RUNLOG 4: 33 clips
  in 5.9 h), so it is out of scope for the pilot.

## 3. Identity-disjoint split (BEDLAM2; `bedlam_split_proposal.json`, script `scripts/bedlam_split.py`)
- **Test:** 40 subjects / 40 clips (one clip per subject, most full-body rows), 2,931 labelled frames (30-96 per clip, median 81).
  They come only from the SPRINT / PEAR-student BEDLAM2 **val jobs** (21 from `20240731_1_1827_hdri`, static HDRI; 19 from
  `20240808_1_250_ai1105_vcam`, moving camera in a scene), so a SPRINT-driven test is also fair. Subjects are drawn by SHA-1 hash;
  MOYO is excluded.
- **Training pool:** subject-disjoint (and no val-job clips) gives 8,770 clips / 1,310 subjects / 586,629 frames. Strict variant
  (also removes every outfit texture used by a test clip) gives 7,773 clips / 1,278 subjects / 520,123 frames. The **strict variant**
  is proposed.
- **Pilot training subset:** at most 2 clips per subject (8 for "moyo"), preferring body height >= 400 px, capped at 1,500 clips:
  **1,500 clips, 933 subjects, 97,953 labelled frames** (516,920 video frames).
- **Validation (best.pt selection):** 10 more clips from the val jobs, from subjects in neither test nor train. To be drawn with the
  same hash rule at conversion time.
- **Upper-body regression check:** a20's own 10 UBody validation clips (347 frames), so the full-body model can be compared with a20
  where a20 is strong.

## 4. Pilot plan
### 4.1 Conversion to GUAVA's tracked format (new script, e.g. `phase3_dnt/avatar_b_convert.py`)
Output: one directory per clip, like `c2t/tracked/<clip>` (`optim_tracking_ehm.pkl`, `id_share_params.pkl`, `videos_info.json`,
`img_lmdb`), so that `guava_finetune.py`'s `MultiClip` loader (or a copy of it) reads it unchanged.
1. **Frames:** stream `<job>_mp4.tar` member → temp mp4 → cv2 decode, labelled frames only (6 fps). Cap at 40 rows per clip
   (spread), which gives 60k rows.
2. **Framing:** perspective crop around the GT mesh box centre (virtual camera rotated onto the ray; homography warp), square side
   1.1 x max box side, 1024 px stored (`origin_image_size` 1024; 512 would save about 70 % of the disk).
3. **Mask:** StyleMatte alpha on the 1024 crop (as EHM does: `data_prepare_pipeline.py:127-129`).
4. **Body:** pose_cam (locked head, flat hand mean = SMPL-X convention in `smplx_only_forward`). Root rotated by the look-at rotation;
   translation recomputed about the pelvis joint. **Shape:** shape16 → SMPL-X 2020 betas200 with
   `/raid/ubx858/outputs/phase1_task1/checkpoint2/betas_locked16_to_2020_200.npz` (M, c; the map SPRINT uses). It is per subject and
   constant per clip, so it goes in `id_share_params.smplx_shape`. **Face:** BEDLAM has no expression: exp50 = 0, jaw = pose_cam
   66:69, eyes = pose_cam 69:75 (copied to `flame_coeffs.eye_pose_params`, eye_copy as a20). For Motion-X test clips use the
   `labels_X` exp50 / jaw from `tools/build_face_labels_smplx2020.py` (motionx_gt source).
5. **Camera:** `camera_RT_params` = diag(-1,-1,1)·[I | t'] and per-row `invtanfov` = 2 f / L (perspective-crop focal / crop side).
   It is stored in a new field (`tanfov`) that the new fine-tune script's patch reads (§1 #3).
6. **Crop boxes:** `body_crop` M (identity at 1024), `head_crop` / `left_hand_crop` / `right_hand_crop` M_c2o from the projected
   head (SMPL-X–FLAME ids) / MANO vertex boxes x1.75 / x1.2, as the tracker does.
7. **Checks before training:**
   - re-render the converted SMPL-X through GUAVA's own `mesh_renderer` / rasterizer camera on 50 rows: mean vertex error < 1 px at
     512 vs the GT projection;
   - mask-vs-mesh precision >= 0.95;
   - 20 visual overlays.

### 4.2 Fine-tune recipe (new script, e.g. `phase3_dnt/guava_fullbody_finetune.py`, derived from `guava_finetune.py --variant a20`)
- **Init:** a20 `best.pt`. **Changes:** the 4 per-sample-focal patches and the empty-box guard. Loss and optimizer are GUAVA's
  (`Optimization_Loss`, `configure_optimizers`, Adam betas (0, 0.99)) with the iteration offset 160k as in a20.
- **Data:** BEDLAM pilot 1,500 clips. Proposed mix: 75 % BEDLAM full-body / 25 % a20's UBody training clips (same converted a20
  data) against forgetting upper-body and face detail. Pre-register this as one run; the pure-BEDLAM ablation is optional.
- **Schedule:** 15k iterations (ceiling 20k), batch 6, lr 5e-5 → 0.1x linear, validation every 1k on the 10 BEDLAM val clips plus
  the UBody val clips, best.pt by mean SSIM over both. a20 peaked at 2k, so validate densely. latest.pt is deleted after a successful
  run (keep best only).
- **Losses:** unchanged weights; optional `lambda_head_crop` 0.25 → 0.5 because faces are small. Pre-register either way.

### 4.3 Evaluation (held-out, identity-disjoint)
- **Data:** 40 BEDLAM test clips (2,931 frames) and the filtered Motion-X real-video clips (target about 20; 4.3 frames per labelled
  second at stride 12).
- **Protocol:** GUAVA's own. Source = frame 0, every frame a target, GT x mask on black. Metrics: PSNR / SSIM / LPIPS(vgg) on the
  full 512 image, plus **face-crop and hand-crop LPIPS** (boxes from the projected GT head / hand vertices, resized to 256) and an
  optional foot crop.
- **Comparisons:** pilot vs **GUAVA (a20)**, both driven by the same GT params. Optionally also SPRINT-driven: test images come from
  SPRINT val jobs.
- **Also report:** a20's UBody val (upper-body regression).
- **Visuals:** 10 side-by-side renders (source | GT | a20 | pilot), 6 BEDLAM + 4 Motion-X, chosen by fixed seed before looking at
  the results.

### 4.4 Time estimates (measured where possible)
| step | basis | estimate |
|---|---|---|
| Decode + crop (CPU) | 0.8 s per clip of 348 frames (`prep_sample`), tar index 0.4 s | 1,540 clips × about 1.5 s ≈ **40 min** (4-8 workers, nice) |
| SMPL-X forward, boxes, pkl writing (CPU) | small batches, CPU | about 30 min |
| StyleMatte 1024 (GPU 0 / 2 / 5, shared) | **5.8 fps** measured on busy GPU 5 | 60k rows ≈ **2.9 h** (98k rows: 4.7 h); about half that on an idle GPU |
| Conversion checks | | about 20 min |
| Fine-tune | a20 measured **0.887 it/s** (batch 6, 512, L40S GPU 5; `guava_a20/run/train_log.jsonl`, 20k in 6.3 h), same render size | 15k ≈ **4.7 h**, 20k ≈ 6.3 h (+ about 3 min per validation) |
| Evaluation | inference about 10 frames/s incl. LPIPS (a20 val: 347 frames per validation) | 2,931 + about 1.5k frames × 2 models ≈ **20-30 min** |
| **Total** | | **about 9-12 h wall clock**, about 3.5 h of it before training can start |

### 4.5 Disk estimates (/raid: **585 GB free**, 95 % used; rule: keep >= 30 GB)
| item | basis | size |
|---|---|---|
| Pilot images + masks in LMDB | 173 KB jpg q95 + 37 KB png per 1024 frame | 60k rows **12.6 GB** (98k: 20.6 GB; 512 storage: about 4-6 GB) |
| Test + val + Motion-X test | about 5k rows | about 1.1 GB |
| pkl / json | about 17 KB per frame (c2t: 4.2 MB per 250 frames) | about 1 GB |
| Checkpoints | best.pt 1.30 GB (a20); latest.pt transient 1.3 GB | 1.3 GB kept |
| Visuals, eval renders | | < 2 GB |
| **Total** | | **about 17-26 GB**; /raid stays above 550 GB |

## 5. Small checks done (all under `/raid/ubx858/outputs/dnt/avatar_B/`)
| check | output | GPU |
|---|---|---|
| BEDLAM2 census, identity, split | `bedlam_census.json`, `bedlam_split_proposal.json` | none |
| BEDLAM2 camera vs GUAVA fixed camera (1,000 rows) | `bedlam_camera_error.json`, `bedlam_camera_error_rows.npz` | none |
| Motion-X 20 overlays + grades + auto scores | `motionx_overlays/` | 5 (StyleMatte, about 3 min) |
| a20 legs / feet representation | `a20_legs/a20_legs_check.json`, `a20_legs/*.jpg` | 5 (inference, about 2 min) |
| Data-prep throughput + StyleMatte sample | `prep_sample/` | 5 (about 1 min) |

## 6. Risks
1. **Face and hand resolution.** A full-body 512 render gives about 60 px faces and about 30 px hands. Face-crop LPIPS will likely be
   worse than a20 on upper-body framing. Mitigations: UBody mixing, a higher head-crop weight; later a 1024 refiner (architecture
   change).
2. **Synthetic-to-real gap.** BEDLAM2 renders use simulated clothing and HDRI lighting, and some hair textures cover the face (seen in
   the sample). Real-video generalisation is measured only on the small Motion-X test set.
3. **Camera code change.** If the per-sample-focal patch is wrong, renders are silently misaligned; hence the < 1 px re-render check.
   Fallback without a code change: rows at depth >= 5 m (about 20 %, median error 0.8 px).
4. **Masks.** StyleMatte can include props or shadows and loses hair or fingers. BEDLAM's own segmentation masks are not on disk (no
   downloads allowed).
5. **Motion-X alignment.** Only 25 % of clips are well aligned and the automatic filter is untested. The real-video test set may end
   up small (about 20 clips) and biased toward easy, frontal, standing clips.
6. **Identity leakage through appearance.** A subject id fixes body shape only; outfits and skin textures repeat across subjects.
   The strict split removes test outfits, but skin textures (150) and hair still overlap.
7. **Forgetting.** a20's best validation was at 2k iterations, so longer full-body training may degrade upper-body / face quality.
   Watch the UBody val track.
8. **Feet.** The test set is all barefoot, while the shoe appearance in training comes only from BEDLAM's shoe jobs.
9. **Labels at 6 fps only.** Source / target pairs are fine; temporal metrics are not available.
