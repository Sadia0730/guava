# GUAVA-FB shared data format (all converters MUST follow this; spec of PREREG "GUAVA-FB", 2026-10-28)

GUAVA-FB = GUAVA extended to the full body (credit GUAVA). Pure SMPL-X 2020 (assets/SMPLX/SMPLX_NEUTRAL_2020.npz: 200 shape, 50 exp).

## Locations
- O = /vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb   (user decision: /raid is near full -> write NOTHING new on /raid)
- clips:   O/data/<source>/<clip>/          source in {bedlam2, motionx, dress4d, ubc}; UBody = a20's existing data (read-only)
- index:   O/data/<source>/index.json       {"source", "licence", "clips": [{"name", "subject", "split": "train"|"val"|"test", "frames", "views"}], "notes"}
- reports: O/reports/<source>/report.json + overlays/*.jpg (>= 10 overlays: top row image crop, bottom row converted mesh outline/overlay)
- scripts: /home/ubx858/guava/phase3_dnt/guava_fb/<name>.py  (do NOT git commit; the coordinator commits)
- python:  /home/ubx858/miniconda3/envs/guava/bin/python with env (as guava_a20/run_chain.sh):
  PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims

## Clip directory = GUAVA tracked format exactly as phase3_dnt/sprint_avatar_fit.py writes it (read that file's output section), i.e.
- optim_tracking_ehm.pkl {frame_key: record}; id_share_params.pkl (smplx_shape (1,200) = THE ONE shape of this clip's subject
  (shape lock: identical for every clip of the same subject), joints_offset zeros, head/hand scale 1, flame_shape zeros);
  videos_info.json; img_lmdb/ (utils/lmdb.LMDBEngine) keys `<frame_key>/body_image` and `<frame_key>/body_mask` (1024 x 1024 crop).
- record["smplx_coeffs"]: global_pose, body_pose (21), left/right_hand_pose (15) axis-angle, exp (50), jaw_pose (1,3), eye_pose (2,3),
  camera_RT_params (3,4) -- same conventions as sprint_avatar_fit.py, which is verified to render correctly with GUAVA (a20)
  (phase3_dnt/guava_finetune.py --variant a20: smplx_only_forward, SMPL-X 2020 template). Dummy flame_coeffs as sprint_avatar_fit.
- record body_crop / head_crop / left_hand_crop / right_hand_crop matrices as sprint_avatar_fit (head / hands from projected
  SMPL-X head / MANO vertices when labels are exact).
- NEW per-record fields: "invtanfov" (float; = 2 f / L for the perspective crop of side L px with focal f px, principal point at the
  crop centre; GUAVA's fixed value is 24), "view" (str camera id; "0" for monocular video).
- Perspective crop: rotate a virtual camera onto the ray through the person-box centre (homography warp of the original image), fold
  that rotation into global_pose / translation (about the pelvis joint), square side 1.1 x the long side of the person box, 1024 px.
- Debug for checks: <clip>/gt_proj.npz with up to 10 rows: frame_key, v2d_1024 (10475, 2) = the SOURCE label's mesh (original model,
  original camera) projected into the 1024 crop, so the coordinator can check GUAVA's renderer against it (< 1 px at 512).
- Masks: StyleMatte (EHM-Tracker/pretrained/matting/stylematte_synth.pt; see sprint_avatar_fit.py step 3) on the 1024 crop.
- Image JPEG/PNG encoding: as sprint_avatar_fit / LMDBEngine dump (uint8 tensors).

## Rules
- GPUs: only 0, 1, 2 (CUDA_VISIBLE_DEVICES); check nvidia-smi; others share them. Any single GPU job > 15 min is fine for conversion
  (user approved the GUAVA-FB pipeline) but keep memory < 20 GB per process.
- NEVER delete / move / modify anything existing (checkpoints, runs, datasets). Read-only on /raid and on the source datasets.
- SPRINT for fitting = SPRINT from scratch EMA: /raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt (NOT sprint_ft).
- NEVER use People-Snapshot, NeuMan, X-Humans, UBC-Fashion test videos, 3DPW in any training data.
- Credentials (~/.mpi_credentials) never printed / logged.
