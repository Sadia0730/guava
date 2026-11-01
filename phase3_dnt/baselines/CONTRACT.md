# AvatarTest-v1 baseline contract (PREREG "AvatarTest-v1 addendum 1")

E = /vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1   (SMB share: slow, no symlinks)

## Input package (written by phase3_dnt/eval_split/convert_test.py; read-only for baselines)
E/base/<dataset>/<clip>/          dataset in {ubody, bedlam2}; clip id = split JSON "clip"
- frames.json   {"dataset", "clip", "keys": [all GT frame keys, time order], "source": key, "recon": [keys], "test": [keys],
                 "fps", "note"}. recon = first half (the ONLY frames a per-subject method may train on), test = latter half.
- images/<key>.png   512 x 512 RGB, the person crop (perspective crop, see below); background NOT removed.
- masks/<key>.png    512 x 512 uint8 alpha (StyleMatte; 255 = person). Training masks: alpha > 127.
- cameras.npz   keys (N,), K (N, 3, 3) per-frame intrinsics of the 512 crop. Continuous pixel convention: the image spans
                [0, 512] x [0, 512], principal point (256, 256), pixel centres at +0.5. Extrinsics are the identity:
                world = camera frame, OpenCV axes (x right, y down, z forward). The focal changes from frame to frame
                (the crop follows the person), so a method that assumes one K per video needs a per-frame K in its loader.
- smplx2020.npz keys, betas (200,), global_orient (N, 3), body_pose (N, 21, 3), jaw_pose (N, 3), leye_pose (N, 3),
                reye_pose (N, 3), left_hand_pose (N, 15, 3), right_hand_pose (N, 15, 3) (ABSOLUTE = flat_hand_mean True),
                expression (N, 50), transl (N, 3). Model: /home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz
                (smplx.SMPLX(..., num_betas=200, num_expression_coeffs=50, use_pca=False, flat_hand_mean=True)).
                verts_camera = smplx_output.vertices (+ transl via the layer's transl argument).
                SMPL-X v1.1 (/raid/ubx858/datasets/models/smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz, read only) has the SAME
                template, joint regressor, skinning and pose correctives; only the shape / expression bases differ, so a
                method using v1.1 keeps all poses and needs only a least-squares fit of its betas / expression to the
                2020 shaped template (and flat_hand_mean=True, or subtract its hand mean).
- smpl.npz      keys, betas (10,), global_orient (N, 3), body_pose (N, 23, 3), transl (N, 3), fit_err_mm (N,).
                Neutral SMPL: /home/ubx858/ml-hugs/data/smpl/SMPL_NEUTRAL.pkl (v1.1.0 python release, 10 betas).
                verts_camera = smpl(betas, global_orient, body_pose, transl).vertices. Fitted to the GT SMPL-X mesh mapped
                by the SMPL-X -> SMPL vertex correspondence (one betas per clip).
- overlay.jpg   a few frames with the projected GT SMPL-X mesh over the image (sanity check of the camera convention).

## Output (written by the baseline)
E/renders/<method>/<dataset>/<clip>/<key>.png  512 x 512 RGB, test keys ONLY, background black (0), same crop / camera
as images/<key>.png (render with K[key], identity extrinsics). 8-bit PNG.
E/renders/<method>/<dataset>/<clip>/run.json   {"method", "commit", "train_s" (wall, incl. the method's own preprocessing /
face fitting), "render_s_per_frame", "gpu" (nvidia-smi name), "iterations", "notes", "loader_changes"}.
Logs may go next to run.json (small). NO checkpoints / caches on /vault: keep them under /dev/shm and delete them after the
test renders are written (only final outputs are kept).

## Rules
- Train on recon keys only; render test keys only. Driving = the GT params above (no test-time pose optimisation in the
  main protocol; if the method's own published protocol fits test poses, do it only as a separately named extra variant).
- Official code at a recorded commit, default hyper-parameters of its monocular-video configuration; only minimal loader
  changes (document each in run.json "loader_changes" and in phase3_dnt/baselines/<method>/NOTES.md).
- Environments, code, pretrained weights, pip / conda caches: /dev/shm/ubx858_baselines/<method>/ (set PIP_CACHE_DIR,
  CONDA_PKGS_DIRS, TORCH_HOME, HF_HOME, XDG_CACHE_HOME there). NEVER write to /raid (near full) or fill /home (29 GB free).
- GPUs are shared: check nvidia-smi first; only use a GPU with < 2 GB in use.
