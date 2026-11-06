# LHM++ baseline notes (PREREG "FinalBench addendum: LHM++", 2026-11-06)

Code: aigc3d/LHM-plusplus @ 6fa576c (`/home/ubx858/LHM-plusplus`, unchanged), model LHMPP-700M (HF snapshot fc9f7366...,
`/raid/ubx858/LHM-plusplus_data/pretrained_models`, read only), env `lhmpp` (torch 2.3.0 + cu121). Weights CC BY-NC 4.0 (research
only), code Apache-2.0. Adapter: `run_lhmpp.py`; driver `lhmpp_run.sh NAME VIEWS GPU` (copy of `/dev/shm/ubx858_eval/lhmpp_run.sh`).

Rows: `lhmpp` = source frame only (single image); `lhmpp_v16` = 16 recon frames (source first, uniform over the recon list).

Adapter decisions (fixed on two ValBench-v1 clips before any test clip):
- Source: package 512 crop x StyleMatte alpha (> 0.5) on white, cropped to the alpha box, `PadRatioWithScale(5/3, [840], val=True)`
  (840 x 504, the format of the repo's example inputs). LHM++'s rembg / BiRefNet and pose estimator are not used.
- Shape: package betas (200, SMPL-X 2020) -> 10 by least squares on the shape displacement (LHM++'s SMPLX_NEUTRAL.npz is the
  2020 file: template, regressor, weights and first shape dirs identical). Residual per clip in run.json (`shape_fit_mm`).
- Expression: LHM++ puts FLAME's 100 expression dirs on the SMPL-X face vertices; they equal SMPL-X 2020's (max diff 2e-7 mm),
  so the package's 50 coefficients are copied (fit residual 0.00 mm, `expr_map_mm`).
- Driving: root / body / jaw / eyes / hands (flat_hand_mean on both) / transl per test frame, camera-frame, identity c2w; per-frame
  focal / principal point from `cameras.npz` (continuous pixel convention, same as LHM++'s).
- Render: LHM++'s evaluation crop (`_process_mask_frame_for_eval` on a box image = projected GT SMPL-X vertices; 1.2 x, symmetric
  about the image centre, 420 px wide, 5:3), its neural renderer, pasted into the 512 board. Its RGB is composited over white
  (median background in the crop = 1.0 on every smoke clip), so the output is `rgb - (1 - alpha)` on black (exact
  un-premultiplication); `--composite mul_alpha` is the alternative, not used.
- Smoke (ValBench-v1, 10 test frames each, renders `valbench_v1/renders/lhmpp_smoke/`): alpha IoU to StyleMatte 0.88 (BEDLAM2,
  GUAVA-FB 0.89) / 0.78 (UBody); 0.1-0.18 s per frame, 1.3-1.8 s avatar build (3.9 s with 16 views).

Environment bug (affects anyone using the `lhmpp` env on this server): its numpy 1.23 / OpenBLAS64 auto-selects a kernel that
returns wrong results on the Xeon Platinum 8480+ (A @ X off by ~75 against torch, `lstsq` off by 1e5; thread count does not fix it).
`OPENBLAS_CORETYPE=SkylakeX` (or Haswell) makes it exact; `run_lhmpp.py` sets it before importing numpy and asserts a check.
LHM++'s heavy compute is torch on the GPU, but any numpy matrix product in that env without the variable is suspect.
