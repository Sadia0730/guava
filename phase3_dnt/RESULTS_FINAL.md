# FinalBench: final comparison tables

Generated 2026-11-05 17:55:32 CST by `eval_split/final_tables.py` (`defaults`); percentile bootstrap over clips, 10,000 resamples, `numpy.random.default_rng(20261108)` (one generator per clip pool; overall stratified by dataset). PREREG section "FinalBench: final comparison tables" (amendment 2026-11-05: main table = rows 1-3). Per-clip rows, frame-weighted means and timing fields: `results_final.csv`.

## Warnings / notes

- **BEDLAM2: ExAvatar [GT face] row shown as — (no scored clips on BEDLAM2); it does not constrain the clip set.**

## Main table: single-image methods, FinalBench (41 clips / 4,865 test frames)

| Method | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | All (41 clips) |
|---|---|---|---|---|---|
| GUAVA (released) | 19.48 / 0.788 / 0.1912 | 25.80 / 0.921 / 0.0780 | 23.23 / 0.896 / 0.0788 | 22.75 / 0.906 / 0.0907 | 20.70 / 0.820 / 0.1618 |
| ↳ 95 % CI | [17.86, 21.12] / [0.761, 0.815] / [0.1698, 0.2131] | [24.56, 27.16] / [0.912, 0.930] / [0.0760, 0.0794] | [22.60, 23.87] / [0.889, 0.902] / [0.0742, 0.0869] | [20.13, 27.01] / [0.893, 0.922] / [0.0822, 0.0958] | [19.49, 21.91] / [0.800, 0.839] / [0.1462, 0.1778] |
| GUAVA-FB [SMPL-X GT] | 19.62 / 0.786 / 0.1909 | 27.83 / 0.933 / 0.0646 | 23.42 / 0.898 / 0.0738 | 24.90 / 0.922 / 0.0749 | 21.18 / 0.822 / 0.1587 |
| ↳ 95 % CI | [17.88, 21.36] / [0.758, 0.815] / [0.1675, 0.2141] | [26.93, 28.73] / [0.927, 0.940] / [0.0638, 0.0657] | [22.67, 24.18] / [0.889, 0.905] / [0.0670, 0.0828] | [22.92, 28.54] / [0.909, 0.937] / [0.0676, 0.0789] | [19.89, 22.47] / [0.801, 0.843] / [0.1416, 0.1757] |
| GUAVA-FB [SPRINT, k = 0] | 17.22 / 0.744 / 0.2357 | 24.47 / 0.908 / 0.0831 | 22.37 / 0.886 / 0.0840 | 21.31 / 0.890 / 0.1056 | 18.73 / 0.785 / 0.1965 |
| ↳ 95 % CI | [15.77, 18.63] / [0.715, 0.772] / [0.2134, 0.2580] | [23.30, 25.37] / [0.897, 0.920] / [0.0802, 0.0859] | [21.94, 22.81] / [0.883, 0.888] / [0.0782, 0.0903] | [18.56, 26.06] / [0.876, 0.909] / [0.1004, 0.1107] | [17.64, 19.81] / [0.763, 0.805] / [0.1802, 0.2128] |

Caption. Self-reenactment on FinalBench (UBody 30, People-Snapshot 4, NeuMan 4, X-Humans 3 clips; one identity per clip; cells: PSNR ↑ / SSIM ↑ / LPIPS ↓ (LPIPS-VGG), second line = 95 % CI; L1 ↓ in its own table below). Single-image methods use the clip's FIRST training frame as the only source image and are driven by the GT body model of each test frame (converted to each method's body model); GUAVA-FB [SPRINT, k = 0] uses no GT at all (own detector, crop, StyleMatte matte and SPRINT pose estimate; k = 0, largest box). No test-time pose, face or expression fitting in any row. Same 512 person crop, masks and metrics for every row (`eval_split/score.py`); per clip = mean over its test frames; per dataset and overall = mean over clips (each clip weighs the same). Clip set = the clips on which all 6 methods of the main and the reference table finished (excluded clips listed below). Disclosures: the AvatarTest-v1 and PublicBench-v1 numbers of these three rows and of the per-person methods were seen before the FinalBench registration (2026-11-02); UBody clips are upper body only (UBody GT is pseudo-GT, UBody's SMPL-X fits); NeuMan test sets are 3-10 frames; released GUAVA may have trained on UBody; X-Humans images are renders of textured scans (mask = rendered depth > 0).

## Reference table: per-person, video-trained methods, FinalBench (same 41 clips)

| Method | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | All (41 clips) |
|---|---|---|---|---|---|
| GART | 19.57 / 0.797 / 0.1856 | 31.29 / 0.959 / 0.0357 | 25.02 / 0.913 / 0.0723 | 26.43 / 0.935 / 0.0714 | 21.75 / 0.834 / 0.1516 |
| ↳ 95 % CI | [17.94, 21.32] / [0.768, 0.825] / [0.1642, 0.2076] | [29.99, 32.97] / [0.952, 0.965] / [0.0293, 0.0428] | [24.35, 25.56] / [0.904, 0.920] / [0.0623, 0.0830] | [24.06, 30.69] / [0.923, 0.949] / [0.0636, 0.0772] | [20.50, 23.06] / [0.813, 0.855] / [0.1359, 0.1678] |
| GaussianAvatar | 19.02 / 0.785 / 0.2036 | 30.37 / 0.951 / 0.0543 | 23.61 / 0.901 / 0.0793 | 25.05 / 0.923 / 0.0715 | 21.02 / 0.823 / 0.1672 |
| ↳ 95 % CI | [17.39, 20.63] / [0.757, 0.813] / [0.1813, 0.2268] | [29.77, 30.83] / [0.947, 0.955] / [0.0484, 0.0610] | [22.90, 24.34] / [0.894, 0.907] / [0.0748, 0.0862] | [22.88, 28.64] / [0.912, 0.937] / [0.0649, 0.0754] | [19.80, 22.22] / [0.802, 0.843] / [0.1509, 0.1842] |
| ExAvatar [GT face] | 19.47 / 0.797 / 0.1993 | 30.67 / 0.955 / 0.0409 | 24.39 / 0.908 / 0.0743 | 23.18 / 0.911 / 0.0875 | 21.32 / 0.831 / 0.1635 |
| ↳ 95 % CI | [17.88, 21.04] / [0.771, 0.822] / [0.1799, 0.2194] | [30.05, 31.68] / [0.949, 0.960] / [0.0366, 0.0452] | [23.76, 25.39] / [0.901, 0.916] / [0.0668, 0.0806] | [17.07, 32.60] / [0.861, 0.963] / [0.0397, 0.1333] | [20.04, 22.61] / [0.812, 0.850] / [0.1490, 0.1783] |

Caption. Trained on each person's video (training frames); not directly comparable with single-image methods. GART and GaussianAvatar are driven by SMPL fitted to the GT SMPL-X mesh of each test frame; ExAvatar [GT face] is the conforming variant (test-frame face = the GT's converted jaw / expression, eyes 0, neutral where the GT is SMPL; no DECA, keypoint or face fitting on test frames; its training-frame refinement kept). Same clip set, crop, masks, metrics and CIs as the main table; cells: PSNR ↑ / SSIM ↑ / LPIPS ↓ (LPIPS-VGG), second line = 95 % CI; L1 ↓ in its own table below.

## BEDLAM2 synthetic table (40 clips / 1,470 test frames)

| Method | BEDLAM2 (40) |
|---|---|
| GUAVA (released) | 24.80 / 0.905 / 0.0810 |
| ↳ 95 % CI | [23.86, 25.74] / [0.896, 0.913] / [0.0744, 0.0877] |
| GUAVA-FB [SMPL-X GT] | 27.19 / 0.928 / 0.0630 |
| ↳ 95 % CI | [26.07, 28.34] / [0.920, 0.936] / [0.0571, 0.0692] |
| GUAVA-FB [SPRINT, k = 0] | 22.67 / 0.874 / 0.1079 |
| ↳ 95 % CI | [21.60, 23.75] / [0.864, 0.884] / [0.1006, 0.1157] |
| *per-person, video-trained (reference)* | |
| GART | 28.17 / 0.934 / 0.0612 |
| ↳ 95 % CI | [27.09, 29.28] / [0.927, 0.942] / [0.0549, 0.0681] |
| GaussianAvatar | 26.79 / 0.925 / 0.0727 |
| ↳ 95 % CI | [25.80, 27.82] / [0.917, 0.932] / [0.0666, 0.0790] |
| ExAvatar [GT face] | — |

Caption. AvatarTest-v1 BEDLAM2 (40 synthetic full-body clips), same rules, source, driving, crop, masks, metrics and CIs as above; cells: PSNR ↑ / SSIM ↑ / LPIPS ↓ (LPIPS-VGG), second line = 95 % CI; L1 ↓ in its own table below. ExAvatar [GT face] has no BEDLAM2 runs (—; only the non-conforming subset24 runs exist and are not shown). 38 of the 40 BEDLAM2 identities appear in other BEDLAM2 clips that SPRINT trained on.

## Paired differences vs GUAVA (released)

Per clip row − GUAVA (released), mean over clips with the same bootstrap CI (cells: ΔPSNR / ΔSSIM / ΔLPIPS / ΔL1; higher is better for PSNR / SSIM, lower for LPIPS / L1; a CI that excludes 0 is a significant paired difference at the 5 % level, uncorrected for the number of comparisons).

### FinalBench (41 clips)

| Method | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | All (41 clips) |
|---|---|---|---|---|---|
| GUAVA-FB [SMPL-X GT] | +0.14 / -0.001 / -0.0003 / +0.0014 | +2.03 / +0.013 / -0.0134 / -0.0032 | +0.19 / +0.002 / -0.0050 / -0.0004 | +2.15 / +0.017 / -0.0157 / -0.0055 | +0.47 / +0.002 / -0.0031 / +0.0003 |
| ↳ 95 % CI | [-0.23, +0.57] / [-0.007, +0.006] / [-0.0066, +0.0051] / [-0.0005, +0.0034] | [+1.59, +2.35] / [+0.008, +0.016] / [-0.0144, -0.0120] / [-0.0036, -0.0026] | [-0.83, +0.88] / [+0.000, +0.004] / [-0.0080, -0.0020] / [-0.0021, +0.0020] | [+1.53, +2.79] / [+0.015, +0.019] / [-0.0175, -0.0145] / [-0.0087, -0.0021] | [+0.19, +0.81] / [-0.003, +0.007] / [-0.0078, +0.0008] / [-0.0011, +0.0018] |
| GUAVA-FB [SPRINT, k = 0] | -2.26 / -0.044 / +0.0445 / +0.0186 | -1.34 / -0.012 / +0.0051 / +0.0021 | -0.87 / -0.010 / +0.0052 / +0.0023 | -1.44 / -0.015 / +0.0150 / +0.0041 | -1.97 / -0.035 / +0.0347 / +0.0143 |
| ↳ 95 % CI | [-2.78, -1.78] / [-0.055, -0.033] / [+0.0363, +0.0534] / [+0.0136, +0.0245] | [-1.90, -0.77] / [-0.017, -0.008] / [+0.0015, +0.0074] / [+0.0016, +0.0026] | [-1.60, -0.19] / [-0.014, -0.006] / [+0.0019, +0.0084] / [+0.0005, +0.0042] | [-1.79, -0.96] / [-0.017, -0.013] / [+0.0099, +0.0183] / [+0.0018, +0.0060] | [-2.36, -1.61] / [-0.043, -0.028] / [+0.0286, +0.0412] / [+0.0107, +0.0187] |
| *per-person, video-trained (reference)* | | | | | |
| GART | +0.09 / +0.009 / -0.0056 / -0.0018 | +5.48 / +0.038 / -0.0423 / -0.0075 | +1.79 / +0.017 / -0.0065 / -0.0047 | +3.68 / +0.029 / -0.0192 / -0.0096 | +1.04 / +0.014 / -0.0103 / -0.0032 |
| ↳ 95 % CI | [-1.21, +1.31] / [-0.008, +0.025] / [-0.0196, +0.0081] / [-0.0089, +0.0054] | [+3.10, +7.42] / [+0.025, +0.050] / [-0.0494, -0.0332] / [-0.0091, -0.0051] | [+1.49, +2.10] / [+0.014, +0.021] / [-0.0127, -0.0004] / [-0.0055, -0.0037] | [+3.44, +3.92] / [+0.027, +0.030] / [-0.0205, -0.0186] / [-0.0125, -0.0050] | [+0.07, +1.97] / [+0.002, +0.026] / [-0.0206, -0.0002] / [-0.0084, +0.0021] |
| GaussianAvatar | -0.46 / -0.003 / +0.0124 / +0.0017 | +4.57 / +0.030 / -0.0237 / -0.0066 | +0.38 / +0.005 / +0.0005 / -0.0018 | +2.31 / +0.018 / -0.0192 / -0.0081 | +0.31 / +0.003 / +0.0054 / -0.0002 |
| ↳ 95 % CI | [-0.89, -0.07] / [-0.011, +0.006] / [+0.0054, +0.0200] / [-0.0008, +0.0043] | [+3.46, +5.48] / [+0.022, +0.035] / [-0.0301, -0.0152] / [-0.0076, -0.0050] | [+0.05, +0.74] / [+0.001, +0.010] / [-0.0010, +0.0024] / [-0.0028, -0.0007] | [+1.63, +2.74] / [+0.015, +0.019] / [-0.0217, -0.0173] / [-0.0109, -0.0033] | [-0.02, +0.63] / [-0.003, +0.009] / [+0.0003, +0.0111] / [-0.0020, +0.0017] |
| ExAvatar [GT face] | -0.01 / +0.009 / +0.0081 / -0.0016 | +4.87 / +0.034 / -0.0371 / -0.0070 | +1.15 / +0.013 / -0.0045 / -0.0037 | +0.43 / +0.005 / -0.0032 / +0.0077 | +0.61 / +0.012 / +0.0017 / -0.0017 |
| ↳ 95 % CI | [-0.55, +0.57] / [-0.003, +0.021] / [-0.0024, +0.0185] / [-0.0064, +0.0030] | [+3.25, +6.07] / [+0.023, +0.043] / [-0.0421, -0.0315] / [-0.0083, -0.0050] | [+0.43, +1.80] / [+0.007, +0.019] / [-0.0128, +0.0038] / [-0.0050, -0.0019] | [-4.02, +5.59] / [-0.032, +0.041] / [-0.0561, +0.0393] / [-0.0056, +0.0295] | [+0.07, +1.17] / [+0.003, +0.020] / [-0.0066, +0.0099] / [-0.0054, +0.0019] |

### BEDLAM2 (40 clips)

| Method | BEDLAM2 (40) |
|---|---|
| GUAVA-FB [SMPL-X GT] | +2.39 / +0.024 / -0.0180 / -0.0042 |
| ↳ 95 % CI | [+1.99, +2.84] / [+0.022, +0.026] / [-0.0198, -0.0162] / [-0.0049, -0.0036] |
| GUAVA-FB [SPRINT, k = 0] | -2.14 / -0.030 / +0.0269 / +0.0063 |
| ↳ 95 % CI | [-2.61, -1.64] / [-0.036, -0.025] / [+0.0225, +0.0314] / [+0.0048, +0.0080] |
| *per-person, video-trained (reference)* | |
| GART | +3.37 / +0.030 / -0.0197 / -0.0068 |
| ↳ 95 % CI | [+2.66, +4.07] / [+0.024, +0.035] / [-0.0239, -0.0153] / [-0.0088, -0.0049] |
| GaussianAvatar | +1.98 / +0.020 / -0.0083 / -0.0050 |
| ↳ 95 % CI | [+1.48, +2.51] / [+0.016, +0.024] / [-0.0117, -0.0048] / [-0.0066, -0.0035] |
| ExAvatar [GT face] | — |

## L1 ↓ (mean absolute error on the 512 crop, [0, 1] scale)

### FinalBench (41 clips)

| Method | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | All (41 clips) |
|---|---|---|---|---|---|
| GUAVA (released) | 0.0519 | 0.0128 | 0.0187 | 0.0199 | 0.0425 |
| ↳ 95 % CI | [0.0391, 0.0666] | [0.0108, 0.0141] | [0.0172, 0.0208] | [0.0099, 0.0252] | [0.0331, 0.0533] |
| GUAVA-FB [SMPL-X GT] | 0.0533 | 0.0097 | 0.0183 | 0.0144 | 0.0428 |
| ↳ 95 % CI | [0.0399, 0.0687] | [0.0082, 0.0108] | [0.0159, 0.0207] | [0.0078, 0.0188] | [0.0330, 0.0541] |
| GUAVA-FB [SPRINT, k = 0] | 0.0704 | 0.0149 | 0.0210 | 0.0240 | 0.0568 |
| ↳ 95 % CI | [0.0549, 0.0883] | [0.0127, 0.0161] | [0.0197, 0.0221] | [0.0117, 0.0305] | [0.0455, 0.0698] |
| *per-person, video-trained (reference)* | | | | | |
| GART | 0.0500 | 0.0053 | 0.0140 | 0.0103 | 0.0392 |
| ↳ 95 % CI | [0.0392, 0.0621] | [0.0047, 0.0059] | [0.0127, 0.0158] | [0.0049, 0.0131] | [0.0313, 0.0481] |
| GaussianAvatar | 0.0536 | 0.0062 | 0.0169 | 0.0118 | 0.0423 |
| ↳ 95 % CI | [0.0404, 0.0685] | [0.0057, 0.0066] | [0.0150, 0.0188] | [0.0066, 0.0144] | [0.0326, 0.0533] |
| ExAvatar [GT face] | 0.0503 | 0.0059 | 0.0150 | 0.0276 | 0.0408 |
| ↳ 95 % CI | [0.0382, 0.0652] | [0.0054, 0.0065] | [0.0131, 0.0167] | [0.0043, 0.0541] | [0.0317, 0.0517] |

### BEDLAM2 (40 clips)

| Method | BEDLAM2 (40) |
|---|---|
| GUAVA (released) | 0.0174 |
| ↳ 95 % CI | [0.0150, 0.0201] |
| GUAVA-FB [SMPL-X GT] | 0.0132 |
| ↳ 95 % CI | [0.0110, 0.0156] |
| GUAVA-FB [SPRINT, k = 0] | 0.0238 |
| ↳ 95 % CI | [0.0202, 0.0277] |
| *per-person, video-trained (reference)* | |
| GART | 0.0106 |
| ↳ 95 % CI | [0.0091, 0.0124] |
| GaussianAvatar | 0.0125 |
| ↳ 95 % CI | [0.0109, 0.0142] |
| ExAvatar [GT face] | — |

## Excluded clips

- FinalBench: 41 registered clips, 41 in the common set, 0 excluded.
- BEDLAM2: 40 registered clips, 40 in the common set, 0 excluded.

## Timing (median over the FinalBench common clips; not comparable across methods)

| Method | Avatar creation / training per clip | Render ms / frame | GPU(s) | Clips with timing |
|---|---|---|---|---|
| GUAVA (released) | 143 ms | 29.5 | NVIDIA L40S | 41 |
| GUAVA-FB [SMPL-X GT] | 99 ms | 27.5 | NVIDIA L40S | 41 |
| GUAVA-FB [SPRINT, k = 0] | 806 ms (model only 73 ms) | 17.6 | NVIDIA L40S | 41 |
| GART | 2.0 min | 8.8 | NVIDIA L40S | 41 |
| GaussianAvatar | 1.2 h | 258.8 | NVIDIA L40S | 41 |
| ExAvatar [GT face] | 1.8 h | 96.9 | NVIDIA L40S | 41 |

Note. GUAVA-family rows: avatar creation from the source frame (`recon_ms`; the SPRINT row includes its source fit: DWPose on CPU, matting, SPRINT, refinement) and `render_ms_per_frame`, measured inside the evaluation runs; per-person rows: `train_s` (per-clip training incl. preprocessing) and `render_s_per_frame` from their `run.json`. The baselines (and parts of the evaluation runs) ran on GPUs shared with other jobs, so these timings are indicative only and not comparable across methods.

## Input score files

- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/guava_full.json`: 2026-11-01 02:03:50
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/fb_gt_full.json`: 2026-11-01 01:56:23
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/fb_sprint_full.json`: 2026-11-01 02:12:02
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/gart_full.json`: 2026-11-01 09:38:07
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/gaussianavatar_full.json`: 2026-11-02 11:38:02
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1/scores/exavatar_gtface_full.json`: 2026-11-05 17:55:31
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/guava_full.json`: 2026-11-02 22:39:48
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/fb_gt_full.json`: 2026-11-02 22:40:52
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/fb_sprint_full.json`: 2026-11-02 22:41:06
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/gart_full.json`: 2026-11-02 22:40:22
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/gaussianavatar_full.json`: 2026-11-02 22:40:37
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v1/scores/exavatar_gtface_full.json`: 2026-11-04 09:38:31
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/guava_full.json`: 2026-11-03 01:10:42
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/fb_gt_full.json`: 2026-11-03 01:13:34
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/fb_sprint_full.json`: 2026-11-03 01:15:20
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/gart_full.json`: 2026-11-03 01:11:59
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/gaussianavatar_full.json`: 2026-11-04 09:39:48
- `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/publicbench_v2/scores/exavatar_gtface_full.json`: 2026-11-05 09:43:52
