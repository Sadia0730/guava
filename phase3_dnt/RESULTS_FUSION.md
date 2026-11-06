# Stream accumulation (feed-forward UV fusion) — evaluation tables (2026-11-06)

Test sets are descriptive for the fused rows (PREREG amendment 2: reported after the decision, no selection). Cells: PSNR ↑ / SSIM ↑ / LPIPS ↓ (LPIPS-VGG), means over clips as in RESULTS_FINAL.md (no bootstrap CI here). Fused rows use the clip's training frames with GT poses, fused once into the UV map (`eval_split/fusion_gt_row.py --mode mean --thr 0 --cap 0`); nothing is optimised per person.

## PublicBench-v1 (test): People-Snapshot 4 clips / 190 frames, NeuMan 4 clips / 27 frames

| Method | People-Snapshot (4) | NeuMan (4) | All (8 clips) |
|---|---|---|---|
| GUAVA-FB [GT], one photo (B28k) | 27.83 / 0.933 / 0.0646 | 23.42 / 0.898 / 0.0738 | 25.63 / 0.916 / 0.0692 |
| GUAVA-FB [GT] + accumulated training frames (B28k, mean, prior 1; D1 row) | 27.66 / 0.936 / 0.0658 | 24.13 / 0.903 / 0.0826 | 25.90 / 0.920 / 0.0742 |
| GUAVA-FB [GT] + accumulated training frames (B28k, prior 2 = live setting) | 27.66 / 0.936 / 0.0658 | 24.13 / 0.903 / 0.0822 | 25.90 / 0.920 / 0.0740 |
| fusion-trained checkpoint A2 + accumulated frames (descriptive) | 27.78 / 0.938 / 0.0608 | 23.89 / 0.903 / 0.0771 | 25.84 / 0.920 / 0.0690 |
| continuation control B2 + accumulated frames (descriptive) | 27.56 / 0.936 / 0.0652 | 24.06 / 0.903 / 0.0818 | 25.81 / 0.919 / 0.0735 |
| GART (per-person, video-trained) | 31.29 / 0.959 / 0.0357 | 25.02 / 0.913 / 0.0723 | 28.15 / 0.936 / 0.0540 |
| GaussianAvatar (per-person) | 30.37 / 0.951 / 0.0543 | 23.61 / 0.901 / 0.0793 | 26.99 / 0.926 / 0.0668 |
| ExAvatar [GT face] (per-person) | 30.67 / 0.955 / 0.0409 | 24.39 / 0.908 / 0.0743 | 27.53 / 0.932 / 0.0576 |

## ValBench-v1 (validation, the registered selection set): UBody 20 clips / 2,556 frames, BEDLAM2 30 clips / 956 frames; frame-weighted (score.py --tag vb)

| Model | UBody one photo | UBody + frames | BEDLAM2 one photo | BEDLAM2 + frames | All one photo | All + frames |
|---|---|---|---|---|---|---|
| B28k (today's model) | 19.06 / 0.783 / 0.1872 | 19.57 / 0.799 / 0.1872 | 26.13 / 0.918 / 0.0700 | 27.00 / 0.927 / 0.0672 | 20.98 / 0.820 / 0.1553 | 21.59 / 0.834 / 0.1545 |
| fusion-trained A2 (lr 3e-5, 12k steps) | 18.76 / 0.781 / 0.1888 | 19.25 / 0.798 / 0.1844 | 25.85 / 0.916 / 0.0707 | 26.94 / 0.927 / 0.0651 | 20.69 / 0.818 / 0.1566 | 21.34 / 0.833 / 0.1519 |
| continuation control B2 | 18.88 / 0.781 / 0.1885 | 19.40 / 0.797 / 0.1884 | 25.84 / 0.917 / 0.0705 | 26.74 / 0.926 / 0.0674 | 20.77 / 0.818 / 0.1564 | 21.40 / 0.833 / 0.1555 |
| round 1: fusion-trained A (lr 8e-6, 4k steps) | 18.99 / 0.781 / 0.1878 | 19.41 / 0.798 / 0.1847 | 26.04 / 0.917 / 0.0704 | 27.07 / 0.927 / 0.0654 | 20.91 / 0.818 / 0.1559 | 21.49 / 0.834 / 0.1523 |
| round 1: control B | 18.99 / 0.782 / 0.1879 | 19.42 / 0.798 / 0.1880 | 26.13 / 0.918 / 0.0702 | 27.00 / 0.927 / 0.0670 | 20.94 / 0.819 / 0.1559 | 21.48 / 0.833 / 0.1551 |

Registered rule (PREREG Phase 7): a trained checkpoint is adopted only if its '+ frames' PSNR on All beats B28k's naive fusion (21.59) by 0.30 dB with LPIPS not worse by more than 0.002, and its one-photo PSNR stays within 0.10 dB of B28k (20.98). A2: 21.34 / 20.69 -> not adopted; B28k stays the model. See `O/reports/p7/decision.md`, RUNLOG 120.

Sources: `O/eval/publicbench_v1/scores/*_full.json`, `/dev/shm/ubx858_eval/valbench_v1/scores/*_vb.json`; view-bin breakdowns in `O/reports/p7/publicbench_viewbins_r2.md`, `valbench_viewbins.md`. O = /vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb.
