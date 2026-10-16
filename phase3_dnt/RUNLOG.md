# Driven, Not Tracked — run log

Every run: git SHAs of both repositories, config, seed, GPU, wall time, output directory. Mirrored at
`/raid/ubx858/outputs/dnt/RUNLOG.md`.

| # | Date | Run | guava SHA | PEAR SHA | Config / command | Seed | GPU | Wall time | Output |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-15 | Keypoint regressor fit, r = 0.12 m / 8 vertices (BEDLAM2 renders, ViTPose+ small) | 228c0fa6 | 57d7c4c | `tools/fit_detector_keypoint_regressor.py --out /raid/ubx858/outputs/dnt/regressor` | 0 | 3 (shared) | ~5 min | `/raid/ubx858/outputs/dnt/regressor/` |
| 2 | 2026-10-15 | Keypoint regressor fit, r = 0.20 m / 16 vertices (USED) | 5c8f8880 | 57d7c4c | `... --radius 0.2 --topk 16 --out /raid/ubx858/outputs/dnt/regressor_r020` | 0 | 3 (shared) | ~5 min | `/raid/ubx858/outputs/dnt/regressor_r020/` (W SHA-256 13b2b384…) |
| 3 | 2026-10-16 | C1 B2 oracle tracking, first attempt (STOPPED at ~35%: unpinned workers thrashed, 15-17 s/frame); partial outputs moved to `c1/aborted_0216/` | 2bdbbe30 | 5dc6b98 | `bash phase3_dnt/run_ehm_tracking.sh 2 6` | — | 2 (shared) | 43 min | `/raid/ubx858/outputs/dnt/c1/aborted_0216/` |
| 4 | 2026-10-16 | C1 B2 oracle tracking, pinned workers (37 cores each); 33 of 36 tracked | 12707359 | 5dc6b98 | `bash phase3_dnt/run_ehm_tracking.sh 2 6` | — | 2 (shared) | 5.9 h (02:14-08:11) | `/raid/ubx858/outputs/dnt/c1/tracked/` |
| 5 | 2026-10-16 | UBody labelled smoke subset (Singing, 2,074 rows) | af06664a | f03aead | `tools/build_ubody_label_crops.py --scenes Singing --max-rows 2000` | 20261015 | CPU | 3 min | `/raid/ubx858/outputs/dnt/data/ubody_labels_smoke/` |
| 6 | 2026-10-16 | Track A smoke test, 200 steps (PEAR not loaded) | a01d902c | f03aead + diff (= 5dc6b98) | `train_student_gt.py --config trackA/smoke/smoke_config.yaml --steps 200 --max-rows 4000 --dump-samples 16` | 20261009 | 1 (shared) | 6.8 min | `/raid/ubx858/outputs/dnt/trackA/smoke/` |
| 7 | 2026-10-16 | UBody labelled crops, every 3rd annotated frame (192,788 rows, 6.0 GB) | 2bdbbe30 | 5dc6b98 | `tools/build_ubody_label_crops.py --every 3 --processes 20` | 20261015 | CPU | 44 min | `/raid/ubx858/outputs/dnt/data/ubody_labels/` |
| 8 | 2026-10-16 | **Track A full run, PEAR-free student** (128k steps, best/latest only, EMA 0.9995) | 12707359 | 5dc6b98 (clean) | `train_student_gt.py --config configs/dnt/student_pearfree.yaml --output-dir /raid/ubx858/outputs/dnt/trackA/run` | 20261009 | 1 (shared) | started 02:53 | `/raid/ubx858/outputs/dnt/trackA/run/` |
| 9 | 2026-10-16 | Part-query evaluation (FT_1q / FT_5q / FT_5q_handcrop best.pt) + evaluator regression | 2bdbbe30 / 395d74d6 | 5dc6b98 / f2ce22b | `parts_eval/run_parts_eval.sh 7` (+ resumable for the hand-crop row) | — | 7 (idle) | running | `/raid/ubx858/outputs/dnt/parts_eval/` |
| 10 | 2026-10-16 | C1 script smoke tests on the 60-frame tracked clip (canonical-camera check, neck-pivot swaps) | 395d74d6 | f2ce22b | `phase3_dnt/c1_live_gap.py --split c1/smoke/split_smoke.json --future-from 30` | 0 | 3 (idle) | ~20 min each | `/raid/ubx858/outputs/dnt/c1/smoke/` |
| 11 | 2026-10-16 | C1 B3-B4 scoring, part 1 (15 clips; crashed on NaN EHM crop boxes) | e58ab499 | 39fd465 | `run_c1_after_tracking.sh 2` | 0 | 2 | ~1.5 h | `/raid/ubx858/outputs/dnt/c1/results/` |
| 12 | 2026-10-16 | C1 B3-B4 scoring, resumed (unreadable frames dropped; 33 identities scored) | eece1b37 | 39fd465 | `phase3_dnt/c1_live_gap.py --future-from 60` | 0 | 2 | 4.2 h | `/raid/ubx858/outputs/dnt/c1/results/` |
