# Deleted outputs (2026-10-23, user-approved cleanup of abandoned work)

Rule: only items no current config / script / run reads (grep of phase3_dnt, tools, configs, main, PEAR configs / tools /
dataset / trainer, RUNLOG and reports excluded). Kept: every report (*.md), summary.json / stats / logs / train_log.jsonl,
PREREG, RUNLOG, git history, PEAR released checkpoint, L-real best.pt, Run A best.pt (+ eval_programme/runA/best_ema.pt,
the SPRINT init), A2-ctrl best.pt (+ trackA2/eval/ckpt/A2ctrl_best_ema.pt), Track A best.pt, refiner v2b, refiner v2 (17 MB;
still read by phase3_dnt/c1_live_gap.py for the C1 "S+R" variant), C1 / C2 results, the c2t tracked clips, the GUAVA (a) / (b)
runs, every dataset / crop root used by SPRINT.

| Deleted | Size | Recreate |
|---|---|---|
| dnt/runB_main/checkpoints (Run B) | 2.2 GB | RUNLOG 42: `train_student_gt.py --config configs/dnt/student_pearfree_RunB.yaml` (PEAR 6624644) after the expert labels below |
| dnt/runs_ab/smokeB/checkpoints | 2.2 GB | RUNLOG 41 smoke command |
| dnt/runs_ab/runB/{expert_gt, expert_arctic, labels_B, coco_expert_labels.npz, ubody_face_labels_B.npz, ubody_hand_labels.npz} (Run B expert labels; jsons / logs kept) | 6.6 GB | RUNLOG 29 / 35 / 40 / 41: tools/expert_hands.py, tools/expert_faces.py, tools/build_runB_labels.py (needs the experts below) |
| dnt/eval_programme/runB/{best_ema.pt, overlays, qual_face} (summaries kept) | 0.3 GB | `phase3_dnt/run_full_eval.sh runB <runB_main> ema <gpu>` |
| /raid/ubx858/experts (HaMeR, WiLoR, SMIRK code + weights, extra site-packages) | 11 GB | git clone github.com/geopavlakos/hamer (3a01849), github.com/rolpotamias/WiLoR (fcb9113), github.com/georgeretsi/smirk (c7de404) into /raid/ubx858/experts; then `bash phase3_dnt/sprint/experts_fetch/fetch_weights.sh` (copy kept; HaMeR demo data, WiLoR ckpt, SMIRK ckpt, MediaPipe); pip --target site: phase3_dnt/sprint/experts_fetch/site_packages.txt; stub_render.py copy kept |
| dnt/guava_{a,b}/smoke | 5.0 GB | `phase3_dnt/guava_finetune.py --variant a/b --iters <smoke>` |
| smoke runs: dnt/data/ubody_labels_smoke, dnt/c1/smoke, phase1_task1/{ldata_smoke, eval_smoke, checkpoint3/smoke}, phase2/{control_headft_L01_smoke, lreal_smoke, refiner_v2_smoke_Ldata} | 5.5 GB | each run's smoke command in its report / RUNLOG |
| old screens: phase1_task1/screen/{a, b, c, c_235k, c_vitpose, L_0.05, L_0.1, L_data}/checkpoints | 13.3 GB | phase1_task1 screening configs (phase1_task1 reports); L_data is also read by historical tools (keypoint offsets, regressor fit) whose outputs are kept |
| phase2/control_headft_L01 (L-0.1 head-only control) checkpoints | 0.3 GB | phase2 REFINER_V2_AND_PARTS.md |
| phase2/refiner_Ldata (refiner v1 on L-data) | 17 MB | phase2/train_refiner_and_eval.sh |
| Run 1: dnt/run1/checkpoints, dnt/eval_programme/run1/best_ema.pt | 2.5 GB | RUNLOG 24: `configs/dnt/student_pearfree_Fa.yaml` |
| part-query runs: phase2/parts/{FT_1q, FT_5q, FT_5q_handcrop}/checkpoints | 48 GB | phase2 REFINER_V2_AND_PARTS.md / multiquery configs |
| latest.pt of finished runs (resume-only; best.pt kept): dnt/trackA/run, dnt/trackA2/ctrl, dnt/runA, phase1_task1/screen/L_real | 4.1 GB | not recreatable bit-exactly; best.pt + train_log.jsonl kept |

**Total freed: ~101 GB** (sum of the sizes above); /raid free 29.9 GB -> 124 GB (2026-10-23, after deletion; other users'
writes continue).
