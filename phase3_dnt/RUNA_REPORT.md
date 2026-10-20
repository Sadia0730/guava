# RUN A "dataset labels" — evaluation (2026-10-20, measured)

Run `/raid/ubx858/outputs/dnt/runA/` (RUNLOG 36): 32k steps from A2-ctrl EMA, GPU 4, 22:15-03:13. Selection (validation
only): best live 3DPW-val PA-MPJPE 47.787 at step 32k; EMA there 47.786 (not > 0.3 mm worse) -> **best.pt EMA reported**.
Evaluation `phase3_dnt/run_full_eval.sh runA ... ema 4` (RUNLOG 38) -> `/raid/ubx858/outputs/dnt/eval_programme/runA/`.

| Model | COCO @0.05 ↑ | LSP @0.05 ↑ | 3DPW MPJPE / PA ↓ | EHF PA-PVE all / body PA-MPJPE / hands / face ↓ | WB face PCK@.1/.2 ↑ | WB hand PCK@.1/.2 ↑ | S ↑ |
|---|---|---|---|---|---|---|---|
| PEAR (released) | 0.790 | 0.566 | 73.78 / 46.73 | 47.9 / 49.2 / 11.04 / 8.72 | 0.675 / 0.882 | 0.230 / 0.555 | +0.353 |
| L-real | 0.772 | 0.484 | 74.51 / 48.75 | 50.6 / 60.0 / 14.81 / 6.39 | 0.538 / 0.860 | 0.094 / 0.305 | −0.109 |
| A2-ctrl EMA | 0.758 | 0.419 | 80.45 / 50.97 | 45.6 / 49.5 / 13.80 / 5.38 | 0.493 / 0.859 | 0.062 / 0.209 | −0.135 |
| RUN 1 EMA | 0.743 | 0.405 | 83.42 / 51.61 | 47.0 / 51.5 / 14.14 / 6.98 | 0.578 / 0.872 | 0.086 / 0.280 | −0.042 |
| **RUN A EMA** | 0.755 | 0.418 | 77.35 / 48.95 | 47.8 / 52.8 / 15.04 / 6.62 | **0.631** / 0.906 | 0.111 / 0.334 | −0.010 |

S = WB hand PCK@0.1 + WB face PCK@0.1 − EHF hand PA-PVE / 20 (the registered A / B choice score).

Face check (24 frames; mean jaw opening deg / FLAME expression norm): PEAR 4.99 / 18.3, L-real 4.88 / 21.1, A2-ctrl
0.22 / 0.02, RUN 1 7.31 / 5.45, **RUN A 4.11 / 4.47**. Grids `eval_programme/runA/qual_face/`, overlays
`eval_programme/runA/overlays/`.

Reading (descriptive; the registered decision is the A vs B choice once RUN B is evaluated): RUN A is the best PEAR-free
student so far on 3DPW test (PA 48.95, −2.0 mm vs A2-ctrl; MPJPE −3.1 mm) and on WB face / hand 2D, and COCO / LSP are back
at A2-ctrl level. EHF hands got worse (15.04 vs 13.80 mm) and EHF body PA-MPJPE too (52.8 vs 49.5), so the 3D hand gain
is not there; S is the highest so far but still far below PEAR.

## Main comparison (requested 2026-10-20): PEAR paper, PEAR-label student, RUN A, RUN B

| Model | Training labels | COCO PCK@0.05 ↑ | LSP PCK@0.05 ↑ | 3DPW MPJPE ↓ | 3DPW PA-MPJPE ↓ | EHF hand PA-PVE ↓ | EHF face PA-PVE ↓ | WB face PCK@0.1 ↑ | WB hand PCK@0.1 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| PEAR, paper (Tables 2-3) | — | 0.81 | 0.55 | 71.3 | 45.3 | 12.8 | — | — | — |
| PEAR released, measured | — | 0.790 | 0.566 | 73.78 | 46.73 | 11.04 | 8.72 | 0.675 | 0.230 |
| L-real (student with PEAR labels) | BEDLAM2 3D GT + PEAR distillation + COCO/MPII 2D | 0.772 | 0.484 | 74.51 | 48.75 | 14.81 | 6.39 | 0.538 | 0.094 |
| **RUN A** (pure SMPL-X, no PEAR) | dataset GT (BEDLAM2, AGORA, ARCTIC, Motion-X, UBody) + COCO/MPII 2D | 0.755 | 0.418 | 77.35 | 48.95 | 15.04 | 6.62 | 0.631 | 0.111 |
| RUN B (experts, no PEAR) | same images; hands HaMeR + WiLoR, face SMIRK | pending | pending | pending | pending | pending | pending | pending | pending |

"—" = not reported in the PEAR paper. RUN B waits on the SMIRK face labels (RUNLOG 35).
