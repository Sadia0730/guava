# RUN A vs RUN B — evaluation and the registered choice rule (2026-10-21, measured)

RUN A: `/raid/ubx858/outputs/dnt/runA/` (RUNLOG 36), EMA step 32k. RUN B: `/raid/ubx858/outputs/dnt/runB_main/` (RUNLOG 42),
selection (validation only): best live 3DPW-val PA-MPJPE 48.30 at step 6k, EMA there 47.83 -> **EMA step 6k**.
Evaluations `phase3_dnt/run_full_eval.sh` -> `eval_programme/{runA,runB}/`.

| Model | COCO PCK@0.05 ↑ | LSP PCK@0.05 ↑ | 3DPW MPJPE ↓ | 3DPW PA-MPJPE ↓ | EHF PA-PVE all / body PA-MPJPE ↓ | EHF hand PA-PVE ↓ | EHF face PA-PVE ↓ | WB face PCK@0.1 ↑ | WB hand PCK@0.1 ↑ | S ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| PEAR, paper | 0.81 | 0.55 | 71.3 | 45.3 | — | 12.8 | — | — | — | — |
| PEAR released (measured) | 0.790 | 0.566 | 73.78 | 46.73 | 47.9 / 49.2 | 11.04 | 8.72 | 0.675 | 0.230 | +0.353 |
| L-real (PEAR labels) | 0.772 | 0.484 | 74.51 | 48.75 | 50.6 / 60.0 | 14.81 | 6.39 | 0.538 | 0.094 | −0.109 |
| RUN A (dataset labels) | **0.755** | **0.418** | 77.35 | 48.95 | 47.8 / 52.8 | 15.04 | 6.62 | **0.631** | **0.111** | **−0.010** |
| RUN B (expert labels) | 0.747 | 0.404 | **77.26** | **48.31** | **43.5 / 47.2** | **14.50** | **6.13** | 0.566 | 0.094 | −0.065 |

S = WB hand PCK@0.1 + WB face PCK@0.1 − EHF hand PA-PVE / 20. Face check (24 frames, jaw deg / FLAME expression norm):
RUN A 4.11 / 4.47, RUN B 4.67 / 6.66 (PEAR 4.99 / 18.3).

**Registered choice rule:** body guard = 3DPW-test PA-MPJPE within 0.5 mm of the better (B 48.31 -> limit 48.81) AND COCO
PCK@0.05 within 0.005 of the better (A 0.7546 -> limit 0.7496):
- RUN A: PA-MPJPE 48.95 > 48.81 -> **fails** (by 0.14 mm); COCO passes.
- RUN B: COCO 0.7469 < 0.7496 -> **fails** (by 0.0027); PA-MPJPE passes.
- **Neither run passes the guard; the rule does not define this case.** S (descriptive): A −0.010 > B −0.065.
Decision left to the user (not resolved post hoc by me).

Qualitative LSP grid (PEAR / L-real / RUN A / RUN B): `eval_programme/lsp_qual_main_AB/`.
