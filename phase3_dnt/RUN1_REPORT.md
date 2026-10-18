# RUN 1 "face/hand keypoints" — evaluation and the PREREG check (2026-10-18, measured)

Run `/raid/ubx858/outputs/dnt/run1/` (RUNLOG 24): 32k steps from A2-ctrl EMA, GPU 6, 03:02-06:57. Selection (validation
only): best live 3DPW-val PA-MPJPE 49.04 at step 16k; EMA there 49.29 (+0.25 <= 0.3) -> **best.pt EMA reported**.
Evaluation `phase3_dnt/run_full_eval.sh run1 ... ema 6` -> `/raid/ubx858/outputs/dnt/eval_programme/run1/`.

| Model | COCO @0.05 | LSP @0.05 | 3DPW MPJPE / PA | EHF PA-PVE all / body PA-MPJPE / hands / face | WB face PCK@.1/.2 | WB hand PCK@.1/.2 | S |
|---|---|---|---|---|---|---|---|
| PEAR (released) | 0.790 | 0.566 | 73.78 / 46.73 | 47.9 / 49.2 / 11.04 / 8.72 | 0.675 / 0.882 | 0.230 / 0.555 | +0.353 |
| L-real | 0.772 | 0.484 | 74.51 / 48.75 | 50.6 / 60.0 / 14.81 / 6.39 | 0.538 / 0.860 | 0.094 / 0.305 | −0.109 |
| A2-ctrl EMA | 0.758 | 0.419 | 80.45 / 50.97 | 45.6 / 49.5 / 13.80 / 5.38 | 0.493 / 0.859 | 0.062 / 0.209 | −0.135 |
| **RUN 1 EMA** | 0.743 | 0.405 | 83.42 / 51.61 | 47.0 / 51.5 / 14.14 / 6.98 | **0.578** / 0.872 | 0.086 / 0.280 | −0.042 |

S = WB hand PCK@0.1 + WB face PCK@0.1 − EHF hand PA-PVE / 20 (the score proposed for the A / B choice; descriptive here).

Face check (24 frames; mean jaw opening deg / FLAME expression norm): PEAR 4.99 / 18.3, L-real 4.88 / 21.1, A2-ctrl
0.22 / 0.02, **RUN 1 7.31 / 5.45** (UBody frames 9.55 deg vs PEAR 7.91). Grids: `eval_programme/run1/qual_face/`,
overlays `eval_programme/run1/overlays/`.

**PREREG (RUN 1 vs A2-ctrl):**
- Body guard: 3DPW-test PA-MPJPE 51.61 vs limit 51.47 -> **fail (by 0.14 mm)**; COCO PCK@0.05 0.743 vs limit 0.753 -> **fail**.
- Credit: WB face PCK@0.1 +0.085 (>= 0.03) -> **earned**; WB hand PCK@0.1 +0.024 (no); EHF hand PA-PVE +0.34 mm (worse);
  EHF face PA-PVE +1.60 mm (worse).
- **Verdict: RUN 1 does not pass** (guard failed, credit earned on the face only).

Reading: the face now moves (jaw / expression no longer frozen; face 2D PCK +0.085), but body MPJPE got 3 mm worse and
COCO / LSP 2D dropped, i.e. the zoom crops / face-hand terms cost body accuracy. EHF face PA-PVE got WORSE although the
face moves: EHF faces are near-neutral, so a frozen neutral face scores well there; EHF face PA-PVE is not a sensitive
face-quality measure. Hands improved only on 2D (+0.07 PCK@0.2) and not in 3D. Distance to PEAR remains large:
3DPW PA +4.9 mm, MPJPE +9.6 mm, EHF hands +3.1 mm, WB hands −0.14 PCK@0.1.
