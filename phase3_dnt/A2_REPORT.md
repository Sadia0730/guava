# Track A2 — results so far (measured)

## A2-ctrl (control: Track A best.pt + output regularisers, no new data)
Run `/raid/ubx858/outputs/dnt/trackA2/ctrl/` (RUNLOG 16), 32k steps, 2.9 h on GPU 4. best.pt = step 16k (validation
PA-MPJPE 50.32 live / 50.47 EMA); EMA within 0.3 mm -> **EMA weights reported** (PREREG). At the end every regularised
output is neutral on 3DPW-val (head / hand scale 1.00, body |exp| 0.035, FLAME |x| <= 0.002).
Evaluation `phase3_dnt/run_a2_eval.sh` (RUNLOG 17), author's protocol evaluator SHA `9838287b`, EHF fp32.

| Model | COCO PCK @0.05 / @0.1 | LSP-Ext PCK @0.05 / @0.1 | 3DPW test MPJPE / PA (author) | EHF PA-PVE all / PVE all / hands / face / body PA-MPJPE |
|---|---|---|---|---|
| PEAR (released ViT-H) | 0.790 / 0.919 | 0.566 / 0.817 | 73.78 / 46.73 | 47.9 / 66.6 / 11.0 / 8.7 / 49.2 |
| L-real | 0.772 / 0.921 | 0.484 / 0.755 | 74.51 / 48.75 | 50.6 / 83.4 / 14.8 / 6.4 / 60.0 |
| Track A best.pt (live) | 0.580 / 0.846 | 0.355 / 0.649 | 86.66 / 59.61 | 84.8 / 119.5 / 13.9 / 64.5 / 58.4 |
| **A2-ctrl best.pt EMA (selected)** | **0.758 / 0.920** | 0.419 / 0.715 | **80.45 / 50.97** | **45.6** / 86.0 / 13.8 / 5.4 / **49.5** |
| A2-ctrl best.pt live | 0.757 / 0.920 | 0.414 / 0.712 | 82.07 / 51.38 | 48.3 / 87.6 / 14.0 / 5.4 / 52.5 |

**The control alone meets the user-specified primary criterion**: it closes 80 % of Track A's 3DPW-test PA-MPJPE gap
to L-real (59.61 -> 50.97; threshold <= 54.18) and COCO PCK@0.05 is 0.758 (>= 0.74). As registered, A2 is credited
with the new data only if it beats this control by >= 1.0 mm PA-MPJPE (A2 <= 49.97) without losing more than 0.005
COCO PCK@0.05 (A2 >= 0.753).

Remaining gap of the control to L-real: +2.2 mm PA-MPJPE, +5.9 mm MPJPE, −0.014 COCO, −0.065 LSP PCK@0.05. On EHF
the control is better than L-real (PA-PVE all 45.6 vs 50.6; body PA-MPJPE 49.5 vs 60.0, at PEAR's 49.2).
