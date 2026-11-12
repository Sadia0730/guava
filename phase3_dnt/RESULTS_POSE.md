# Pose regressor (SPRINT) in the format of PEAR's Table 3

Assembled 2026-11-12 from existing runs; no new run. Checkpoint: SPRINT from scratch, EMA, step 128,000, plain SMPL-X head
(the paper's pose regressor; RUNLOG 60 / 61).

Protocol: `tools/eval_author_protocol.py`, our port of the PEAR author's `evaluation_kp.py` (reference script SHA-256 `ff787035…`,
the same for every row measured here): HMR2 evaluation label files and crops, PCK @0.05 / @0.1 on COCO-val (4,648 persons) and
LSP-Extended (9,428), MPJPE / PA-MPJPE in mm on 3DPW-test (35,515). See `phase2/EVAL_SUITE.md`.

| Method | Model | COCO @0.05 ↑ | COCO @0.1 ↑ | LSP-Ext @0.05 ↑ | LSP-Ext @0.1 ↑ | PoseTrack @0.05 ↑ | PoseTrack @0.1 ↑ | 3DPW MPJPE ↓ | 3DPW PA-MPJPE ↓ | AGORA Body MVE ↓ | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OSX † | SMPL-X | 0.70 | 0.87 | 0.42 | 0.73 | 0.82 | 0.90 | 74.7 | 45.1 | 80.2 | 20 |
| Multi-HMR † | SMPL-X | 0.65 | 0.82 | 0.43 | 0.70 | – | – | 78.0 | 45.9 | 59.6 | 10 |
| SMPLest-X † | SMPL-X | 0.71 | 0.91 | 0.40 | 0.74 | 0.82 | 0.91 | 74.8 | 45.8 | 63.5 | 20 |
| PEAR † | EHM-s | 0.81 | 0.94 | 0.55 | 0.81 | 0.87 | 0.97 | 71.3 | 45.3 | 59.3 | 100 |
| PEAR, released weights, measured here | EHM-s | 0.79 | 0.92 | 0.57 | 0.82 | not run | not run | 73.8 | 46.7 | not run | – |
| **SPRINT (ours)** | SMPL-X | 0.74 | 0.91 | 0.43 | 0.74 | not run | not run | 83.1 | 52.1 | not run | not measured |
| SPRINT (ours), calibrated camera ‡ | SMPL-X | 0.74 | 0.91 | 0.43 | 0.74 | not run | not run | 73.8 | 48.7 | not run | not measured |

† Copied from PEAR's paper (Table 3), not re-run here. ‡ SPRINT takes the camera as an input; the main row gives it no extra
information (focal length = image width for every dataset), the ‡ row gives it 3DPW's calibrated intrinsics, which the other
methods' rows do not use. COCO and LSP-Extended have no calibration, so those columns are identical.

Exact values: SPRINT 0.7408 / 0.9145, 0.4276 / 0.7382, 83.07 / 52.08 (calibrated 73.83 / 48.66); PEAR released 0.790 / 0.919,
0.566 / 0.817, 73.78 / 46.73. Files: `/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch_{camwidth,camknown}/author/`,
`/raid/ubx858/outputs/phase2/author/pear/`.

## AGORA (validation, computed here; not comparable with the test-server column above)

Added 2026-11-12 (PREREG "Pose table ... AGORA" + amendment 1; RUNLOG 147). 1280x720 validation images, YOLOX-L boxes, SPRINT
scratch EMA 128k; official AGORA protocol in our own implementation (`tools/score_agora.py`; self-test with the ground truth as
prediction: 7,811 / 7,811 matched, 0 mm, F1 1). 1,077 images, 7,811 valid persons; mm.

| Camera input | Body MVE ↓ | Body MPJPE ↓ | Left / right hand MVE ↓ | Face MVE ↓ | Full-body MVE ↓ | Precision / recall / F1 | Body N-MVE ↓ |
|---|---|---|---|---|---|---|---|
| none (focal = image width; the registered setting) | 121.0 | 119.2 | 58.3 / 59.4 | 62.4 | 181.1 | 0.96 / 0.78 / 0.86 | 140.7 |
| AGORA's focal length (registered diagnostic) | 75.3 | 74.6 | 48.9 / 50.6 | 40.4 | 121.9 | 0.96 / 0.78 / 0.86 | 87.6 |

Hand, face and full-body values are means over matched persons (the official code averages hands and face over missed persons as
zeros: full-body MVE 167.7 / 111.6 that way). Body N-MVE = Body MVE (1 decimal) / F1 (2 decimals), as the official code.

Why the two rows differ: without the real focal length the predicted body is rotated about the vertical axis by an angle that
grows with the person's distance from the image centre (median -23 deg at the far left, +21 deg at the far right for the 28 mm
scenes; +31 deg for the 18 mm scenes), i.e. the output stays in the crop's viewing-ray frame; with AGORA's focal length as camera
input the median residual rotation is under 1 deg on every axis. The same effect explains part of the 3DPW gap between the two
SPRINT rows (83.1 vs 73.8 mm MPJPE). Persons missed by the detector (22 %) are not in Body MVE.

Notes
- The port reproduces PEAR's published row within about 2 PCK points and 2.5 mm (COCO −2.0 / −2.1 pt, LSP-Extended +1.6 / +0.7 pt,
  3DPW +2.5 / +1.4 mm), so differences of that size against the † rows are not meaningful.
- SPRINT is never trained on 3DPW (project rule). Which † methods were is not checked here.
- PoseTrack: the label file is on disk, the 2018 validation images are not (posetrack.net unreachable on 2026-10-13 and 2026-11-12).
- AGORA: PEAR's column matches the test-set leaderboard numbers of the baselines (e.g. OSX 80.2), i.e. predictions uploaded to the
  AGORA evaluation server; the author's script does not cover AGORA. On disk: AGORA train (used to train SPRINT) and validation
  (not used in training); no test images.
- FPS: PEAR's column is "tested on L40S"; SPRINT's forward pass has not been timed alone on an idle L40S.
