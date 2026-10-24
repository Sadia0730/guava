# SPRINT Phase 3 data (proposed, 2026-10-24; batch of 64, 128k steps from scratch)

SMPL-X labels only (SMPLX_NEUTRAL_2020, 200 shape, 50 expression; no FLAME anywhere). Times seen = rows x 128k / unique train rows.
| Dataset | Rows | Share | Unique train rows | Times seen in 128k | Supervision |
|---|---|---|---|---|---|
| BEDLAM2 (all train jobs + tilt repeats x3 > 60 deg / x5 > 120 deg) | 30 | 46.9 % | 1,379,616 | 2.8 | 3D GT: root / body / hands, shape, 3D joints, camera (real K), 2D, full-frame 2D |
| COCO + MPII | 10 | 15.6 % | 149,465 (NeuralAnnot valid 145,646) | 8.6 | NeuralAnnot SMPL-X: root / body (+ hands, jaw, expression on COCO), shape, root-relative 3D joints; 2D 17 kp (TALS); COCO-WholeBody face / hand 2D; zoom crops |
| UBody (own released SMPL-X labels) | 9 | 14.1 % | ~189,000 | 6.1 | 3D pseudo: root / body (legs if full body), valid hands, jaw + expression (SMPL-X 2020); head zoom |
| Motion-X (+ tilt repeats) | 8 | 12.5 % | 366,300 | 2.8 | 3D pseudo: root / body / hands, shape, 2D; jaw + expression (SMPL-X 2020) |
| AGORA v2 (hand fix) | 3 | 4.7 % | 62,569 | 6.1 | 3D GT: root / body / hands, shape, 3D joints, camera (real K), 2D; jaw + expression (SMPL-X 2020) |
| ARCTIC (no ego view, s10 held out) | 2 | 3.1 % | 15,065 | 17 | 3D mocap: root / body / hands (own validity), shape, camera (real K), 2D |
| MOYO (4 train takes; stronger appearance aug) | 2 | 3.1 % | 11,872 | 21.6 | 3D mocap: root / body (+ hands on camel), shape, camera (real K), 2D |
Validation: 3DPW-val (pure SMPL-X mesh, camera input), hard-pose set = BEDLAM2-val tilt > 90 (4,489) + MOYO val (1,504, same
subject / studio as train), BEDLAM2-val, ARCTIC s10. Not used: UBody-EHM (files kept), 3DPW train data, PEAR outputs.
