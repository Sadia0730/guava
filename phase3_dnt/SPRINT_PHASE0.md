# SPRINT (SMPL-X Pose Regression IN real Time) — Phase 0: data (2026-10-23, measured)

SPRINT is the PEAR-free ViTPose-S student (base recipe: Run A). Its decoder follows PEAR's design (PEAR licence notice
kept for reused code). Code: PEAR 2037486 (augmentation, shape fix, label / subset builder), 2f30f8f (NeuralAnnot),
0e31bf9 (UBody-EHM).

## New / fixed sources
| Source | Access / licence | Size | Rows | Check | Overlays |
|---|---|---|---|---|---|
| NeuralAnnot SMPL-X, COCO train | official Drive (NeuralAnnot_RELEASE / Hand4Whole_RELEASE); **no licence stated for the labels** (Hand4Whole code MIT) | 0.44 GB | 117,878 kept of 121,302 matched (100 %) | reprojection median 5.3 px; hands verified vs COCO-WholeBody (mean hand added: 3.0 vs 5.5 px) | `/raid/ubx858/outputs/dnt/sprint/phase0/neuralannot/coco_row*.jpg` |
| NeuralAnnot SMPL-X, MPII train | as above | 0.07 GB | 27,768 kept of 28,163 (100 % matched); body only | median 5.1 px | `.../neuralannot/mpii_row*.jpg` |
| UBody-EHM (EHM-tracker fits, 269 UBody training clips) | own tracking | 1.1 GB | 36,689 | identity-disjoint from C1 test + UBody test lists (checked); body 7.6 mm vs EHM; shape fit 1.1 mm; hands 14 px (EHM hand scale / offsets dropped); FLAME face labels | `/raid/ubx858/outputs/dnt/sprint/phase0/ubody_ehm/overlay_*.jpg` |
| AGORA v2 (hand-mean bug fixed) | — | 0.14 GB (npz; images reused) | 62,569 | finger bend 71 -> 41 deg; body joints unchanged; stored 2D reproduced < 0.001 px | `/raid/ubx858/outputs/dnt/sprint/phase0/agora_v2/` |
| ARCTIC | — | — | 16,378 (ego view 0 dropped) | face labels off (were zero placeholders), own hand_valid (95 %) | — |
| MOYO | needs user registration at moyo.is.tue.mpg.de (non-commercial) | 1.3 GB labels + ~18 GB for 4 inversion takes | ~3-5k crops | not built | — |
| Harmony4D | open (MIT card) | 17 GB smallest useful zip | — | SMPL only, few hard poses: **skipped** | — |

NeuralAnnot cameras use virtual focal lengths (4-10k px): losses on those rows use rotations, root-relative 3D joints and
2D only (no metric depth). Val rows of COCO / MPII get no labels.

## Hard-pose augmentation and shape fix
- `augment.rotation_mix` ([[p, max_deg], ...]); default unchanged (bit-identical draws). Tilt repeats (x3 > 60 deg, x5 > 120 deg)
  via subset keep lists: BEDLAM2 tilt > 90 deg 2.5 % -> 8.7 % (single), 8.8 % -> 23.8 % (multi); Motion-X 0.9 % -> 2.8 %.
- `loss.shape_fix`: body and FLAME shape detached on rows without a shape label in the 2D keypoint losses and UBody 3D joints.

## Proposed batch (64 rows, 128k steps)
| Dataset | Rows | Share | Samples | Times each unique row seen | Supervision |
|---|---|---|---|---|---|
| BEDLAM2 (all 1.38M train rows + tilt repeats) | 38 | 59.4 % | 4.86M | 3.5 | 3D GT: root/body/hands, shape, 3D joints, camera, 2D |
| COCO + MPII | 12 | 18.8 % | 1.54M | 10.3 | NeuralAnnot 3D: root/body (+ hands, jaw/expr on COCO where valid), shape, root-relative 3D joints; 2D 17 kp; COCO-WholeBody face / hand 2D; zoom crops |
| UBody (pseudo-GT) | 4 | 6.2 % | 0.51M | 2.7 | 3D pseudo: root/body (legs if full body), valid hands, jaw/expr; head zoom |
| UBody-EHM | 3 | 4.7 % | 0.38M | 10.5 | 3D tracker fits: root/body/hands, shape, camera, 2D; FLAME expression / jaw / eyelids |
| Motion-X (+ tilt repeats) | 4 | 6.2 % | 0.51M | 1.4 | 3D pseudo: root/body/hands, shape, 2D; jaw/expr |
| AGORA v2 | 2 | 3.1 % | 0.26M | 4.1 | 3D GT: root/body/hands, shape, camera, 2D; jaw/expr |
| ARCTIC (no ego view) | 1 | 1.6 % | 0.13M | 7.8 | 3D mocap: root/body/hands (own validity), shape, camera, 2D |
Augmentation: rotation_mix [[0.4, 30], [0.15, 90], [0.25, 180]] (~20 % no rotation), flip 0.5. MOYO (if registered) would
take 2 rows from COCO + MPII.
