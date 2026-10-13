# Run L-real: prepared, NOT launched (2026-10-13)

**What it is.** The L-data recipe plus real images with 2D keypoints (COCO train2017 + MPII), as a fixed share of
every batch:
- the real images get a confidence-weighted 2D keypoint loss and PEAR distillation;
- the 3D GT losses see BEDLAM2 rows only (`has_gt = 0` for UBody and real rows).

AI Challenger is skipped, as decided.

**Code (PEAR):**
- `tools/build_real2d_crops.py`
- `tools/mpii_posetrack_scene_match.py`
- `dataset/real2d_dataset.py`
- `student_gt_losses.real_keypoints2d`
- `train_student_gt.py`, which gains a `data.real2d` section
- `configs/phase1_task1/longrun_vitpose_lreal.yaml`

Commits: `76f4feb`, `0475edd`.

## Data (built; measured)

| Source | Rows (train / val) | Unique images | Size | Crops |
|---|---|---|---|---|
| COCO train2017 | 121,302 / 2,424 | 51,597 | 6.1 GB | 512 px context crops, q90 |
| MPII (training images) | 28,163 / 591 | 17,362 | 1.2 GB | same |

Output: `/raid/ubx858/datasets/processed/real2d/{coco,mpii}/`, with `build_summary_{coco,mpii}.json`.

**Raw downloads** are in `/raid/ubx858/datasets/train2d/` (31.4 GB). SHA-256 of the three archives is in `downloads/SHA256SUMS`; the keyframe map is `f0b459cd…`:

| File | Size |
|---|---|
| COCO `train2017.zip` | 19.3 GB |
| MPII `mpii_human_pose_v1.tar.gz` | 12.1 GB |
| MPII annotations `u12_2` | 12 MB |
| MPII keyframe map | 0.2 MB |

**Format:**
- COCO-17 keypoints.
- Confidence: visible 1.0, occluded 0.5, missing 0.
- MPII is mapped to COCO-17 (12 joints; no face points) and converted from MATLAB 1-based to 0-based pixels.

### Exclusions (no image from any evaluation split)

- **COCO.** Only train2017 is used; 0 image names are shared with `coco_val.npz`.
- **Near-duplicates.** A 64-bit dHash is compared with all 14,428 LSP-Extended and COCO-val images; Hamming ≤ 4 is dropped. That dropped 35 COCO images. A visual check found 5 true duplicates (COCO-train vs COCO-val) and 30 false positives, with no true LSP duplicate. For MPII it dropped 11 images, which were not inspected.
- **MPII videos used in PoseTrack val.** Steps:
  1. All 134 PoseTrack-2018-val sequences are `NNNNNN_mpii_test`. NNNNNN is the 1-based MPII image index: 134 of 134 are MPII *test* images, where chance would give about 28%.
  2. The public MPII annotations withhold the video index of test images, so the video rule cannot be applied from the annotations. The keyframe map only gives each image its own clip folder.
  3. We use only MPII training images, but a training image can share footage with a test image. A Hamming-distance-4 dHash hit showed this: training image `048737481.jpg` is a black-and-white copy of the scene in PoseTrack-val sequence 007128.
  4. So `tools/mpii_posetrack_scene_match.py` ranks all 18,076 MPII training images against the 134 PoseTrack-val keyframes by DINOv2 ViT-B/14 CLS cosine. The known match ranks #1.
  5. The top 120 pairs (cosine ≥ 0.811) were inspected visually. Three MPII videos were dropped, all their images, 105 rows (`configs/phase1_task1/real2d_mpii_posetrack_drop_videos.json`):
     - **588**: same aerobics class as sequences 006537, 006538 and 006540;
     - **2538**: same playground as 007128;
     - **2503**: same indoor-rowing event as 003508, possibly another camera, dropped to be safe.
  6. A dHash check against the keyframes dropped one more image, a false positive.
  - **Limitation:** pairs below cosine 0.811 were not inspected, and the PoseTrack clips themselves (other frames than the keyframe) are not available here.

## Loss checks (measured)

**Released PEAR used as the "student"** on 512 val rows per source, with no augmentation:

| Source | Loss | Left/right swapped | Error per keypoint |
|---|---|---|---|
| COCO | 0.043 | 0.184 | 6–9 px except hips |
| MPII | 0.043 | 0.194 | 6–9 px except hips |

Projection convention and handedness are correct.

**Hips.** Annotated hips sit a median 0.15 (COCO) / 0.11 (MPII) torso lengths below PEAR's SMPL-X hips, with no sideways
offset (COCO 16 px, MPII 13 px mean error). This is the same definition mismatch as in the COCO evaluation. The L-real config therefore sets
`hip_weight: 0`; the other 15 keypoints keep their annotation confidence.

**Overlays.** Dataset class and trainer step-0 dumps, both sources, with and without augmentation:
- keypoints sit on the joints;
- left and right are consistent through flips;
- the exact flip test gives 0.0 px.

## Smoke test (measured)

**Run:** 200 steps; 4,000 BEDLAM2 rows and a 4,000-row real subset; GPU 0 (shared with another user's job); exit 0.
Output: `/raid/ubx858/outputs/phase2/lreal_smoke/`.

| Check | Result |
|---|---|
| Batch | 38 BEDLAM2 + 13 UBody + 13 real (GT fraction 0.594 = 38/64) |
| Peak memory | 8.68 GB (L-data: 8.7 GB) |
| Real-image 2D loss | 0.64 → about 0.10 |
| Other losses | all decrease, all finite |
| In-loop 3DPW validation | runs |
| Throughput | 83 img/s on the shared GPU (L-data: 110–117 img/s steady) |
| Real loader alone | 670 img/s with 4 workers (51 steps/s), not a bottleneck |

## Launch proposal (for approval)

```
cd /home/ubx858/guava/third_party/PEAR && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/raid/ubx858/datasets/eval_assets/pylib CUDA_VISIBLE_DEVICES=<free GPU> \
nohup /home/ubx858/miniconda3/envs/guava/bin/python train_student_gt.py \
  --config configs/phase1_task1/longrun_vitpose_lreal.yaml \
  --output-dir /raid/ubx858/outputs/phase1_task1/screen/L_real --device cuda:0 \
  > /raid/ubx858/outputs/phase1_task1/screen/L_real.log 2>&1 &
```

**Batch share: 20% real (13 of 64).** The batch becomes 38 BEDLAM2 + 13 UBody + 13 real; L-data's was 51 + 13. Passes over each source in 128k steps:

| Source | Passes |
|---|---|
| BEDLAM2 | 3.5 (L-data: 4.7) |
| UBody | 3.3 |
| Real (COCO + MPII) | 11.1 |

Alternative: 10% (6 of 64), keeping 45 BEDLAM2 rows. That is about 5 passes over real data, with less 2D signal.

**Run settings:**
- **Steps:** 128,000, as L-data, so the two runs compare at matched steps.
- **Validation:** 3DPW val every 2,000 steps; `best.pt` is chosen by PA-MPJPE.
- **Seed:** 20261009, as L-data.

**Time and memory (estimates, except the measured peak):**
- **Time:** about 23–25 h on an otherwise free L40S. This is an estimate from L-data's 22.7 h; the real rows add one small SMPL-X forward per step.
- **Shared GPU:** up to about 29–30 h, from the 83 img/s smoke rate.
- **Memory:** 8.7 GB peak, measured.
- **Disk:** about 55 GB of periodic checkpoints at peak (0.86 GB each validation, as L-data); /raid has about 215 GB free.
