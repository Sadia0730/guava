# E1 — Where is the 3D error: depth or image plane? (analysis only, 2026-10-13)

**Question.** Split each joint's 3D error into the component along the camera ray (depth) and the
component perpendicular to it (image plane). Compare PEAR, (L-0.1), and (L-0.1) + the A4 no-render
refiner (ii); overall and per part.

**Method** (`phase1/task3/e1_depth_split.py`; output `/raid/ubx858/outputs/phase1_task3/e1/`; 400 s on
GPU 0, shared):
- **Data and joints.** 3DPW **validation**, all 10,413 frames, GT-keypoint crop. The standard
  evaluator's joints (`tools/eval_3dpw_standard.py`, imported unchanged): H36M regressor → 14 LSP joints,
  hip-midpoint centring. The error vectors are therefore exactly the ones MPJPE averages.
- **Split.** For joint *j* with centred error **e**, the ray **r** is the unit vector from the camera
  centre to the *true* joint. That uses the GT SMPL with 3DPW's world translation and camera extrinsics:
  offset = R_cam · trans + t_cam. Depth component d = **e**·**r**; image-plane component
  **p** = **e** − d**r**; |**e**|² = d² + |**p**|².
- **Check.** Mean |**e**| equals each model's standard validation MPJPE: PEAR 77.55, (L-0.1) 78.95,
  refiner 76.71 (A4 eval). The decomposition is exact.
- **Models.** Released PEAR; (L-0.1) `best.pt` (step 102k); refiner = the A4 (ii) checkpoint, step
  20,000, applied once.

## Results (mm; measured)

"Depth share" is Σd² / Σ|**e**|². An isotropic error would give 33%.

| Model | Part | Mean \|e\| (MPJPE) | Mean \|depth\| | Mean \|plane\| | **Depth share** | Joints with \|depth\| > \|plane\| |
|---|---|---|---|---|---|---|
| PEAR | all 14 | 77.55 | 52.38 | 48.28 | **54.9%** | 49.6% |
| PEAR | legs (ankles, knees, hips) | 73.82 | 53.21 | 43.09 | **62.4%** | 51.3% |
| PEAR | arms (wrists, elbows, shoulders) | 80.75 | 51.80 | 52.47 | 49.0% | 47.9% |
| PEAR | thorax + head | 79.17 | 51.62 | 51.33 | 49.8% | 49.8% |
| (L-0.1) | all 14 | 78.95 | 52.91 | 49.29 | **55.2%** | 48.0% |
| (L-0.1) | legs | 70.50 | 49.64 | 41.72 | **61.6%** | 48.3% |
| (L-0.1) | arms | 86.83 | 56.99 | 54.98 | 52.3% | 49.0% |
| (L-0.1) | thorax + head | 80.68 | 50.51 | 54.95 | 47.6% | 44.1% |
| (L-0.1) + refiner | all 14 | 76.71 | 51.60 | 47.72 | **54.8%** | 49.6% |
| (L-0.1) + refiner | legs | 62.41 | 41.79 | 38.92 | **55.4%** | 47.6% |
| (L-0.1) + refiner | arms | 88.93 | 59.24 | 55.69 | 53.5% | 50.0% |
| (L-0.1) + refiner | thorax + head | 82.95 | 58.12 | 50.22 | 58.6% | 54.4% |

### What the refiner changed: (L-0.1) + refiner minus (L-0.1), mm

| Part | Δ mean \|e\| | Δ mean \|depth\| | Δ mean \|plane\| | Δ depth share |
|---|---|---|---|---|
| all 14 | **−2.24** | −1.31 | −1.57 | −0.4 pt |
| legs | **−8.09** | **−7.85** | −2.80 | **−6.2 pt** |
| arms | +2.10 | +2.25 | +0.71 | +1.2 pt |
| thorax + head | +2.27 | **+7.61** | −4.73 | +11.0 pt |

## What this shows

- **The 3D error is depth-heavy for every model.** Depth accounts for about 55% of the squared error,
  against 33% for an isotropic error, and about 62% in the legs. PEAR and the student behave the same
  way. Even PEAR's leg error is mostly along the ray.
- **The refiner mainly fixes leg depth.** Most of its −8.1 mm leg improvement is along the ray (−7.9 mm
  mean |depth|), and the leg depth share falls from 61.6% to 55.4%. This is error that 2D keypoints and
  image losses cannot see, consistent with A0/A3's finding that image-based losses can't rank it.
- **But it moves error into the upper body.** Arms get +2.1 mm worse. Thorax + head gets +7.6 mm worse
  in depth while improving 4.7 mm in the image plane, i.e. the refiner leans the upper body along the
  ray.
- This matches the Phase 2 table under the author's protocol: the refiner improves PA-MPJPE on 3DPW
  but lowers 2D PCK on COCO and LSP-Extended (2D alignment).
- **Implication.** Supervision that constrains depth (3D GT, as BEDLAM2 gives the refiner) is what
  moves these errors. A fix that keeps the leg-depth gain without disturbing the upper body and 2D
  alignment would need the refiner (or the student) to also keep its 2D projection consistent, e.g. a
  2D reprojection term, or training on real 2D data as in run L-real.

## Caveats

- **Validation only.** (L-0.1)'s `best.pt` was chosen on this split.
- **The rays use the true joint positions.** Using the predicted positions would change the split only
  marginally: the ray angle varies by a few degrees across the body.
- **Thorax + head is two joints** (thorax, head top) and is noisier than the other parts.
