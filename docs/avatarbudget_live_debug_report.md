# AvatarBudget Live Debug Report

This report summarizes the current AvatarBudget implementation, the live-debug
results, and what we learned from testing PEAR teacher, PEAR student, scout,
temporal prediction, budget routing, and GUAVA rendering.

## Current Status

AvatarBudget has been integrated as a configuration-driven PEAR/student to
GUAVA live runtime.

Implemented pieces:

- One-time source identity loading/fitting path for GUAVA.
- PEAR teacher-output cache infrastructure for student/router training.
- Cached PEAR student training integration in `third_party/PEAR`.
- Temporal pose prediction.
- Always-on cheap scout.
- Render-impact-aware budget router interface.
- Conditional face, hand, and body pose update/merge.
- GUAVA progressive rendering.
- Hard frame scheduler and stage profiler.
- End-to-end live and benchmark commands.
- Unit tests for core AvatarBudget module interfaces.
- Occlusion-rest option for hidden parts.

Important current limitation:

- We did **not** train a learned router MLP yet.
- Live runs without `--router_ckpt` use the hand-written/statistical router.
- The current conditional update skips or runs the shared PEAR/student estimator
  as a whole, then commits selected part outputs. It does not yet split PEAR
  into independent face/hand/body expert backbones.
- 50 FPS is **not verified** by the current runs.

## Main Runtime Files

- `main/live_pear_guava.py`
  - Direct live pipeline.
  - Webcam -> PEAR teacher/student -> GUAVA -> display.
  - Does not use scout, temporal predictor, or budget router.

- `main/live_avatarbudget.py`
  - AvatarBudget live pipeline.
  - Webcam -> scout -> temporal prediction -> budget router -> optional
    PEAR/student update -> conditional pose merge -> GUAVA progressive render.

- `main/debug_pear_motion.py`
  - Diagnostic script.
  - Runs only PEAR/student or PEAR teacher on webcam frames.
  - Prints frame-to-frame pose deltas.
  - Used to decide whether the pose estimator itself is changing.

- `avatarbudget/scout.py`
  - Always-on cheap image observer.

- `avatarbudget/temporal.py`
  - Constant-velocity pose predictor.
  - Now includes an override hook for putting occluded parts into rest pose.

- `avatarbudget/router.py`
  - Statistical or learned render-impact router.

- `avatarbudget/rendering.py`
  - Progressive GUAVA rendering wrapper.

- `avatarbudget/profiler.py`
  - CUDA event and host timing profiler.

- `configs/avatarbudget_rtx3080_laptop.yaml`
  - Runtime configuration for 50 FPS / 20 ms target.

## Direct PEAR Teacher Result

Command used:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/live_pear_guava.py \
  --input 0 \
  --source_data_path assets/example/tracked_image/random_google_pic/blue_shirt \
  --pear_backend teacher \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --pipeline async \
  --pear_stride 1 \
  --window 60 \
  --device cuda:0 \
  --input_crop_scale 1.8 \
  --no-smooth
```

Observed result:

- The avatar moved.
- PEAR teacher produced visible body/hand animation.
- PEAR-to-GUAVA conversion worked.
- GUAVA rendered the teacher-driven avatar correctly.

Representative timing:

```text
PEAR ~5.3 FPS, about 185-190 ms
GUAVA ~5.8-6.2 FPS, about 160-173 ms
end-to-end ~5.2-5.4 FPS
```

Conclusion:

```text
PEAR teacher -> GUAVA works.
GUAVA is not frozen.
The PEAR-to-GUAVA pose mapping is functional.
```

## Direct PEAR Student Result

Command used:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/live_pear_guava.py \
  --input 0 \
  --source_data_path assets/example/tracked_image/random_google_pic/blue_shirt \
  --pear_backend student \
  --student_config configs/student_l70.yaml \
  --student_ckpt /data/GUAVA/pear_student_ckpt/checkpoints/latest.pt \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --pipeline async \
  --pear_stride 1 \
  --window 60 \
  --device cuda:0
```

Observed result:

- The avatar appeared not to move, or moved too little to be useful.
- PEAR/student was being called, but output changes were very small.

Diagnostic command:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/debug_pear_motion.py \
  --input auto \
  --pear_backend student \
  --student_config configs/student_l70.yaml \
  --student_ckpt /data/GUAVA/pear_student_ckpt/checkpoints/latest.pt \
  --precision fp16 \
  --device cuda:0 \
  --frames 30 \
  --skip 5 \
  --warmup 1 \
  --csv outputs/pear_motion_debug.csv
```

Observed student pose deltas:

```text
summary frames=30 mean_delta=0.001159 max_delta=0.002925 changing_pairs=29/29
global_pose mean      0.000695
body_pose mean        0.001353
left_hand_pose mean   0.000771
right_hand_pose mean  0.000835
expression mean       0.006152
jaw mean              0.000168
eyes mean             0.000170
eyelids mean          0.000180
```

Conclusion:

```text
The student checkpoint is not completely constant, but its live pose changes are
too small to produce visible GUAVA animation.
```

Most likely reason:

- The student was trained on many random/person crops, not specifically on the
  current webcam distribution.
- The live webcam input is a full room/camera frame, while PEAR/student expects
  centered person crops.
- Even with `--input_crop_scale 1.8`, the student remained much weaker than the
  teacher for visible live animation.

Important note:

```text
The PEAR student is not subject-specific. It is supposed to be a generic pose
estimator. The failure here is not because the avatar identity is different; it
is because the student output is too close to an average pose on this live input.
```

## AvatarBudget Teacher Result

Command used:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/live_avatarbudget.py \
  --config configs/avatarbudget_rtx3080_laptop.yaml \
  --input 0 \
  --source_data_path assets/example/tracked_image/random_google_pic/blue_shirt \
  --pear_backend teacher \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --device cuda:0 \
  --input_crop_scale 1.8 \
  --no-smooth
```

Observed result:

- The avatar moved after fixing the forced-refresh freeze bug.
- Scout, temporal prediction, router, and progressive rendering all executed.
- The pipeline improved apparent throughput compared with direct teacher mode,
  but did not reach the 50 FPS target.

Representative log:

```text
frames=420 measured=411 e2e=9.95 FPS deadline=41.4%
capture                  40.46/88.15/93.00 ms
upload_preprocess         1.68/2.77/4.10 ms
cheap_scout               1.86/2.88/3.51 ms
temporal_prediction       1.27/1.88/2.98 ms
budget_router_gpu         1.96/5.51/6.89 ms
budget_router_host        2.41/6.68/8.22 ms
pear_estimator          122.44/133.57/140.95 ms
conditional_pose_merge    0.48/0.82/1.72 ms
ehm_target_conversion     2.54/3.32/3.63 ms
guava_deformation        20.86/23.85/28.41 ms
gaussian_rasterization   25.64/34.60/36.91 ms
neural_refiner            0.00/0.00/0.00 ms
readback                  6.49/30.07/34.40 ms
display                   6.00/13.12/16.80 ms
end_to_end_work          97.72/282.34/289.15 ms
frame_release_period    100.93/282.36/289.20 ms
```

Each triplet is:

```text
mean / p95 / p99 latency
```

Conclusion:

```text
Scout, temporal prediction, and router are working.
PEAR teacher is being called.
The avatar animates.
FPS improved from around 5 FPS direct teacher mode to around 10 FPS in
AvatarBudget teacher mode.
50 FPS is not verified.
```

## Freeze Bug Found and Fixed

Initial AvatarBudget runs showed:

```text
overlay: low | predict
pear_estimator 0.00/0.00/0.00 ms
```

Meaning:

```text
The router was predicting/reusing pose and never waking PEAR again.
```

Root cause:

- Camera capture was already slower than the 20 ms frame budget.
- The router marked the frame as deadline-infeasible.
- The live code duplicated the previous output whenever the decision was
  infeasible.
- This also erased forced refreshes from stale parts.
- Result: once over budget, PEAR could be skipped forever and the avatar froze.

Fix:

- `live_avatarbudget.py` now allows duplicated previous frames only when no part
  is forced stale.
- If parts are forced stale, the estimator runs even if the frame misses the
  20 ms deadline.
- The overlay appends `forced` when this happens.

Expected overlay examples:

```text
low | predict
low | face,left_hand,right_hand,body forced
low | face,body forced
```

## Occlusion-Rest Behavior

New option:

```bash
--rest_occluded_parts
--occlusion_visibility_threshold 0.25
```

Command:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/live_avatarbudget.py \
  --config configs/avatarbudget_rtx3080_laptop.yaml \
  --input 0 \
  --source_data_path assets/example/tracked_image/random_google_pic/blue_shirt \
  --pear_backend teacher \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --device cuda:0 \
  --input_crop_scale 1.8 \
  --no-smooth \
  --rest_occluded_parts \
  --occlusion_visibility_threshold 0.25
```

Behavior:

- Scout computes visibility for face, left hand, right hand, and body.
- If visibility is below the threshold, that part can be replaced with a neutral
  rest pose.
- This prevents hidden parts from continuing stale/extrapolated motion.

Overlay examples:

```text
low | predict rest:left_hand
low | body forced rest:right_hand
```

Report output includes:

```json
"occlusion_rest_counts": {
  "face": 0,
  "left_hand": 0,
  "right_hand": 0,
  "body": 0
}
```

## Scout Architecture

File:

```text
avatarbudget/scout.py
```

The scout is currently a cheap heuristic observer, not a trained neural network.

Input:

```text
RGB frame [B, 3, H, W], float, range [0, 1]
```

Step 1: resize to low resolution.

```text
[B, 3, H, W] -> [B, 3, 96, 128]
```

Config:

```yaml
scout:
  height: 96
  width: 128
  roi_expand: 0.12
  motion_gain: 4.0
  appearance_gain: 2.0
```

Step 2: sample four fixed ROIs.

Default normalized ROIs:

```text
face       [0.34, 0.02, 0.66, 0.34]
left side  [0.02, 0.25, 0.40, 0.82]
right side [0.60, 0.25, 0.98, 0.82]
upper body [0.18, 0.12, 0.82, 0.98]
```

Each ROI is represented as:

```text
[x1, y1, x2, y2]
```

where `0.0` is left/top and `1.0` is right/bottom.

Important limitation:

```text
The scout does not yet detect face/hands dynamically. It assumes the subject is
roughly centered.
```

Step 3: compute per-ROI features.

Each ROI is sampled to `16 x 16`, then the scout computes:

```text
feature 0 = mean red
feature 1 = mean green
feature 2 = mean blue
feature 3 = contrast
feature 4 = dx edge/change inside ROI
feature 5 = dy edge/change inside ROI
```

Output:

```text
features [B, 4, 6]
```

Step 4: compute image motion.

The scout stores the previous low-resolution frame internally:

```text
motion = abs(current_low_res - previous_low_res)
```

Then it averages the difference inside each ROI:

```text
motion_score [B, 4]
```

On the first frame there is no previous frame, so:

```text
motion_score = ones
appearance_delta = ones
```

Step 5: compute appearance delta.

```text
appearance_delta =
  mean(abs(current_roi_features - previous_roi_features)) * appearance_gain
```

Output:

```text
appearance_delta [B, 4]
```

Step 6: compute visibility.

```text
contrast = features[..., 3]
brightness = mean(R, G, B)
visibility = clamp(contrast * 8, 0, 1) * (brightness > 0.02)
```

Output:

```text
visibility [B, 4]
```

Step 7: compute confidence.

```text
confidence = 0.25 + 0.75 * visibility
```

Output:

```text
confidence [B, 4]
```

Final scout output:

```text
motion_score      [B, 4]
appearance_delta  [B, 4]
visibility        [B, 4]
confidence        [B, 4]
rois              [B, 4, 4]
features          [B, 4, 6]
```

Part order:

```text
face, left_hand, right_hand, body
```

## Temporal Prediction Architecture

File:

```text
avatarbudget/temporal.py
```

The temporal predictor is also not a trained neural network. It is a
constant-velocity predictor over PEAR pose records.

Input:

```text
previous PEAR/student pose records
```

Pose record contract:

```text
global_pose        [1, 6]
body_pose          [21, 6]
left_hand_pose     [15, 6]
right_hand_pose    [15, 6]
exp                [50]
expression_params  [50]
jaw_params         [3]
pose_params        [3]
eye_pose_params    [6]
eyelid_params      [2]
```

Config:

```yaml
temporal:
  history: 3
  velocity_damping: 0.85
  uncertainty_growth: 0.12
  max_prediction_gap: 3
```

Current implementation:

- Stores up to 3 pose records.
- Uses the latest 2 records for constant-velocity prediction.

If only one pose exists:

```text
predicted_pose_t = pose_(t-1)
```

If two poses exist:

```text
predicted_pose_t =
  pose_(t-1) + 0.85 * (pose_(t-1) - pose_(t-2))
```

Example:

```text
previous hand pose = 10
latest hand pose   = 14
velocity           = 4
prediction         = 14 + 0.85 * 4 = 17.4
```

Uncertainty:

```text
uncertainty =
  mean(abs(latest_pose - previous_pose)) + uncertainty_growth * staleness
```

where staleness is the number of frames since that part received a real
PEAR/student update.

Output:

```text
predicted pose record
uncertainty [4]
```

Part order:

```text
face, left_hand, right_hand, body
```

Important behavior:

- Temporal does not use scout output directly.
- Temporal does not look at the image.
- Temporal only uses previous pose history.
- If PEAR/student is skipped too early, temporal has no real motion history and
  can only repeat the same pose.

## Budget Router Architecture

File:

```text
avatarbudget/router.py
```

The router can be statistical or learned.

In the current tested live runs:

```text
No router checkpoint was passed.
Therefore the router was statistical/rule-based.
```

Router input features:

```text
features [B, 4, 7]
```

For each part:

```text
feature 0 = scout motion_score
feature 1 = scout appearance_delta
feature 2 = temporal uncertainty
feature 3 = normalized staleness
feature 4 = render impact prior
feature 5 = scout visibility
feature 6 = scout confidence
```

Config:

```yaml
router:
  max_staleness: 3
  minimum_risk: 0.35
  render_quality_weight: 0.6
  impact_weights: [1.0, 1.2, 1.2, 0.8]
  weights:
    motion: 1.0
    uncertainty: 1.1
    staleness: 0.8
    render_impact: 1.3
    visibility: 0.4
    appearance: 0.7
```

Render-impact priors:

```text
face       1.0
left hand  1.2
right hand 1.2
body       0.8
```

Risk formula without learned router:

```text
risk =
  1.0 * motion
+ 0.7 * appearance
+ 1.1 * uncertainty
+ 0.8 * staleness
+ 1.3 * render_impact
+ 0.4 * visibility
```

Then the value is normalized and clamped to `[0, 1]`.

Routing decision:

- Mark parts with `risk >= minimum_risk` as eligible.
- Force parts whose staleness reaches `max_staleness`.
- Enumerate all eligible part subsets.
- Enumerate render quality levels:

```text
low, medium, high
```

- Estimate cost for each candidate.
- Keep candidates that fit the current budget.
- Choose the highest scoring candidate.

Cost model:

```yaml
fixed_ms: 1.8
expert_shared_ms: 7.5
face_head_ms: 0.15
left_hand_head_ms: 0.10
right_hand_head_ms: 0.10
body_head_ms: 0.15
render_low_ms: 5.0
render_medium_ms: 7.5
render_high_ms: 10.0
```

Cost formula:

```text
cost =
  fixed_ms
+ render_cost
+ expert_shared_ms if any part updates
+ selected part head costs
```

Example:

```text
update face + body, render low

cost = 1.8 + 5.0 + 7.5 + 0.15 + 0.15
     = 14.6 ms
```

Output:

```text
update_parts
forced_parts
risk [4]
render_level
estimated_ms
deadline_feasible
```

## Learned Router MLP

The code supports a learned router, but it has **not been trained yet**.

If trained and passed with:

```bash
--router_ckpt outputs/avatarbudget/router/router.pt
```

then the router uses this MLP:

```text
Input per part: 7 features

Linear 7 -> 32
SiLU
Linear 32 -> 32
SiLU
Linear 32 -> 1
Softplus

Output:
predicted render damage [B, 4]
```

Training target:

```text
counterfactual GUAVA render damage
```

Meaning:

```text
Render with full teacher pose.
Render again with one part replaced by temporal prediction.
Measure how much the final image changes.
Train router to predict that damage.
```

Current status:

```text
Router MLP training has not been performed.
Live runs currently use the statistical router.
```

## Progressive GUAVA Rendering

File:

```text
avatarbudget/rendering.py
```

Current levels:

```yaml
rendering:
  low:
    gaussian_fraction: 0.35
    refine: false
  medium:
    gaussian_fraction: 0.65
    refine: true
  high:
    gaussian_fraction: 1.0
    refine: true
```

Meaning:

- Low keeps all mesh-vertex Gaussians and samples 35% of UV Gaussians.
- Low skips the neural refiner.
- Medium keeps 65% of UV Gaussians and uses the refiner.
- High keeps all UV Gaussians and uses the refiner.

In the AvatarBudget teacher run:

```text
neural_refiner 0.00/0.00/0.00 ms
```

This means the router selected low render quality, so the refiner was skipped.

## Why `low | predict` Appears

Overlay:

```text
low | predict
```

Means:

```text
low     = GUAVA is using low progressive render quality
predict = no PEAR/student/teacher update for this frame
```

The frame uses temporal prediction instead of a real pose estimate.

Overlay:

```text
low | face,left_hand,right_hand,body forced
```

Means:

```text
Router forced a real estimator update because parts became stale.
```

Overlay:

```text
low | predict rest:left_hand
```

Means:

```text
No estimator update this frame.
Left hand was low visibility, so it was set to rest pose.
```

## FPS Analysis

Direct PEAR teacher:

```text
end-to-end around 5.3 FPS
```

AvatarBudget with PEAR teacher:

```text
end-to-end around 10.0 FPS
```

So AvatarBudget improved the observed frame release rate in the teacher test.

However:

```text
50 FPS is not verified.
```

Reasons:

- Camera capture is slow:

```text
capture p95 around 88 ms
```

- PEAR teacher is slow:

```text
PEAR teacher p95 around 130 ms
```

- GUAVA deformation + rasterization are still significant:

```text
GUAVA deformation p95 around 24 ms
Gaussian rasterization p95 around 30-35 ms
```

- End-to-end p95 is much larger than the 20 ms target:

```text
end_to_end_work p95 around 275-282 ms
```

The correct conclusion is:

```text
AvatarBudget blocks are functioning and improve throughput in the teacher
setting, but this run does not meet the hard 20 ms / 50 FPS requirement.
```

## What Still Needs Work

Highest priority:

1. Make the PEAR student produce teacher-like live pose.
2. Train the render-impact router MLP.
3. Use real measured router costs from the RTX 3080 Laptop instead of priors.
4. Improve capture path, because current webcam capture alone is too slow for
   a 20 ms frame budget.
5. Add dynamic ROIs using previous PEAR pose/camera projection or a small
   detector instead of fixed scout boxes.
6. Optimize GUAVA deformation/rasterization/readback.
7. Run a finite-video end-to-end benchmark before making any 50 FPS claim.

Student-specific next steps:

- Compare teacher and student outputs on the same live frames.
- Train or fine-tune student with live-like centered webcam crops.
- Add stronger collapse diagnostics:

```text
pose variance
hand/body amplitude
student-vs-teacher render difference
temporal jitter
```

Router-specific next steps:

- Generate teacher caches.
- Generate counterfactual GUAVA render labels.
- Train `RenderImpactRouterNet`.
- Pass `--router_ckpt` to `main/live_avatarbudget.py`.
- Compare statistical router vs learned router.

## Bottom Line

What works:

```text
PEAR teacher animates GUAVA.
AvatarBudget scout/router/temporal execute.
Forced refresh bug is fixed.
Occluded parts can now be reset to rest pose.
AvatarBudget teacher mode improves throughput compared with direct teacher mode.
```

What does not work yet:

```text
The available PEAR student checkpoint does not produce strong visible live
motion on the webcam.
The learned router has not been trained.
50 FPS has not been verified.
```

