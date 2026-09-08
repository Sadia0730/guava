# AvatarBudget implementation

`AvatarBudget_CVPR_Pipeline.md` is the system specification. The implementation
keeps PEAR as the teacher or online estimator, keeps the existing CNN-transformer
PEAR student, and keeps GUAVA as the identity/deformation/rendering backend.

## Tensor contracts

The online pose record is unbatched. `TargetBuilder` adds batch dimension 1 when
it converts the record to GUAVA's EHM input.

| Tensor | Shape | Routed part |
|---|---:|---|
| `global_pose` | `[1, 6]` | body |
| `body_pose` | `[21, 6]` | body |
| `left_hand_pose` | `[15, 6]` | left hand |
| `right_hand_pose` | `[15, 6]` | right hand |
| `exp` | `[50]` | face |
| `expression_params` | `[50]` | face |
| `jaw_params` | `[3]` | face |
| `pose_params` | `[3]` | face |
| `eye_pose_params` | `[6]` | face |
| `eyelid_params` | `[2]` | face |
| scout scalar fields | `[B, 4]` | face, left hand, right hand, body |
| scout ROIs | `[B, 4, 4]` | normalized `x1,y1,x2,y2` |
| scout features | `[B, 4, 6]` | RGB mean, contrast, x/y change |
| router features | `[B, 4, 7]` | scout, uncertainty, staleness, impact |
| router damage | `[B, 4]` | counterfactual render damage |

## 1. One-time identity fitting

This runs EHM-Tracker once. Repeating the command reuses
`optim_tracking_ehm.pkl` unless `--force` is passed.

```bash
cd /data/GUAVA
conda activate guava
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/fit_avatar_identity.py \
  --source_image assets/example/tracked_image/random_google_pic/blue_shirt/blue_shirt.jpg \
  --output_dir outputs/avatar_identities
```

Use the reported `tracked_identity` directory as `--source_data_path` below.

## 2. Cache PEAR teacher outputs

Prepare cropped-frame manifests with PEAR's existing
`tools/prepare_student_distill_data.py`, then cache teacher parameters and
backbone features once:

```bash
cd /data/GUAVA
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/cache_pear_teacher.py \
  --manifest /path/to/train_manifest.jsonl /path/to/val_manifest.jsonl \
  --teacher_ckpt /path/to/pear_model.pt \
  --output_dir /path/to/pear_teacher_cache \
  --batch_size 8 --precision fp16 --device cuda:0
```

The cache is sharded, indexed by absolute frame path, stored in fp16, and tagged
with the teacher checkpoint SHA-256. Cached training disables random horizontal
flips because a flipped image would no longer match its cached target.

## 3. Train and evaluate the CNN-transformer student

```bash
cd /data/GUAVA/third_party/PEAR
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python train_pear_student_distill.py \
  --student_config configs/student_l70.yaml \
  --train_manifest /path/to/train_manifest.jsonl \
  --val_manifest /path/to/val_manifest.jsonl \
  --teacher_cache /path/to/pear_teacher_cache \
  --output_dir /path/to/avatarbudget_student \
  --batch_size 8 --clip_length 3 --temporal_stride 2 \
  --velocity_weight 0.5 --steps 300000 --precision fp16 --device cuda:0
```

```bash
cd /data/GUAVA/third_party/PEAR
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python tools/eval_pear_student_distill.py \
  --student_ckpt /path/to/avatarbudget_student/checkpoints/latest.pt \
  --student_config configs/student_l70.yaml \
  --teacher_ckpt /path/to/pear_model.pt \
  --manifest /path/to/val_manifest.jsonl \
  --output_dir /path/to/avatarbudget_student/eval --device cuda:0
```

## 4. Train and evaluate the render-impact router

`cache_render_impact.py` uses `CounterfactualRenderLabeler` to render the full
teacher pose, replace one part at a time with its temporal prediction, render
again, and compute L1, an SSIM surrogate, and silhouette IoU damage.
The part's ROI L1 receives extra weight so small face and hand errors are not
diluted by the full frame.

```bash
cd /data/GUAVA
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/cache_render_impact.py \
  --manifest /path/to/train_manifest.jsonl \
  --teacher_cache /path/to/pear_teacher_cache \
  --source_data_path outputs/avatar_identities/blue_shirt \
  --output /path/to/router_train.pt --render_size 256 --device cuda:0

CUDA_VISIBLE_DEVICES=0 python main/train_budget_router.py \
  --train_data /path/to/router_train.pt \
  --val_data /path/to/router_val.pt \
  --output_dir outputs/avatarbudget/router --device cuda:0

CUDA_VISIBLE_DEVICES=0 python main/evaluate_budget_router.py \
  --checkpoint outputs/avatarbudget/router/router.pt \
  --data /path/to/router_test.pt --device cuda:0 \
  --output outputs/avatarbudget/router/test_metrics.json
```

Without `--router_ckpt`, live mode uses the same configurable risk equation as
the specification. With a checkpoint, predicted counterfactual render damage
replaces that hand-tuned score.

## 5. Live demo

```bash
cd /data/GUAVA
conda activate guava
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/live_avatarbudget.py \
  --config configs/avatarbudget_rtx3080_laptop.yaml \
  --input 0 \
  --source_data_path outputs/avatar_identities/blue_shirt \
  --pear_backend student \
  --student_config configs/student_l70.yaml \
  --student_ckpt /data/GUAVA/pear_student_ckpt/checkpoints/latest.pt \
  --router_ckpt outputs/avatarbudget/router/router.pt \
  --render_size 256 --precision fp16 \
  --compile_targets pear refiner --device cuda:0
```

The scout runs every frame. PEAR/student runs only when at least one routed part
needs refresh. Non-selected parameters come from constant-velocity prediction;
every part is forcibly refreshed after the configured maximum gap. Progressive
GUAVA levels retain every mesh-vertex Gaussian, reduce UV Gaussians, and can
skip neural refinement at the lowest level.

## 6. End-to-end 50 FPS benchmark

Use a finite video so every configuration sees the same frames:

```bash
cd /data/GUAVA
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python main/benchmark_avatarbudget.py \
  --input /path/to/driving_video.mp4 \
  --source_data_path outputs/avatar_identities/blue_shirt \
  --student_ckpt /data/GUAVA/pear_student_ckpt/checkpoints/latest.pt \
  --router_ckpt outputs/avatarbudget/router/router.pt \
  --frames 300 --render_size 256 --device cuda:0 \
  --report outputs/avatarbudget/benchmark_3080_laptop.json
```

GPU stages use CUDA events and report count, mean, p95, and p99. Capture,
routing, readback, display, and the complete frame use a host monotonic clock.
Setup and compile warmup are reported separately. The benchmark only verifies
50 FPS when at least 100 post-warmup complete frames achieve 99% of the target,
at least 99% meet 20 ms, and end-to-end work p99 is at most 20 ms. Isolated
PEAR or GUAVA measurements cannot set this flag.

## Tests

```bash
cd /data/GUAVA
PYTHONNOUSERSITE=1 python -m unittest discover -s tests -v
```
