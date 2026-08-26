# PEAR student distillation v2 — fixing the constant-pose collapse

## What went wrong in the first run

The `pear_student_l70_kd` student predicted essentially the same skeleton for
every image. Measured on 32 held-out frames from 32 different sequences:

| metric | v1 @ 150k | healthy |
| --- | --- | --- |
| `pose_spread_ratio` (student pose variation ÷ teacher's) | **0.047** | → 1.0 |
| `pose_vs_constant` (student error ÷ error of a fixed average pose) | **1.34** | ≪ 1.0 |
| `blindness_norm` (how much the output moves when the image is replaced by black) | **0.07** | large |
| `deviation_cosine` (does the student move the way the teacher moves?) | **0.03** | → 1.0 |

`pose_vs_constant > 1` means the student was **worse than a program that always
prints the average human pose**.

The per-stage probe showed the failure was not in the backbone:

```
stem          1.29  ####################
stage1..4     ~1.0  ####################   <- backbone reacts strongly to the image
proj          1.00  ####################
transformer   0.34  #######
decoder_token 0.31  ######
pose_out      0.016                        <- information dies here
```

The image signal reaches the head and the head throws it away. Four causes in
the code:

1. **Pose was ~5% of the loss.** `l1_tree` averaged each output tensor and
   summed them unweighted, so `body_pose` was 1 of 21 equal pieces, while
   200-d body shape and 300-d FLAME shape (near-constant identity vectors, and
   FLAME was weighted 2×) dominated. The head starts by predicting the SMPL
   mean pose and learning a correction on top; with pose that weakly weighted,
   "never correct anything" was a good enough optimum. Val loss confirms it:
   5.64 at step 5000, 5.59 at step 150000 — 145k steps of nothing.
2. **The losses that would have prevented this were off.** `--rot_weight` and
   `--joint3d_weight` defaulted to `0.0`, so `rot_deg` and `joint3d_mm` were
   only ever measured, never trained.
3. **The velocity loss paid the student to hold still.** At
   `temporal_stride 2` on 30 fps footage, consecutive frames are 66 ms apart and
   the teacher's velocity is ≈ 0, so a 0.25-weighted loss was explicitly
   rewarding constant output.
4. **Camera supervision was diluted 5×.** `pd_cam` is a 4×4 matrix in which only
   3 translation entries carry information; L1 over all 16 divided the error by
   about five.

Secondary: `BatchNorm` in the backbone with a batch of 8 near-duplicate frames
(train 2.3 / val 5.6 gap), no LR schedule, and only 0.55 epochs of data seen.

## What changed

| file | change |
| --- | --- |
| `train_pear_student_distill.py` | per-parameter loss weights (pose 10×, shape 0.1×); camera loss on translation only; feature distillation against the teacher backbone; 2D joint loss; `rot`/`joint3d` on by default; velocity off by default; warmup + cosine LR; `--init_head_from_teacher` / `--freeze_head`; health probe wired into training |
| `models/backbones/student_backbone.py` | `norm='group'` option (BatchNorm still available), positional embedding, `token_dim` separate from `embed_dim` so the student can emit 1280-wide teacher-compatible tokens cheaply |
| `models/pipeline/student_pipeline.py` | returns backbone features and per-stage activations; can load and freeze the teacher head |
| `models/pipeline/ehm_pipeline.py` | added `forward_features` (the feature-distillation target) |
| `configs/student_l70_v2.yaml` | new config; HEAD identical to `configs/infer.yaml` so the teacher head loads with `strict=True` |
| `tools/student_diagnostics.py` | the probe: per-stage signal, collapse metrics, plots, skeleton strips |
| `tools/check_student_health.py` | run the probe on any checkpoint |

**The biggest single change is head reuse.** Instead of training a 39M-parameter
head from scratch to re-learn a mapping that already exists, phase 1 copies the
teacher's head, freezes it, and trains only the backbone. The task becomes
"produce features the teacher's head understands", which a constant output
cannot satisfy.

## Commands

Set these once:

```bash
cd /home/ubx858/guava/third_party/PEAR
PY=/home/ubx858/miniconda3/envs/guava/bin/python
TEACHER=/home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt
DATA=/raid/ubx858/datasets/processed/pear_student
OUT=/raid/ubx858/outputs
```

### 0. Check the old checkpoint (reproduces the diagnosis above)

```bash
CUDA_VISIBLE_DEVICES=0 $PY tools/check_student_health.py \
  --student_ckpt $OUT/pear_student_l70_kd/checkpoints/step_0150000.pt \
  --student_config configs/student_l70.yaml \
  --teacher_ckpt $TEACHER \
  --frames_root $DATA/frames/val \
  --output_dir $OUT/pear_student_l70_kd/health_check \
  --skeletons --device cuda:0
```

### 1. Phase 1 — train the backbone against the frozen teacher head

```bash
CUDA_VISIBLE_DEVICES=0 nohup $PY train_pear_student_distill.py \
  --student_config configs/student_l70_v2.yaml \
  --teacher_ckpt $TEACHER \
  --train_manifest $DATA/train.jsonl \
  --val_manifest $DATA/val.jsonl \
  --output_dir $OUT/pear_student_v2_phase1 \
  --init_head_from_teacher --freeze_head \
  --batch_size 8 --clip_length 2 --temporal_stride 8 \
  --steps 150000 --lr 3e-4 --warmup_steps 2000 \
  --feature_weight 2.0 --velocity_weight 0.0 \
  --health_every 2500 --skeleton_every 5000 \
  --num_workers 8 --device cuda:0 \
  > $OUT/pear_student_v2_phase1.log 2>&1 &
```

Follow it:

```bash
tail -f $OUT/pear_student_v2_phase1.log
```

### 2. Phase 2 — unfreeze the head and fine-tune everything at a lower LR

Only start this once phase 1 reaches `pose_vs_constant < 0.5`.

```bash
CUDA_VISIBLE_DEVICES=0 nohup $PY train_pear_student_distill.py \
  --student_config configs/student_l70_v2.yaml \
  --teacher_ckpt $TEACHER \
  --train_manifest $DATA/train.jsonl \
  --val_manifest $DATA/val.jsonl \
  --output_dir $OUT/pear_student_v2_phase2 \
  --resume $OUT/pear_student_v2_phase1/checkpoints/latest.pt \
  --batch_size 8 --clip_length 2 --temporal_stride 8 \
  --steps 250000 --lr 5e-5 --warmup_steps 500 \
  --feature_weight 1.0 --velocity_weight 0.1 \
  --health_every 2500 --skeleton_every 5000 \
  --num_workers 8 --device cuda:0 \
  > $OUT/pear_student_v2_phase2.log 2>&1 &
```

### 3. Full evaluation with skeleton sheets (unchanged script)

```bash
CUDA_VISIBLE_DEVICES=0 $PY tools/eval_pear_student_distill.py \
  --student_ckpt $OUT/pear_student_v2_phase1/checkpoints/latest.pt \
  --student_config configs/student_l70_v2.yaml \
  --teacher_ckpt $TEACHER \
  --manifest $DATA/val.jsonl \
  --output_dir $OUT/pear_student_v2_phase1/eval \
  --skeleton --device cuda:0
```

### 4. Re-plot the health curves from a log at any time

```bash
$PY -c "
from pathlib import Path
from tools.student_diagnostics import plot_health_history
plot_health_history(Path('$OUT/pear_student_v2_phase1/train_log.jsonl'),
                    Path('$OUT/pear_student_v2_phase1/health_curves.png'))"
```

## What the training run writes

```
<output_dir>/
  train_log.jsonl            train / val / health records
  health_curves.png          all health metrics over the whole run (auto-updated)
  health/step_XXXXXXX.png    per-stage signal bar chart at each probe
  skeletons/step_XXXXXXX.png input | teacher | student | overlay strips
  checkpoints/
```

## How to read the numbers while it trains

The console prints a probe every `--health_every` steps:

```
--- health probe @ step 20000 ---
  LEARNING: student tracks the teacher's pose variation
  per-stage signal (std/mean across images; collapse -> 0):
    stem            1.29  ####...
    ...
    pose_out        0.15  ####
  pose_spread_ratio      0.9100
  pose_vs_constant       0.3200
  blindness_norm         1.1000
  deviation_cosine       0.7400
  feature_cosine         0.8100
```

Watch these three, in order of importance:

- **`pose_vs_constant`** — must fall below 1.0 early and keep falling. If it sits
  at or above 1.0 after ~10k steps, the run has collapsed again; stop it.
- **`pose_spread_ratio`** — should climb toward 1.0. Below 0.2 = collapsed.
- **`feature_cosine`** — in phase 1 this is what you are actually training;
  it should climb steadily toward 1.0.

`deviation_cosine` near 0 means the student's frame-to-frame changes are
unrelated to the teacher's, i.e. it is not tracking, whatever the loss says.

If a probe ever shows a **backbone** stage below the 0.1 line, the problem moved
upstream (dead CNN stage) and the fix is different — check `dead/<stage>` in
`train_log.jsonl` for the fraction of fully dead channels.

## Note on old checkpoints

`configs/student_l70.yaml` (BatchNorm, no positional embedding, 640-wide) still
loads — the new backbone options default to the old behaviour. v1 and v2
checkpoints are not interchangeable; always pass the config the checkpoint was
trained with.
