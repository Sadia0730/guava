# PEAR student data setup

The downloaded data is valid, but it is still archive-only:

- BEDLAM 2.0: 163 GB, 66 MP4 archives plus metadata archives.
- UBody: 14 GB, with `videos.zip`, `splits.zip`, and `annotations.zip`.

Distillation needs RGB video and sequence identity. The frozen PEAR teacher
provides the parameter targets, so the first training stage does not need the
BEDLAM metadata or UBody annotations. Keep those archives for later evaluation.

## 1. Expand only the required files

Run on `desai-lambda1`:

```bash
mkdir -p /raid/ubx858/datasets/expanded/UBody
mkdir -p /raid/ubx858/datasets/expanded/BEDLAM

bsdtar -xf /raid/ubx858/datasets/raw/UBody/videos.zip \
  -C /raid/ubx858/datasets/expanded/UBody
bsdtar -xf /raid/ubx858/datasets/raw/UBody/splits.zip \
  -C /raid/ubx858/datasets/expanded/UBody

find /raid/ubx858/datasets/raw/bedlam2.0 \
  -maxdepth 1 -type f -name '????????_1_*_mp4.tar' \
  -exec bsdtar -xf '{}' -C /raid/ubx858/datasets/expanded/BEDLAM ';'
```

The BEDLAM filename field after the date is the number of people. Starting
with `_1_` keeps the initial experiment single-person and avoids identity
switches. Multi-person BEDLAM should be a separate ablation with a real tracker.

## 2. Check the person detector

The cropper uses EHM-Tracker's YOLOX detector. This file must exist:

```bash
ls -lh /home/ubx858/guava/EHM-Tracker/pretrained/dwpose/yolox_l.onnx
```

If it is missing, use the same package ID as EHM-Tracker's download script.
The server has `bsdtar` instead of `unzip`:

```bash
conda activate guava
cd /home/ubx858/guava/EHM-Tracker
python -m pip install gdown
gdown --id 1g_4YKQvLSWo8yzYHgNstr91RCD4rne8p -O pretrained.zip
bsdtar -xf pretrained.zip -C .
```

## 3. Prepare stable PEAR crops

First run two videos as a smoke test:

```bash
conda activate guava
cd /home/ubx858/guava

python third_party/PEAR/tools/prepare_student_distill_data.py \
  --ubody_root /raid/ubx858/datasets/expanded/UBody \
  --bedlam_root /raid/ubx858/datasets/expanded/BEDLAM \
  --output_root /raid/ubx858/datasets/processed/pear_student \
  --device cuda \
  --limit 2
```

Inspect the generated crops under
`/raid/ubx858/datasets/processed/pear_student/frames/`. They should contain one
centered person with the full body visible. Then run the same command without
`--limit 2`. Completed videos are reused, so the smoke-test work is retained.

The preparation writes:

```text
pear_student/
  train.jsonl
  val.jsonl
  rejected.jsonl
  frames/
    train/{bedlam,ubody}/<sequence>/000000.jpg
    val/{bedlam,ubody}/<sequence>/000000.jpg
```

UBody uses its official inter-scene and intra-scene test identities for
validation. BEDLAM is split by complete scene archive, not by frame or short
clip, preventing adjacent-frame leakage.

## 4. Build the training loader

```python
from dataset.student_distill_dataset import build_distillation_dataloader

train_loader = build_distillation_dataloader(
    "/raid/ubx858/datasets/processed/pear_student/train.jsonl",
    batch_size=8,
    clip_length=4,
    temporal_stride=1,
    train=True,
    num_workers=8,
)

batch = next(iter(train_loader))
images = batch["images"]                 # [B, T, 3, 256, 256], values in [0, 1]
model_images = images.flatten(0, 1)       # [B*T, 3, 256, 256]
teacher_outputs = teacher(model_images)
student_outputs = student(model_images)
```

Training is balanced 50/50 between BEDLAM and UBody even if their clip counts
differ. The validation loader returns every validation clip exactly once.
