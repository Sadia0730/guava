# RAGA timing on the Linux RTX 3080 Laptop (16 GB)

Use the evaluated model settings with a short timing window. The default runs **one
clip**, `neuman/bike`, with **10 warm-up and 50 timed animation frames**, cycling
over ten unique test frames. It profiles four RAGA rows separately, including
appearance updates, with a 45-second pause and temperature check between rows.
It does not run the 163-clip quality benchmark. No measured laptop numbers are
included in this kit: GPU execution must be checked on your laptop.

## 1. Get the code and the small input ZIP

The code is on branch `driven_not_tracked` of
<https://github.com/Sadia0730/guava>. Keep the existing working `guava_live`
installation, environments and model assets. For example:

```bash
git clone --depth 1 --single-branch --branch driven_not_tracked \
  https://github.com/Sadia0730/guava.git /data/GUAVA/guava_timing_src
cd /data/GUAVA/guava_timing_src
python phase3_dnt/timing/install_into_package.py /data/GUAVA/guava_live
```

If that source checkout already exists, update it with `git pull --ff-only`, then
rerun the installer. The installer backs up different existing code files under
`guava_live/backups/timing_<UTC timestamp>/` before copying the timing code and
its existing evaluation/live dependencies. It copies no datasets or weights.
`TIMING_CODE_MANIFEST.json` records the source revision and copied hashes.

**Dataset ZIP location on the server:**

```text
/home/ubx858/guava/phase3_dnt/timing/bundles/timing_subset_laptop_v2.zip
/home/ubx858/guava/phase3_dnt/timing/bundles/timing_subset_laptop_v2.zip.sha256
```

The ZIP is **128,613,078 bytes** (about 129 MB). It is deliberately excluded from
Git. Download both files using the chat links or your usual server connection.
For `scp`, substitute the hostname you normally use for this server:

```bash
mkdir -p /data/GUAVA/timing_data
scp YOUR_SERVER:/home/ubx858/guava/phase3_dnt/timing/bundles/timing_subset_laptop_v2.zip /data/GUAVA/timing_data/
scp YOUR_SERVER:/home/ubx858/guava/phase3_dnt/timing/bundles/timing_subset_laptop_v2.zip.sha256 /data/GUAVA/timing_data/
cd /data/GUAVA/timing_data
sha256sum -c timing_subset_laptop_v2.zip.sha256
unzip -n timing_subset_laptop_v2.zip
export DATA=/data/GUAVA/timing_data/timing_subset_laptop_v2
```

`$DATA/manifest.json` must exist. Point `--root` at this directory, **not** its
parent, a dataset subfolder, or the ZIP. Leave these inputs read-only and write
results elsewhere. You do not need to download full datasets for this run.

The ZIP includes one fixed first-sorted clip from each dataset:

| Clip | Reconstruction frames retained | Support frames | Test frames |
|---|---:|---:|---:|
| `ubody/ConductMusic_S15` | 7 | 6 | 12 |
| `peoplesnapshot/female-1-casual` | 7 | 6 | 12 |
| `neuman/bike` (default) | **83, the full evaluated partition** | 6 | 10 |
| `xhumans/00028` | 7 | 6 | 12 |
| `ubc/91-3003CN5S` | 7 | 6 | 12 |

It contains original source/support/test images, evaluated 512-pixel images and
masks, source/support/driving body parameters and cameras, compact tracked LMDBs,
and the original split keys needed by the adapters. Test frames are excluded from
appearance fusion and per-person optimization. Image/mask and LMDB values are
copied from the evaluated inputs; original support images are restored from the
existing raw sources when absent from the earlier timing ZIP. The manifest records
file hashes and provenance. Only NeuMan retains the **complete** reconstruction
partition; use that clip for comparisons involving per-person optimization.

## 2. Activate the existing environment and check paths

```bash
cd /data/GUAVA/guava_live
source /home/virlab/miniconda3/bin/activate guava  # adjust only if installed elsewhere
cp phase3_dnt/timing/laptop.env.example.sh /data/GUAVA/laptop_timing.env.sh
# Edit the copied file to match your installed environments/assets.
source /data/GUAVA/laptop_timing.env.sh
export DATA=/data/GUAVA/timing_data/timing_subset_laptop_v2
python phase3_dnt/timing/preflight.py --root "$DATA" --data-only
python phase3_dnt/timing/preflight.py --root "$DATA" --methods raga --gpu-info
```

RAGA needs the existing B28k appearance checkpoint, pose checkpoint, SMPL-X/body
assets, DWPose/YOLOX and StyleMatte. Fine-tuning and the live UV worker also need
ArcFace. These licensed/large assets and Python environments are **not in the
input ZIP or this Git commit**. `RAGA_FB_CKPT`, `POSE_CKPT`, `RAGA_ASSETS` and
`ARCFACE_ONNX` select existing files. Use the existing laptop package to supply
the remaining model assets/configs; this is a code overlay, not a new environment.

Do not automatically upgrade Torch, change precision or replace dependencies.
CUDA extensions built only for the server GPU may need rebuilding using each
method's existing installation procedure for the RTX 3080 Laptop's architecture
(`sm_86`). The preflight checks files and CUDA identity, not extension execution.
Do not copy all baseline archives/environments just to measure RAGA.

## 3. Run the short RAGA profile, then collect the table

Close other GPU applications, use AC power and the same laptop power mode for all
methods. Keep the model resolution/precision/configuration unchanged.

```bash
cd /data/GUAVA/guava_live
export RUN=/data/GUAVA/timing_results/raga_$(date -u +%Y%m%dT%H%M%S)
python phase3_dnt/timing/run_suite.py --root "$DATA" --out-dir "$RUN" --methods raga --dry-run
python phase3_dnt/timing/run_suite.py --root "$DATA" --out-dir "$RUN" --methods raga
python phase3_dnt/timing/collect_timing.py "$RUN" --out-dir "${RUN}_report"
```

Results are fresh directories; previous measurements are never overwritten. The
report contains `runtime.tex`, `runtime.md`, `summary.csv`, `phases.csv` and JSON
provenance. Keep raw JSONs, logs, thermal records and the code/asset configuration.
There is **no macro PSNR or quality reevaluation** in this timing report.

The four rows are photo initialization with supplied parameters, photo initialization
with estimated parameters, and each of these **with appearance update**. For the
updated rows, six reconstruction support observations are processed before the
animation timer; animation uses the final refreshed avatar. Source construction
uses one warm-up and three repetitions. Update timings use three warm-ups with
discarded accumulator state, then the six unique supports once each.

The suite waits for GPU <=75 C and CPU <=80 C before each job, and stops its own
job process group if sampled GPU >=86 C or CPU >=95 C. It polls once per second
and records available telemetry; unavailable sensors remain null. This reduces
unnecessary heating but does not guarantee absence of throttling. Inspect clocks,
temperature and power records; keep a stopped run and label it incomplete. You can
increase `--cooldown 90` without changing the model. Do not shorten a training
schedule and present it as the evaluated reconstruction cost.

## 3b. Recommended laptop run (do this; it covers every row that fits the laptop)

LHM, LHM++ and IDOL need about 15-17 GB of GPU memory (measured on the server), so they are timed on a server GPU,
not here. GART, GaussianAvatar and ExAvatar are not installed on the laptop; they are timed on the server too.
Everything below writes a new folder; if a step fails, keep its folder and log and rerun only that step into a new one.

```bash
cd /data/GUAVA/guava_live
source /data/GUAVA/laptop_timing.env.sh
export DATA=/data/GUAVA/timing_data/timing_subset_laptop_v2
python phase3_dnt/timing/preflight.py --root "$DATA" --methods raga,guava,ft --gpu-info
T=/data/GUAVA/timing_results/$(date +%Y%m%d)

# a) our four rows (estimated pose with the fixed centre crop, as in the live app)
python phase3_dnt/timing/run_suite.py --root "$DATA" --out-dir $T/raga --methods raga

# b) the two estimated-pose rows again with the person detector on every frame (as the table's quality rows)
for r in fb_sprint fb_sprint_update; do sleep 60
  bash phase3_dnt/timing/run_ours.sh "$DATA" $T/detector/$r --rows $r --box yolox --tag yolox; done

# c) released GUAVA (needs assets/GUAVA/checkpoints/best_160000.pt; else extract backup_for_desktop/models/guava_released.tar)
python phase3_dnt/timing/run_suite.py --root "$DATA" --out-dir $T/guava --methods guava

# d) our fine-tuned rows: 500 training steps on neuman/bike, then animation (needs live_assets/arcface/glintr100.onnx)
python phase3_dnt/timing/run_suite.py --root "$DATA" --out-dir $T/ft --methods ft --allow-training

# e) report
python phase3_dnt/timing/collect_timing.py $T/raga $T/detector/* $T/guava $T/ft --out-dir $T/report

# f) optional: the live app's own frame rate, without / with the appearance update (section 6)
python phase3_dnt/timing/run_live_replay.py --root "$DATA" --out-dir $T/replay \
  --python "$GUAVA_PY" --assets "$RAGA_ASSETS" --frames 120 --warmup 20 --workers off,uv

# g) send everything back
scp -r $T YOUR_USER@YOUR_SERVER:/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/timing/laptop_$(date +%Y%m%d)
```

## 4. Interpret the phases correctly

- **Reconstruction:** input preprocessing, source pose/keypoints/regression/fitting
  where used, segmentation where used, appearance inference and Gaussian assembly.
  Model loading is outside the feed-forward construction timer. Estimated source
  fitting uses 150 refinement iterations; it is not hidden inside a 55 ms claim.
  Source preprocessing includes the existing tracked-record serialization path.
- **Animation:** preprocessing/crop and parameter conversion, pose regression when
  used, deformation (body-model evaluation plus Gaussian skinning), rasterization,
  and image refinement where present. RAGA output remains 512x512. Original PNG
  decoding/capture, display and background workers are outside this component timer.
  GART's existing adapter includes its evaluated device-to-host output copy; other
  adapters retain their evaluated output boundary, recorded in each JSON. Account
  for this boundary difference when presenting cross-method latency comparisons.
- **Segmentation:** supplied-input rows reuse supplied masks and show null. Raw
  target driving uses no new segmentation or iterative fitting. Fresh segmentation
  is timed during estimated source/support fitting. Null means absent/unmeasured,
  not a fabricated zero.
- **Appearance update:** support detection/fitting (60 iterations, source shape
  locked) where used, preprocessing, feature extraction, canonical fusion and
  decoder refresh. Fits with empty mattes/nonfinite error or error >20 pixels are
  rejected; rejection cost and acceptance counts are retained. This component profile
  refreshes after each accepted support. The offline benchmark uses prefix-based
  fusion/refresh; the experimental worker has separate gating and scheduling.
  These are different policies, not interchangeable timing claims.
  Source feature extraction and accumulator allocation are separately recorded as
  `update_initialization`; they are not hidden inside the per-support update median.
- **Parent/child quantities:** `pose_fit` contains keypoint, segmentation,
  regression and refinement children; `rendering` contains raster/refiner children.
  Do not add parent and child times. Aggregated stages sum paired samples before
  computing statistics. A sum of stage medians need not equal the total median.
- **Latency and FPS:** per-phase CUDA synchronization perturbs execution. The compact
  LaTeX table labels reciprocal phase-timed latency as a **timed rate**, not live FPS.
  RAGA additionally has an independent serial pass with no internal phase barriers;
  its latency/rate are in the CSV/JSON. These rates exclude capture/display and
  background-worker contention. For application throughput, use the separate replay
  below. See [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
  and [CUDA asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution).

## 5. Optional baseline and per-person timing

First set the paths in your copied environment file to the **existing** method
installations. LHM, LHM++ and IDOL require their matching environment, not the
GUAVA environment; that environment is not supplied by this kit. Check the
selected assets with `preflight.py --methods guava,lhm,lhmpp,idol`.

```bash
python phase3_dnt/timing/run_suite.py --root "$DATA" \
  --out-dir /data/GUAVA/timing_results/photo_baselines_01 --methods guava,lhm,lhmpp,idol
```

GART, GaussianAvatar, ExAvatar and the 500-step RAGA fine-tuning row require actual
per-person optimization to measure reconstruction time. **A few test frames do not
make those optimization budgets short.** They are excluded from the default run.
If you need them, request one method at a time; the wrappers preserve the evaluated
schedules and retain their working checkpoints/logs:

```bash
python phase3_dnt/timing/run_suite.py --root "$DATA" \
  --out-dir /data/GUAVA/timing_results/ft_01 --methods ft --allow-training
# Replace ft with gart, gaussianavatar or exavatar only when their environments are ready.
```

Fine-tuning uses supplied reconstruction supervision even for its estimated-driving
row; it is not fully automatic reconstruction. Its three rows reuse one training
run, whose timing is separately identified. Do not infer per-person reconstruction
cost from a pre-trained checkpoint's render timing. Optional early-stop flags in
baseline Python scripts are debugging only; the collector rejects shortened runs.
Do not use the other four truncated reconstruction partitions for the per-person
table. For photo-only profiles on another clip, use `--clip ubc/91-3003CN5S` (etc.).

Collect multiple completed result directories from the **same GPU/configuration**:

```bash
python phase3_dnt/timing/collect_timing.py /data/GUAVA/timing_results/raga_RUN \
  /data/GUAVA/timing_results/photo_baselines_01 --out-dir /data/GUAVA/timing_results/combined_report_01
```

## 6. Optional actual application replay FPS, worker off versus appearance updates

```bash
python phase3_dnt/timing/run_live_replay.py --root "$DATA" \
  --out-dir /data/GUAVA/timing_results/replay_01 --python "$GUAVA_PY" \
  --assets "$RAGA_ASSETS" --frames 120 --warmup 20 --workers off,uv
```

This runs the existing application with center driving crop, parallel pose/render,
no pacing/display, and the experimental UV worker off/on. It makes a short cyclic
JPEG replay from the subset and records actual `fps_observed`, application stage
latencies, worker acceptance/update/rollback records and thermal telemetry. Source
initialization still uses its detector and fitting. Inspect worker counts: a short
run may accept no updates. A replay rate alone does not establish sustained adaptive
30 FPS, webcam capture latency, or update quality. Keep the off/on results separate
from the offline component table. Increase replay length only if needed to observe
the worker, with a cooldown. Do not reuse the earlier unverified 51/38 FPS values.

## Validation and scope

Server validation is CPU-only: syntax/help checks, timing-statistics/collector tests,
dry-run commands, ZIP integrity, split disjointness and input hashes. No GPU timing,
training or quality experiment was launched for this kit. All GPU paths still need
their first laptop run; stop and retain logs on an installation/API failure. Fix
paths/dependencies without silently changing evaluated settings. Manuscript tables
and existing experiment outputs were left unchanged.
