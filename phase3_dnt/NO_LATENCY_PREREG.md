# Fixed per-frame time experiments (2026-11-02)

User explicitly authorises experiments and training while requiring no increase
in live per-frame time. All choices use validation data. AvatarTest-v1 is fixed.
This record is separate because a concurrent experiment process is maintaining
PREREG.md and implementing H4 SPRINT distillation. Do not overwrite its work.

## Existing result to verify, not rerun

H3 tracking alone already has complete saved ValBench scores: all-frame PSNR
18.1030 -> 18.2175, LPIPS 0.200897 -> 0.198790; 50 clips / 3,512 frames, no missing
clips. UBody improves; BEDLAM2 unchanged. Preserve source subject selection
disclosure. This pass discovered these files after reading the earlier notes;
the result is not a new run of this pass.

## N1: remove unused raw-driving work

Opt-in backend, same SPRINT checkpoint, image crop, camera features, shape lock,
and inference precision. k=0 driving skips keypoints, matting, mesh evaluation,
and optimisation diagnostics. Source avatar creation still uses the full fit
and matte. Raw driving's part-crop placeholders must never be used for source
creation. No architecture or Gaussian capacity change.

Before adoption compare every returned raw pose/camera/expression field on a
fixed 32-frame timing sample from the first five ValBench BEDLAM2 clips. The
entire raw output must be identical or differ by <= 1e-6. Then rerender a fixed
six-clip validation subset (first two UBody clips, two wrong-person clips,
first two BEDLAM2 clips) with k=0/H1c and verify PNG equality against saved H3
renders. Report all differences. Adopt only if outputs are equivalent and
paired wall-time median and p90 do not increase. CPU provider remains default.

## N2: ONNX Runtime CUDA provider

Same detector and DWPose ONNX weights and all fitting math; only execution
provider changes. Never silently fall back to CPU. First benchmark k=0 lean
and k=60 against matching CPU arms on the same frames. Disclose that comparing
CUDA k=60 to an unoptimised baseline does not establish equal cost against an
equally optimised k=0 control. If CUDA k=60 is faster than the deployed CPU k=0
baseline, evaluate the fixed validation subset, then the full validation set
if promising. Keep quality claims validation-only. Accept CPU->CUDA replacement
only after no dataset regresses by >0.1 dB or 0.003 LPIPS, no non-finite results,
and timing median/p90 do not increase against the named control. Report GPU
memory, provider failures, numeric differences, and all measured arm times.

## Budget and storage

First stage: <15 GPU minutes for timing and equivalence; second stage up to
two GPU hours for validation if warranted. Use an available GPU with <2 GB
resident memory (GPU 2 idle at launch; GPU availability is checked each time). No killing other jobs. Respect the
existing checkpoints/data; only new caches under /dev/shm/ubx858_no_latency_20261102
and reports under O/reports/no_latency_20261102. Training H4 is owned by the
other ongoing pass and is not duplicated here. No test run or publication.

## Runtime amendment (after initial benchmark, before retry)

Initial 32-frame CPU run: raw full vs lean output fields exactly identical;
wall median 3,151 vs 2,935 ms on the CPU-saturated host, not an idle timing
claim. CUDA provider failed because this ONNX Runtime expects CUDA 11 BLAS/FFT
libraries while guava's PyTorch ships CUDA 12. Do not install or alter either
environment: explicitly load only the existing NVIDIA CUDA 11 libraries from
the isolated GART baseline env, then require actual CUDA execution. A new
32-frame benchmark uses four ONNX threads in both full and lean CPU arms,
records host load, and writes benchmark_cuda_threads4.json separately. This
is a runtime repair and thread-count control, not a change to weights or fit
objectives. Measure all arms again; do not combine their times across runs.

## N2 precision correction and N1 export batching (before validation runs)

CUDA ran after the library repair. On the busy host, 32-frame median wall ms:
CPU full k0 2670, CPU lean k0 1828, CPU k60 2827, CUDA lean k0 251, CUDA k60 1002.
These are contention measurements, not idle performance. Default CUDA provider
math gave differences up to 0.064 in raw camera translation, so it is not
accepted as equivalent. ONNX Runtime defaults to TF32; explicitly disable it
(`use_tf32=0`, official CUDA provider docs) and require confirmation from both
sessions before any quality evaluation. New runs use separate row names.

Raw output export is batched into one rotation conversion and one GPU-to-CPU
transfer, preserving all fields and FP32 precision. Repeat the numeric test
against the full baseline before using it; require <=1e-6 CPU-provider error.
Detect an idle GPU at launch rather than hardcoding an index. If other jobs
arrive during a benchmark, disclose contention and do not claim idle latency.

## N3: spend more time once on source fitting, keep k0 driving

Exploratory source-only pilot: 500 source-fit iterations versus the existing
150, same B28k, original SPRINT, H1c, k0 driving, no extra images or GT. Use the
same verified backend in both arms. Live pose network and renderer unchanged.
The fixed validation subset is the union specified in N1 (five unique clips
because Olympic is already among the first two UBody clips). Compare PSNR /
LPIPS on all frames, per dataset and per clip. Proceed to full validation only
if all-frame PSNR gains >=0.20 dB, LPIPS does not worsen, neither dataset loses
>0.10 dB / +0.003 LPIPS, and >=2/3 clips improve. Report creation time separately;
any adoption also requires paired render median/p90 not materially increased
(existing 2% timing-noise tolerance). More setup time is allowed by the user's
per-frame constraint, but must be disclosed. Reject non-finite fits.

## Scheduling note (before remaining correctness / quality runs)

No GPU currently meets the initial <2 GB / idle selection rule. The user's
request authorises the experiments; correctness and bounded validation may use
an explicitly selected shared GPU with sufficient free memory. GPU 3 currently
has ~6 GB of 46 GB resident and 4% utilisation. Do not stop existing jobs. Such
runs are labelled shared/contended and cannot establish idle absolute latency.
The auto-selection default remains conservative. This changes scheduling only,
not any quality or timing acceptance rule.

## N4: lower driving-matte input cost (before smoke or quality runs)

After moving detection to the GPU, 1024-pixel StyleMatte still costs ~70 ms and
60-step refinement ~126 ms in the contended sample. Test StyleMatte at 512
instead of 1024 for driving refinement ONLY. Same keypoints, SPRINT, fitting
iterations/objective and 256-pixel fit mask. Source fitting and source matte
stay at 1024; k0 is unchanged. No target GT or extra frames. This is a distinct
quality/runtime tradeoff, not a numerical-equivalence claim.

Smoke on the first eight already-cached timing frames: both 1024 and 512 arms,
finite fits, keypoint diagnostics and synchronised wall time. Abort if any
non-finite fit. Then the same fixed five-clip subset, GPU k60/matte512 vs GPU
k60/matte1024. Continue only if neither dataset loses >0.10 dB or +0.003 LPIPS,
and driving time decreases. Also compare complete frame time to the original
CPU-heavy deployed pipeline and to fast GPU k0: refinement has additional cost
against the latter and must never be described as free. Any idle absolute
latency/real-time claim requires a separate isolated timing run. Budget: <=
15 GPU-minutes for smoke and subset; full validation only if warranted.

## User timing clarification and N5 (before runs)

The user now specifies **30 FPS, <=33 ms TOTAL per frame**, tracking + SPRINT +
deformation/rasterisation/neural refinement. This supersedes comparison only
to the original CPU-heavy pipeline. H1 (60 fitting iterations) is ruled out.
N4 completed only its eight-frame smoke (1024:243 ms, 512:202 ms; all finite);
no N4 quality/full run will be launched under this requirement.

N1 CPU raw fields match exactly within a fitter on 32 frames. Independent
end-to-end rerenders do NOT have identical PNGs (426 / 426 differ, mean pixel
deltas 0.017..1.105/255 per clip), so do not claim full-pipeline bit equivalence.
Five-clip score differences are small (all PSNR -0.0069 dB, LPIPS +0.00020).
Source-fit reproducibility/thread settings are a possible confound, not an
established cause. N2 FP32 CUDA quality gate passes on this subset. N3 source500
gains +0.1982 dB vs its CUDA source150 control but only 2/5 clips improve: fails.

N5 factorial candidates, k0/H1c/B28k, source150 unchanged:
(a) CUDA FP32 detector + original SPRINT; (b) FP16 driving detector; (c) CUDA
Graph replay of unchanged FP32 SPRINT; (d) both. FP16 detector is a NEW ONNX
artifact from the original weights (keep FP32 input/output); output retains
coordinates/objectness/person score only. The person class has independent
class-aware NMS, so unused other-class outputs can be omitted. Source fitting
keeps the FP32 detector and full fitting. SPRINT graph uses static input
buffers updated each frame, same weights and arithmetic, no extra prediction.

First smoke/latency, fixed 32 cached validation frames; synchronise GPU and
include H1c box selection, crop, SPRINT, all render preparation, deformation,
rasteriser and neural image refinement. Initialisation/warmup is separate.
Use the SAME source avatar for all candidates. Report p50/p95/max and actual
sample count, GPU/CPU load. Then the fixed five-clip quality subset for promising
variants, original FP32 CUDA row as control. Quality guard: neither dataset
loses >0.10 dB / +0.003 LPIPS; all frames finite; source subject policy unchanged.
Runtime gate: p50 AND p95 <=33 ms in the complete path. Explicitly report max
and missed-budget fraction; passing p95 is not a hard real-time guarantee.
Do not adopt a candidate that only meets the time target by omitting stages.
If needed, CPU copies/camera scalar synchronisations may be removed as a
separately documented runtime step, preserving model capacity/resolution.

N5 live timing uses 200 timed frames per arm cycling the same 32 unique cached
validation frames, plus ten untimed warmup frames per arm. Frames are prefetched
as RGB, matching the live camera handoff; disk/video decoding and display are
outside the defined tracking/rendering path. SAME fixed source avatar for all
four arms, full 512-pixel renderer; source and graph initialisation are excluded
from steady state and disclosed. Store median component times to guide any
further runtime change. No H1 or target matte is used.

N5 implementation smoke failures (no timing/quality result): the built-in FP16
converter emitted an unsorted ONNX graph; topologically sort before checking
and saving the new artifact. Full SPRINT CUDA capture failed on torchvision's
normaliser creating CUDA constants. Capture the backbone + context/transformer
token computation instead; keep the identical normaliser and original parameter
decoders outside capture, including their actual cost in end-to-end timing.
No weight, crop, tensor precision, endpoint, or acceptance rule changed.

N5 first completed timing (live_benchmark.json): full FP32 path p50/p95 53.68 /
54.45 ms; half detector 37.32 / 46.93; graph token 39.05 / 40.10; both 32.30 /
33.43 (15.5% frames >33 ms). Therefore none meets the registered runtime gate.
Refine implementation only: cache the fixed normalisation/camera constants,
capture the original decoder arithmetic and raw output packing as well as the
token computation, and omit the unused full-projection matrix product. All
learned layers, weights, precision, crop, and output fields remain the same.
Record raw-output differences against eager FP32 for the SAME fixed avatar and
input frames. Repeat all four arms into live_fullgraph.json, preserving the
failed timing result. Person-only NMS equivalence has passed a CPU regression
test against the original class-aware implementation (empty + multiclass cases).
