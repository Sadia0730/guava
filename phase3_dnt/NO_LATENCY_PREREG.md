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
