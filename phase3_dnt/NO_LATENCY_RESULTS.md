# Fixed frame-time experiments

Completed bounded pass, 2026-11-03: **a small tracking improvement is confirmed;
no substantial quality improvement has been verified within 33 ms per frame.**
Keep B28k and the existing tracking-only quality result. All new acceleration
flags remain experimental and opt-in; no final test-set row was launched.

The user requires tracking, pose prediction and full avatar rendering within
33 ms per frame. These experiments use validation data only. B28k remains the
avatar checkpoint; H1 target-frame refinement is excluded from the candidate
live pipeline. No AvatarTest-v1 result is used to select these variants.

## Confirmed quality findings

| Experiment | Validation comparison | Outcome |
| --- | --- | --- |
| H1c tracking, no refinement (H3, existing complete run) | Full ValBench, 50 clips / 3,512 frames: PSNR 18.1030 → 18.2175 dB; LPIPS 0.200897 → 0.198790 | Small gain; fixes two subject switches. |
| Lean driving + CUDA FP32 provider | Fixed five-clip subset, 426 frames: tracking control 16.8356 / 0.233832 → 16.8421 / 0.233415 PSNR / LPIPS | Meets dataset quality non-regression guards. |
| Source fitting 150 → 500 iterations | Same five clips: 16.8421 → 17.0404 dB; LPIPS 0.233415 → 0.231944 | Reject: +0.1982 dB and only 2/5 clips improve; misses registered gain / consistency gate. |
| Offline H1 distillation (H4, concurrent pass) | ValBench-S, 13 held-out clips: arm 17.7885 vs control 17.7534 dB; original SPRINT 17.8488 dB | Reject: +0.035 dB vs control, below +0.30 dB rule and below original checkpoint. |

The five-clip subset deliberately includes the known tracking failures; its
gain must not be presented as a full-split or independent held-out gain.
ValBench UBody clips were used by SPRINT training. ValBench-S clips were held
out, although nine of thirteen share identities with other training clips.
BEDLAM2 validation is unseen by SPRINT. The model's GT-input gap to GART has
not been closed by any of these driving changes.

## Complete-path timing

L40S, full 512px deformation/rasterisation/neural refiner, original B28k
Gaussian capacity. Input RGB is prefetched, matching a live camera handoff;
video decoding, metric warping, PNG writes and display are outside the timed
path. Source fitting / avatar creation and CUDA Graph initialization are
separate from steady-state time. Each arm has 10 warmup and 200 timed frames,
cycling 32 unique native BEDLAM2 validation images with one fixed source avatar.

| First completed timing | Median ms | p95 ms | Frames over 33 ms |
| --- | ---: | ---: | ---: |
| CUDA FP32 detector + eager SPRINT | 53.68 | 54.45 | 100% |
| FP16 person detector + eager SPRINT | 37.32 | 46.93 | 100% |
| CUDA FP32 detector + partial SPRINT graph | 39.05 | 40.10 | 100% |
| FP16 detector + partial SPRINT graph | 32.30 | 33.43 | 15.5% |

None passes the registered median-and-p95 ≤33 ms rule in that run. A later
complete SPRINT graph preserves raw pose, camera and expression fields exactly
against eager execution on the same timing inputs. Its first repeat was
contended by another active job on GPU 0 (both 56.60 / 146.08 ms); it cannot
establish an idle latency claim.

| Subsequent complete candidates | Median ms | p95 ms | Maximum ms | Frames over 33 ms |
| --- | ---: | ---: | ---: | ---: |
| Extended FP32 SPRINT graph + FP16 detector, GPU 0 (contended) | 56.60 | 146.08 | 155.67 | 100% |
| Same candidate, GPU 6 (shared host, idle resident service) | 40.54 | 46.55 | 55.19 | 97% |
| Mixed-precision SPRINT graph + FP16 detector + host camera scalars + local CPU affinity, GPU 6 | 40.23 | 58.45 | 77.00 | 95.5% |

The GPU 6 repeats also fail the runtime rule. The final run passed all 840
finite-pose/render assertions (200 timed + 10 warmup for each of four arms).
Mixed precision changes outputs: its combined arm has maximum rotation-vector
field differences up to 0.001464 radians, expression 0.006297 and camera
translation 0.008984 against eager FP32 with the same FP16 detector.
FP16 detector / mixed-precision image-quality pilots were not launched after
the runtime gate failed. They must not be adopted on numerical checks alone.

Scheduling correction: an intermediate update attributed GPU 6's increased
memory use to another newly active job. That was not established: this
experiment's graph allocations can also increase residency. GPU 6 had an
existing resident service; the cause of its timing variation is unverified.
Only GPU 0's additional active process was directly confirmed. These are
shared-host measurements, with no verified isolated or hard real-time claim.
The original 32.30/33.43 ms result remains a near miss, not an accepted 30 FPS
result. Maximum and p95 both matter; the registered p95 rule itself would not
guarantee every future frame finishes within 33 ms.

## Checks and reproducibility

Person-only class-aware NMS matches the original postprocessing in the CPU
regression test (empty and multiclass / overlap / threshold cases). Raw lean
driving fields exactly matched the original CPU fitter on 32 frames within
the same fitter. Independent rerenders did not give bit-identical PNGs:
426/426 differ, per-clip mean pixel differences 0.017–1.105/255. The strict N1
PNG-equivalence gate therefore fails; source-fit reproducibility is a possible
confound, not a demonstrated explanation.

Commands use the host wrapper because GPU devices and experiment caches are
outside the workspace sandbox:

```bash
EXPERIMENT_GPU=6 bash tools/run_no_latency_experiments.sh live_benchmark --n 200 --name live_fullgraph_gpu6
EXPERIMENT_GPU=6 taskset -c 56-71 bash tools/run_no_latency_experiments.sh live_benchmark --n 200 --name live_amp_host_gpu6 --sprint-amp --host-camera-scalars
/home/ubx858/miniconda3/envs/guava/bin/python -m unittest discover -s tests -p test_sprint_person_detector.py -v
```

Registered rules and failures: [NO_LATENCY_PREREG.md](NO_LATENCY_PREREG.md).
Machine-readable reports and logs:
`/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/no_latency_20261102/`.
H4 results: `O/reports/improve/h4_report.json`; H3 results:
`O/reports/improve/valbench_scores/fb_sprint_k0_trkc_vb.json`, where `O` is the
GUAVA-FB output directory above. No publication or new test-set run was made.

Validation: the NMS regression passed; all four changed Python files parsed;
`git diff --check` passed. Full-path GPU runs exercised source fitting, driving,
graph replay, camera transfer, deformation, rasterisation and neural refinement.
Existing unrelated work and other users' GPU jobs were left intact.
