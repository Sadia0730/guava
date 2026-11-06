---
name: raga
description: Playbook for RAGA, the user's real-time monocular full-body avatar (MJPG webcam → centre crop → SPRINT pose → GUAVA-FB Gaussian avatar, async), run by phase3_dnt/live/live_sprint_guavafb.py on the laptop package guava_live. Use it for any work on the live app or laptop package, even if the user doesn't say "RAGA" — speeding it up, profiling or timing it, FPS, latency, frame drops, timing JSON, the webcam or MJPG, the crop box or detector, fp16 or precision, CUDA streams, async, CUDA graphs or torch.compile, the live appearance update (--worker uv), image quality, blur or colour of the avatar, or explaining its FPS. RAGA must run at 27–30 FPS end to end with no measurable loss in pose or avatar quality, and this skill shows how to prove both.
---

# RAGA: 27–30 FPS, no quality loss

RAGA is the user's live full-body avatar. A webcam frame goes through a person crop, then SPRINT (body, hand and face pose,
k = 0, body shape locked to the session) and GUAVA-FB (a Gaussian avatar, ~207k Gaussians + StyleUNet refiner, rendered
at 512 px, built once from the reference frame), and onto the screen. It is run by
`phase3_dnt/live/live_sprint_guavafb.py`, on the laptop from the extracted package `/data/GUAVA/guava_live`
(Razer x Lambda TensorBook, RTX 3080 Laptop GPU 16 GB). `phase3_dnt/live/update_package.sh PKG` copies the repo's live
scripts into the package; edit the repo copy, then update the package.

**The research goal: improve the monocular avatar during inference, from the frames of the stream.** The avatar is
built from one reference frame, so its back and sides are guesses. While the user moves, the stream shows those parts,
and the live update (`--worker uv`, worker6 / uv_fusion.py) is meant to fold them into the avatar, and to improve what
was already seen, without pausing the live loop. Speed work (crop, async, fp16) serves this goal: it buys the time budget
the update needs. See "Improving quality from the stream" below.

**Three design decisions are the user's and are fixed:** MJPG capture, a fixed centre crop instead of YOLOX, and async
SPRINT and rendering. All three are the defaults of the live script since 2026-11-06 (`--fourcc MJPG`, `--parallel`;
`--crop center` is still a switch because the YOLOX crop stays the quality baseline). Don't reopen them unless a
measurement shows that one costs quality, and then report that to the user; never switch a choice back silently.

The project has one constraint with two halves: **27–30 FPS end to end, and no measurable loss in output quality.**
- A speed-up that costs quality fails.
- A feature that drops the rate below 27 FPS fails. That includes the update worker: on session_01 it halved the live
  rate (8 → 4.1 FPS), so its GPU and CPU cost count against the budget.
- The point of RAGA is that neither rule is relaxed to meet the other. A change that fails one is reported, not adopted.

Back both halves with numbers. Feelings are misleading on this laptop:
- Heat or battery power can halve the frame rate without any code change.
- A pose that is a few pixels off looks fine on screen but drags the avatar.

## Before calling a change done

1. **Speed.** Report `fps_observed` and the per-stage p50 / p95 from `--timing-json`.
   - Use a `--no-pace` replay for the A/B comparison, because it is repeatable.
   - Use a live webcam run for the final claim, because the camera, the capture thread and the display only exist live.
   - `scripts/check_raga.py speed TIMING.json [--record DIR]` applies the strict 30 FPS form of the rule: observed FPS
     ≥ 29.5, display-to-display interval ≤ 34 ms at the median and ≤ 40 ms at p95, ≥ 98% of camera frames processed,
     and it warns above 100 ms capture-to-display latency (an async pipeline adds about one frame; above ~100 ms is
     felt). The user's stated target is 27–30 FPS; say which form a run meets.
2. **Quality.** Run `scripts/score_replays.py` on replays of the same recording, with the current pipeline as the
   baseline.
   - It passes when no metric is worse than the baseline beyond run-to-run noise.
   - If it fails, report the failure with its numbers; don't explain it away. The user chose "no measurable loss", so a
     small but real loss is a fail, not a judgement call.
   - For a change that should not alter renders at all (async, buffers, CUDA graphs, the decode path),
     `scripts/check_raga.py same REF_RUN RAGA_RUN --floor REF_RUN_2` is a quick screen: pass when the renders are
     byte-identical, or their median PSNR to the reference is within 0.5 dB of a second reference replay's. The floor is
     needed because the reference path is not bit-deterministic (two replays of one recording came out 38–44 dB apart).
     The gate above is the real judgement.
3. **Conditions.** Log the GPU temperature, SM clock and power source with every timing, so a slow run can be told apart
   from a slow change.

When reporting:
- Lead with the verdict: is 27–30 FPS met, and is quality unchanged?
- Then give baseline vs candidate numbers (FPS, frame interval p50 / p95, latency, dropped frames, PSNR / SSIM / LPIPS
  and pose deltas) and the logged conditions.
- Name any check you haven't run yet; a replay-only result is not a live result. Laptop and webcam numbers are trials,
  not registered results.

## Making an A/B pair

Add the change behind a command-line switch whose default is today's behaviour. The baseline and the candidate are then
the same script, run with the same replay command, so nothing but the change differs between them.

```bash
cd /data/GUAVA/guava_live
PY=/home/virlab/miniconda3/envs/guava/bin/python      # the system python lacks the dependencies
REC=/data/GUAVA/rec/session_01                        # the standard A/B recording (references/findings.md §8)
OUT=<session scratchpad>/raga_ab                      # ~45 MB per replay; keep it out of the repo
$PY phase3_dnt/live/live_sprint_guavafb.py --input $REC --worker off --no-display --no-pace \
    --record OUT/base --timing-json OUT/base/timing.json            # then OUT/cand with the new switch
```

- **`--no-pace` is essential for throughput.** Without it, the replay waits for the recorded capture times, and
  session_01 was captured at 10 FPS. Every run would then read about 10 FPS whatever changed.
- **Replays build the same avatar** from the session's `reference.png`, so differences come from the change.
- **Run a second baseline** (`OUT/base2`) to measure run-to-run noise. Alternate the run order when repeating, because
  the GPU heats up during a run.
- **Replay FPS slightly understates live throughput.** Replays decode JPEGs inside the loop, and `--record` encodes
  JPEGs in a background thread. Live capture decodes in its own thread. The costs are the same for both runs of a pair.
- **Async or pipelined changes render a pose from an earlier frame.** Keep each frame's pose with that frame: render(t)
  uses SPRINT(t) and is recorded under frame t's index. Pairing a new camera frame with an older pose is a quality loss
  (misalignment) even though it looks faster, and the quality check would charge the lag as error (100–400 ms on
  session_01).

Final live check:
```bash
$PY phase3_dnt/live/live_sprint_guavafb.py --input 0 --crop center --worker off --timing-json OUT/live_timing.json
```
- `--print-fps 30` (the default) prints a `[fps]` line to the terminal every 30 frames: the smoothed rate, the mean since
  warm-up and the stage times. The summary JSON printed at the end has `fps_observed` (first to last displayed frame).
- Press `q` in the window to stop. Stand inside the green square when using `--crop center`.
- `--lock-exposure` freezes auto exposure and white balance 1.5 s before the reference capture; `--ref-best-of N` picks
  the sharpest, stillest of the last N countdown frames. Both are logged in `session.json`. Use them: the reference on
  session_01 was over-exposed (findings §9c).
- Speed runs go without `--record` when possible; the recorder's JPEG writes cost CPU. Count dropped camera frames on a
  separate recorded run (`check_raga.py speed --record DIR`).

Conditions, logged next to every timing:
```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader; \
  powerprofilesctl get; cat /sys/class/power_supply/AC0/online
```
- On battery or in power-saver mode, the GPU is held to 54 W instead of 80 W.
- At ~87 °C and above, the SM clock drops to 210–780 MHz.
- One replay logged 121 ms for YOLOX against 28 ms on a cool GPU; with nothing logged, the run couldn't be explained.

## Measuring quality

```bash
$PY /data/GUAVA/.claude/skills/raga/scripts/score_replays.py --session $REC \
    --runs base=OUT/base cand=OUT/cand base2=OUT/base2 --repeat base2 --out OUT/gate
```

All measures use the same frames of every run:
- **Pose.** The avatar's driving pose (the recorded SPRINT output) is projected into the frame and compared with DWPose
  keypoints. Errors are in frame pixels, for body, face and hands. DWPose runs on the baseline's box, so every run is
  judged against the same reference.
- **Avatar.** PSNR, SSIM and LPIPS compare the recorded render with the camera frame cut out by a StyleMatte matte.
  - Where a run's crop differs from the baseline's (detector or crop changes), both runs are scored in both crops.
  - The reason is that resampling a render blurs it, which raises PSNR and SSIM by about as much as the tolerances.
  - LPIPS is the measure that catches blur; PSNR and SSIM can rise when a render gets blurrier. This is why every metric
    has to pass, not just one.
- **Pass rule.** Each metric may be worse than the baseline by at most 0.05 dB PSNR, 0.002 SSIM, 0.002 LPIPS, or 2% pose
  error per group.
  - With `--repeat`, each tolerance widens to twice the baseline-vs-repeat difference, when that is larger. That follows
    the run-to-run noise.
  - With a `--worker uv` run, only the frames that the uv update never used are scored.
  - The project's own scorer, `phase3_dnt/live/score_ablation.py --session REC --runs a=.. b=.. --out OUT --video`,
    gives PSNR / SSIM / LPIPS with a turned-away subset and a side-by-side video; it has no pose score and no crop
    handling.
- **Limits.** `--worker pose` drives the avatar with a corrected pose that the recorder doesn't save, so the pose score
  then covers SPRINT's raw output only. The score covers one session; mention that when the change could behave
  differently with other motion or framing.
- The scorer runs DWPose on the CPU and took this laptop to 100 °C; it has a heat guard (`scripts/heat_guard.py`) and
  `--dwpose-threads` (use 4–6). Budget an hour or more on a hot laptop.

Pose and quality experiments that don't need a modified live script:
- `scripts/sprint_crop_test.py REC OUT` runs SPRINT on the recorded frames with different input crops and scores each
  against DWPose. Use the guava env; it takes ~1 min.
- `scripts/box_staleness.py REC` measures how far a reused or fixed box drifts from the fresh YOLOX box.
- `scripts/camera_fps.py` shows the webcam's frame rate per pixel format. It turns the camera on for ~15 s; tell the user
  first.
- `scripts/diagnose_quality.py --session REC --every 9 --out OUT` explains low image quality rather than comparing runs.
  It splits the render-vs-camera error into the avatar's floor, the live pose error and the rest (unseen views, blur,
  colour), by view angle and body region. It takes ~15 min on a hot laptop.
  - On session_01 (findings §9) the split is 6% floor, 24% pose (40% facing the camera) and 70% the rest.
  - Speed changes that alter the pose (crop, detector, async lag) move the pose part. Render changes (fp16, refiner) move
    the rest.

## Where the time goes (laptop, worker off)

| Stage | Serial baseline (YOLOX) | Notes |
|---|---|---|
| Camera | 10 FPS at 1280×720 in YUYV | MJPG (now the default) gives 30 FPS |
| YOLOX-L detector | ~28 ms | every frame, fp32, 640 input; gone with `--crop center` |
| SPRINT stage | 18–26 ms | the network alone is 11–14 ms; the rest is the CPU crop, copies and the `.cpu()` sync (not yet profiled) |
| GUAVA-FB render | ~30 ms | upper-body GUAVA for comparison: refiner 17.9, deformation 6.4, rasterise 5.8 ms |
| Loop | 10.3 FPS serial | 22.3 FPS with `--crop center --parallel` on a replay from a cool start |

27–30 FPS means at most 33–37 ms per frame of throughput. Async hides CPU work and synchronisation stalls, not GPU time:
if SPRINT's GPU time plus the render's GPU time per frame exceed about 33 ms, no amount of pipelining reaches 30 FPS
(in parallel on session_01, 75 ms of stage time fit in 43 ms of wall time). Measure GPU ms per stage with CUDA events
first; the remaining budget has to come from changes that keep the output identical. Details are in
`references/findings.md`. On session_01, `--crop center --parallel` passed the quality check: the avatar scored slightly
better, and body and face pose 6–7% better. `--crop center` alone failed on hands by 0.2 points, which is within frame
noise (findings §2). The laptop throttles within a minute of load, so cooling limits sustained speed as much as the code.

## What is known about each lever

- **Camera (MJPG).** `Capture.__init__` sets `CAP_PROP_FOURCC` to MJPG before width and height (some drivers ignore it
  afterwards); `--fourcc none` restores the camera's default. Read the format back: the script prints
  `webcam: MJPG 1280x720 @ 30 fps`. Webcams quietly fall back to a lower rate in dim light (auto exposure,
  `exposure_dynamic_framerate`), so check lighting before blaming the code. Decode in the capture thread and keep only the
  latest frame, as `Capture` does.
- **Crop.** YOLOX costs ~28 ms per frame. SPRINT needs a person crop, but not a fresh YOLOX box every frame.
  - **Fixed centre square** (`--crop center`, side = height / 1.1, through the same crop code as a detector box): pose as
    accurate as the YOLOX crop on session_01. It only holds while the person stays inside the square. A fixed square can
    be larger than a tight detector box, so SPRINT's input and the 512 render spend fewer pixels on the body (3% smaller
    on session_01, more for a person standing further back); measure it against YOLOX on each new recording.
  - **Zero-cost alternative if it loses quality:** derive each frame's box from the previous frame's SPRINT 2D joints
    plus a margin, with the centre crop only at session start or when tracking is lost. Propose it; don't switch alone.
  - **Reused box:** reusing the YOLOX box for up to ~10 frames at 30 FPS kept the body inside the crop in 461 of 462
    cases. Running YOLOX in a background thread keeps it off the critical path.
  - **fp16 YOLOX** with weights cast once: 15.9 instead of 27.6 ms, box IoU 0.998.
  - **Rejected:** feeding SPRINT the whole frame. The person shrinks to 126 px in its 256 input (body error +65%, face
    +70%), and the avatar renders at half size.
- **Async** (`--parallel`, default). A pose thread runs the crop and SPRINT for frame t+1 on the high-priority LIVE
  stream while the main thread renders frame t on the default stream.
  - Keep the render on the default stream, because GUAVA's rasterizer launches on the legacy default stream.
  - PyTorch side streams don't wait for the legacy default stream on their own: hand tensors across with events, and
    outputs consumed on another stream need an event wait plus `tensor.record_stream(...)` or a held reference. The
    CUDA illegal-memory-access crashes in RUNLOG 104/105 came from cross-stream buffer reuse.
  - Hand off through one slot (the script's one-item queue with a "taken" handshake), never an unbounded queue, so
    latency doesn't creep up. The cost is about one frame of pose lag.
  - Parallel is not bit-identical to serial (renders 36–39 dB apart); it passed the gate.
- **fp16.**
  - Cast weights once, with `.half()` on a copy of the module. `torch.autocast` re-casts every weight on every call and
    was slower than fp32 (43 vs 30 ms on PEAR's ViT).
  - Small batch-1 models can get slower in fp16 (ViTPose-S: 9.90 vs 9.51 ms), and the rasterizer is fp32-only.
  - The server's fp16 vs fp32 results don't carry over to this GPU, so time each component here and run it through the
    quality check.
- **CUDA graphs and torch.compile.**
  - Capture needs static shapes and no tensors built from Python values inside the captured code (`torch.tensor(...)`
    constants break it).
  - A failed capture can leave the CUDA context broken, so restart the process.
  - On torch 2.2, `torch.compile(mode="reduce-overhead")` works; `max-autotune` fails.
- **Refiner.** It is the biggest render cost but carries quality. Removing it would fail the quality check unless
  something replaces it.

Speed-ups that keep the output identical (try these first): preallocated and pinned host buffers, `non_blocking` copies,
one device-to-host copy per frame; no hidden synchronisations in the hot path (`.item()`, `.cpu()`, printing tensors);
warm-up before timing; CUDA graphs for SPRINT (static shapes), verified identical; display and recording in their own
threads, with `cv2.setNumThreads` sized so capture decode and the main loop don't fight over cores.

Speed-ups that change the output, allowed only through the gate: lower render resolution, fewer or pruned Gaussians,
skipping or shrinking the refiner; fp16 / bf16, TensorRT, ONNX or `torch.compile` of SPRINT or GUAVA-FB; a smaller SPRINT
input, another checkpoint or SPRINT path (the live path is SPRINT scratch EMA 128k at k = 0; k = 60 refinement costs
~115 ms); rendering every other frame, interpolating, or reusing an old render or pose; a webcam resolution below
1280×720.

## Improving quality from the stream, without paying FPS

The live loop renders whatever avatar is published (`worker.published_av`, one pointer read per frame). Render time
depends only on the Gaussian count and the fixed refiner, so **an update costs the live loop nothing as long as the
worker produces it off the live thread and the Gaussian count stays bounded.** What does cost FPS is the worker competing
for the same GPU and CPU (a server test cut the loop to ~14 FPS, the laptop to 4.1). Rules:
- Worker GPU work goes on its low-priority stream in short kernels; a long kernel can't be pre-empted by the LIVE stream.
  Chunk long jobs (the 150-step fit, the encoder) across frames and cap the update cadence.
- Keep the CPU cool: DWPose on the CPU takes this laptop to 100 °C, and the GPU then throttles to 210–780 MHz, which is
  what really halves FPS. Fewer threads, or a GPU keypoint source, before anything else.
- Prove the budget with the gate: live FPS with the worker on vs off, and quality on frames the update never used.
  Under RAGA the worker stays off unless the FPS check still passes with it on.

Known quality facts that shape the update (findings §9):
- **The reference is often over-exposed.** The webcam's auto exposure was still converging during the countdown on
  session_01: the reference is 18–20% brighter and 25–33% less saturated than the first live frame. The whole avatar,
  front included, inherits that whitish look, and live frames fused into it mismatch. Use `--lock-exposure` and
  `--ref-best-of`; estimate a global colour transform between the source and the stream before fusing.
- **Texture averaging blurs.** Fusion with a ~6 px pose error smears texels; prefer selecting the sharpest, best-aligned
  frame per texel over running means, and align before fusing.
- **Texture can't add geometry.** Hair and loose cloth outside the template need new Gaussians (silhouette-driven,
  bounded count); the UV features already move the template's Gaussian offsets a little.
- **Gradient steps on the avatar in the worker are test-time fitting.** The user dropped that from the benchmark under
  fairness rule 3; ask before proposing it for the registered method.

## Working rules

- Ask the user before installing packages, committing or pushing. Never rebase or force-push. The user pushes from
  their own terminal; this shell has no GitHub credentials.
- Run everything with `/home/virlab/miniconda3/envs/guava/bin/python`.
- Keep experiment scripts and their outputs in the session scratchpad, not in the repo.
- Server runs go into `phase3_dnt/RUNLOG.md` with commit SHAs; ask before any server GPU job longer than 15 min and check
  `nvidia-smi` first (the GPUs are shared).
- Keep reports brief, with plain names (no internal labels). Research and non-commercial use only; credit GUAVA.
