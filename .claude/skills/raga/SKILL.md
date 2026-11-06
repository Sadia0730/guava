---
name: raga
description: Speed-and-quality playbook for RAGA, the user's real-time full-body avatar (webcam → person crop → SPRINT pose → GUAVA-FB Gaussian avatar), run by guava_live/phase3_dnt/live/live_sprint_guavafb.py on an RTX 3080 Laptop. Use it for any work on RAGA or the live full-body pipeline, even if the user doesn't say "RAGA" — speeding it up, profiling or timing it, changing the camera, detector, crop, fp16 or precision, CUDA streams or async, CUDA graphs or torch.compile, adding features to the live loop, or explaining its FPS. RAGA must run at 27–30 FPS end to end with no measurable loss in pose or avatar quality, and this skill shows how to prove both.
---

# RAGA: 27–30 FPS, no quality loss

RAGA is the user's live full-body avatar. A webcam frame goes through a person box, then SPRINT (body, hand and face pose)
and GUAVA-FB (a Gaussian avatar rendered at 512 px), and onto the screen. It is run by
`/data/GUAVA/guava_live/phase3_dnt/live/live_sprint_guavafb.py` on a Razer x Lambda TensorBook (RTX 3080 Laptop GPU, 16 GB).

**The research goal: improve the monocular avatar during inference, from the frames of the stream.** The avatar is
built from one reference frame, so its back and sides are guesses. While the user moves, the stream shows those parts,
and the live update (`--worker uv`, worker6 / uv_fusion.py) is meant to fold them into the avatar, and to improve what
was already seen, without pausing the live loop. Speed work (crop, async, fp16) serves this goal: it buys the time budget
the update needs. See "Improving quality from the stream" below.

The project has one constraint with two halves: **27–30 FPS end to end, and no measurable loss in output quality.**
- A speed-up that costs quality fails.
- A feature that drops the rate below 27 FPS fails. That includes the update worker: on session_01 it halved the live
  rate (8 → 4.1 FPS), so its GPU and CPU cost count against the budget.

Back both halves with numbers. Feelings are misleading on this laptop:
- Heat or battery power can halve the frame rate without any code change.
- A pose that is a few pixels off looks fine on screen but drags the avatar.

## Before calling a change done

1. **Speed.** Report `fps_observed` and the per-stage p50 / p95 from `--timing-json`.
   - Use a `--no-pace` replay for the A/B comparison, because it is repeatable.
   - Use a live webcam run for the final 27–30 FPS claim, because the camera, the capture thread and the display only
     exist live.
2. **Quality.** Run `scripts/score_replays.py` on replays of the same recording, with the current pipeline as the
   baseline.
   - It passes when no metric is worse than the baseline beyond run-to-run noise.
   - If it fails, report the failure with its numbers; don't explain it away. The user chose "no measurable loss", so a
     small but real loss is a fail, not a judgement call.
3. **Conditions.** Log the GPU temperature, SM clock and power source with every timing, so a slow run can be told apart
   from a slow change.

When reporting:
- Lead with the verdict: is 27–30 FPS met, and is quality unchanged?
- Then give baseline vs candidate numbers and the logged conditions.
- Name any check you haven't run yet; a replay-only result is not a live result.

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
- **Async or pipelined changes render a pose from an earlier frame.** Record each render under the index of the frame
  its pose came from. Otherwise the quality check charges the lag as error, and on session_01 that lag is 100–400 ms.

Final live check:
```bash
$PY phase3_dnt/live/live_sprint_guavafb.py --input 0 --worker off --timing-json OUT/live_timing.json
```
- `--print-fps 30` (the default) prints a `[fps]` line to the terminal every 30 frames: the smoothed rate, the mean since
  warm-up and the stage times. The summary JSON printed at the end has `fps_observed` (first to last displayed frame).
- Press `q` in the window to stop. Stand inside the green square when using `--crop center`.

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
- **Limits.** `--worker pose` drives the avatar with a corrected pose that the recorder doesn't save, so the pose score
  then covers SPRINT's raw output only. The score covers one session; mention that when the change could behave
  differently with other motion or framing.

Pose experiments that don't need a modified live script:
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

## Where the time goes (2026-11-05, worker off)

| Stage | Now | |
|---|---|---|
| Camera | 10 FPS at 1280×720 | YUYV by default; this webcam gives 30 FPS at 720p only in MJPG |
| YOLOX-L detector | ~28 ms | every frame, fp32, 640 input |
| SPRINT stage | ~26 ms | the network alone is 11–14 ms; the rest is probably the CPU crop, copies and the `.cpu()` sync (not yet profiled) |
| GUAVA-FB render | ~30 ms | upper-body GUAVA for comparison: refiner 17.9, deformation 6.4, rasterise 5.8 ms |
| Loop | serial | ~8.4 FPS seen, capped at 10 by the camera |

27–30 FPS means at most 33–37 ms per frame of throughput. Details and sources are in `references/findings.md`.

The live script now has `--fourcc MJPG` (default), `--crop center` and `--parallel` (default since 2026-11-06;
`--no-parallel` gives the serial loop, which is the baseline for async A/B pairs). On a replay from a cool start,
`--crop center --parallel` ran at 22.3 FPS (43 ms per frame) against 10.3 FPS for the baseline. On session_01 it passed
the quality check: the avatar scored slightly better, and body and face pose 6–7% better. `--crop center` alone failed on
hands by 0.2 points, which is within frame noise (findings §2). The laptop throttles within a minute of load, so cooling
limits sustained speed as much as the code does.

## What is known about each lever

- **Camera.**
  - Add `self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))` before the width / height `set` calls in
    `Capture.__init__`.
  - Without it, nothing else can lift RAGA above 10 FPS at 1280×720.
- **Detector.** It costs ~28 ms per frame. SPRINT needs a person crop, but not a fresh YOLOX box every frame.
  - **Fixed 720×720 full-height centre square:** pose as accurate as the YOLOX crop on session_01. It only holds while
    the person stays inside that square.
  - **Reused box:** reusing the YOLOX box for up to ~10 frames at 30 FPS kept the body inside the crop in 461 of 462
    cases. Running YOLOX in a background thread keeps it off the critical path.
  - **fp16 YOLOX** with weights cast once: 15.9 instead of 27.6 ms, box IoU 0.998.
  - **Rejected:** feeding SPRINT the whole frame. The person shrinks to 126 px in its 256 input (body error +65%, face
    +70%), and the avatar renders at half size.
- **Async.** Run SPRINT for frame t+1 while frame t renders. The old upper-body pipeline (`main/live_pear_guava.py`)
  reached 27–30 FPS this way.
  - Keep the render on the default stream, because GUAVA's rasterizer launches on the legacy default stream. Put SPRINT
    on a side stream.
  - PyTorch side streams don't wait for the legacy default stream on their own, so hand tensors across with events.
  - Both stages share one GPU, so the gain lands between serial and the slowest stage; measure it.
  - The cost is about one frame of pose lag.
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

## Improving quality from the stream, without paying FPS

The live loop renders whatever avatar is published (`worker.published_av`, one pointer read per frame). Render time
depends only on the Gaussian count and the fixed refiner, so **an update costs the live loop nothing as long as the
worker produces it off the live thread and the Gaussian count stays bounded.** What does cost FPS is the worker competing
for the same GPU and CPU. Rules:
- Worker GPU work goes on its low-priority stream in short kernels; a long kernel can't be pre-empted by the LIVE stream.
  Chunk long jobs (the 150-step fit, the encoder) across frames and cap the update cadence.
- Keep the CPU cool: DWPose on the CPU takes this laptop to 100 °C, and the GPU then throttles to 210–780 MHz, which is
  what really halves FPS. Fewer threads, or a GPU keypoint source, before anything else.
- Prove the budget with the gate: live FPS with the worker on vs off, and quality on frames the update never used.

Known quality facts that shape the update (findings §9):
- **The reference is often over-exposed.** The webcam's auto exposure was still converging during the countdown on
  session_01: the reference is 18–20% brighter and 25–33% less saturated than the first live frame. The whole avatar,
  front included, inherits that whitish look, and live frames fused into it mismatch. Settle or lock exposure before
  the capture; estimate a global colour transform between the source and the stream before fusing.
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
