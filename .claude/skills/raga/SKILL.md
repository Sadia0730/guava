---
name: raga
description: RAGA, the real-time full-body avatar (webcam -> center crop -> SPRINT k = 0 -> GUAVA-FB render), held to a sustained 30 fps with no loss of render quality. Use this skill for any work on the live app or laptop package (phase3_dnt/live/live_sprint_guavafb.py, guava_live, live_assets), webcam capture, MJPG, the center-crop box, async / pipelined SPRINT and rendering, CUDA streams, FPS, latency, frame drops or timing JSON, and whenever someone asks to make the live avatar faster, smoother or "real-time", even if they don't say RAGA.
---

# RAGA: live avatar at 30 fps, no quality loss

RAGA is the live path of this project:

MJPG webcam -> fixed center crop (no person detector) -> SPRINT (k = 0, body shape locked to the session) -> GUAVA-FB render
(512 px, ~207k Gaussians + StyleUNet refiner, avatar built once from the reference frame) -> display.
SPRINT and rendering run asynchronously, as a pipeline.

The user has made three design decisions: MJPG capture, a center crop instead of YOLOX, and async SPRINT and rendering. Treat
them as fixed. Don't reopen them unless a measurement shows that one of them costs quality, and then report that to the user;
never switch a choice back silently.

## The two rules, both at once

1. **30 fps, sustained.** Measure on the laptop GPU with the background worker off. Pass when:
   - the observed fps is at least 29.5;
   - the display-to-display frame interval is at most 34 ms at the median and 40 ms at p95;
   - at least 98 % of camera frames are processed.

   Report the capture-to-display latency as well. An async pipeline adds about one frame of latency, which is acceptable;
   latency above about 100 ms is felt, so flag it.
2. **No quality loss.** A speed change must not make the avatar look worse than the reference path does on the same input
   (the reference path is the synchronous loop with the same box source). How to check depends on the change; see
   "Checking a change".

A change that buys speed by giving up quality is not acceptable under RAGA, and neither is one that buys quality by
dropping below 30 fps. Report that change instead of adopting it. The point of RAGA is that neither rule is relaxed to meet
the other.

## Design choices and their pitfalls

**MJPG webcam.** Over USB 2, uncompressed YUYV at 1280x720 usually runs at only about 10 fps, while MJPG reaches 30.
- Open the camera with `cv2.CAP_V4L2` on Linux. Set `CAP_PROP_FOURCC` to `MJPG` before width and height (some drivers
  ignore it afterwards), then set `CAP_PROP_FPS` 30 and `CAP_PROP_BUFFERSIZE` 1.
- Read the FOURCC back and measure the real capture rate in the capture thread. Webcams quietly fall back to a lower rate in
  dim light (auto-exposure), so check lighting before blaming the code.
- Decode in the capture thread and keep only the latest frame, as `Capture` already does.

**Center crop instead of YOLOX.** This saves the whole detector stage (YOLOX-L plus H1c subject linking).
- The box is a centered square of side about frame height / 1.1 (`SAF.box_crop_matrix` adds a 1.1 margin), passed through
  the same crop code as a detector box.
- It is the one choice that can cost quality. The person must stay inside the square. A fixed square is usually larger than
  a tight detector box, so SPRINT's 1024 crop and the 512 render spend fewer pixels on the body, which can mean less detail
  and worse pose.
- Measure it against YOLOX on the same recording (check C below).
- If it loses quality, there is a zero-cost alternative: derive each frame's box from the previous frame's SPRINT 2D
  joints (plus margin), and use the center crop only at session start or when tracking is lost. Propose this to the user
  rather than switching on your own.

**Async SPRINT and rendering.** Pipelining makes throughput equal to the slowest stage instead of the sum of the stages.
- Keep each frame's pose with that frame: render(t) uses SPRINT(t) and is shown next to camera frame t. Pairing a new camera
  frame with an older pose is a quality loss (misalignment), even though it looks faster.
- Hand off through one slot where the newest result wins, never an unbounded queue, so latency doesn't creep up.
- Watch the stream handling:
  - The Gaussian rasterizer launches on the legacy default stream, so rendering stays on the default stream.
  - SPRINT outputs consumed on another stream need an event wait plus `tensor.record_stream(...)`, or a held reference
    until the consumer is done.
  - The CUDA illegal-memory-access crashes in RUNLOG 104/105 came from exactly this kind of cross-stream buffer reuse.
- Both stages share one GPU. Async hides CPU work and synchronization stalls, not GPU time: if SPRINT's GPU time plus the
  render's GPU time per frame exceeds about 33 ms, no amount of pipelining reaches 30 fps. Measure GPU ms per stage with
  CUDA events first. The remaining budget has to come from changes that keep outputs identical.

## Speedups that change the output (not allowed unless check B shows identical renders)

- Lower render resolution, fewer or pruned Gaussians, skipping or shrinking the refiner.
- fp16 / bf16, TensorRT, ONNX conversion or `torch.compile` of SPRINT or GUAVA-FB.
- A smaller SPRINT input, a different checkpoint, or a different SPRINT path. The live path is SPRINT scratch EMA 128k at
  k = 0; k = 60 refinement is far outside the budget at about 115 ms.
- Rendering every other frame, interpolating, or reusing an old render or pose for a new frame.
- Lowering the webcam resolution below 1280x720 without running check C.
- The live appearance update (`--worker uv`). It shares the GPU, and in a server test it cut the loop to roughly 14 fps
  (73 ms p50). Under RAGA it stays off unless the 30 fps check still passes with it on.

## Speedups that keep the output identical (try these first)

- Preallocated and pinned host buffers, `non_blocking` copies, one device-to-host copy per frame.
- No hidden synchronizations in the hot path (`.item()`, `.cpu()`, printing tensors).
- Warm-up before timing.
- CUDA graphs for SPRINT (static shapes), verified identical.
- Display (`imshow` / `waitKey`) and recording in their own threads; `cv2.setNumThreads` sized so capture decode and the
  main loop don't fight over cores.

## Checking a change (every time)

**First, find out what the code does.** The repo app as of commit d82f3e21 is still the synchronous loop with YOLOX + H1c.
The MJPG, center-crop and async version may exist only on the user's laptop (`/data/GUAVA/guava_live`). Look for it before
editing. If it isn't in git, ask the user to push it, or add each choice as an opt-in flag. Keep the existing path working,
because it is the reference for the comparisons below.

**A. Speed (laptop, worker off, no recording; the recorder's JPEG writes cost CPU):**
```bash
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --frames 1500 --no-display --timing-json raga_timing.json
python .claude/skills/raga/scripts/check_raga.py speed raga_timing.json
```
Add `--record DIR` on a separate run and pass `--record DIR` to the speed check to count dropped camera frames.

**B. Changes that should not alter renders** (async, MJPG decode path, CUDA graphs, buffers):
1. Record one session.
2. Replay it with `--no-pace` (every frame processed) through the reference path twice and through the changed path once.
3. Run:
```bash
python .claude/skills/raga/scripts/check_raga.py same REF_RUN_1 RAGA_RUN --floor REF_RUN_2
```
It passes when the RAGA renders are identical to the reference, or when their median PSNR to the reference is within
0.5 dB of the noise floor's.

The floor run is needed because the reference path is not bit-deterministic: two replays of the same recording on the server
gave renders 39–44 dB apart (Nov 5). This check is a quick screen; check C is the real quality judgement.

**C. Choices that change the input** (center crop vs YOLOX; resolution): replay the same recording through both and score
against the ground truth (camera frame x person matte) with the project scorer:
```bash
python phase3_dnt/live/score_ablation.py --session REC --runs yolox=OUT/yolox crop=OUT/crop --out OUT --video
```
No loss means that on all scored frames, and on the turned-away subset, PSNR is no more than 0.05 dB lower and LPIPS no
more than 0.001 higher. These are default margins; the user can change them.

## Reporting

- Keep it brief, with plain names (no internal labels).
- Give fps, frame interval p50 / p95, latency, the share of dropped frames and the quality result (identical, or PSNR /
  LPIPS deltas).
- Say plainly when a check fails.
- Laptop and webcam numbers are trials, not registered results.
- Server runs go into `phase3_dnt/RUNLOG.md` with commit SHAs; ask before any server GPU job longer than 15 min and check
  `nvidia-smi` first (the GPUs are shared).
- Research and non-commercial use only; credit GUAVA.
