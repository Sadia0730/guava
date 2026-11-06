# RAGA: measured findings

Laptop: Razer x Lambda TensorBook (late 2021), RTX 3080 Laptop GPU 16 GB, `guava` conda env, torch 2.2.
Measured 2026-11-05 unless noted. A change can move any of these numbers, so re-measure before building on one.

Contents: 1 Camera · 2 Live loop · 3 Detector and SPRINT crop · 4 Precision, graphs, compile · 5 GUAVA render ·
6 The old upper-body pipeline · 7 Power and thermal · 8 Recordings · 9 Image quality: where the error comes from

## 1. Camera

`v4l2-ctl -d /dev/video0 --list-formats-ext` (Integrated RGB Camera), frames per second:

| Size | MJPG | YUYV |
|---|---|---|
| 1920×1080 | 30 | 5 |
| 1280×720 | 30 | 10 |
| 960×540 | 30 | 15 |
| 848×480 | 30 | 20 |
| 640×480 and smaller | 30 | 30 |

- `live_sprint_guavafb.py` asks for 1280×720 without a pixel format, so OpenCV uses YUYV: 10 FPS.
  rec/session_01 confirms it: the camera's frame counter advances once per 100 ms (9.89 FPS).
- Fix (not applied yet): `self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))` before the width / height
  `set` calls in `Capture.__init__`. Check with `scripts/camera_fps.py`.
- Camera controls as found: `exposure_dynamic_framerate=1` (may lower the frame rate in dim light; `=0` locks it),
  `auto_exposure=3` (aperture priority), `white_balance_automatic=1`, `power_line_frequency=1` (50 Hz; US mains is 60 Hz,
  `=2`, else lights can flicker in the image).
- **Auto exposure changed the scene after the reference.** On session_01 the static background is 4–9% darker in the
  live frames than in reference.png. The avatar keeps the reference's colours, so it reads 5–10% too bright on later
  frames (§9).

## 2. Live loop (worker off)

| Stage | Live, on screen | Notes |
|---|---|---|
| YOLOX-L detector | ~28–30 ms | every frame, 640 input, fp32 |
| SPRINT stage | ~26 ms | the network alone is 11–14 ms (measured in isolation); the rest is probably the CPU crop to 1024, copies and the final `.cpu()` sync (not yet profiled) |
| GUAVA-FB render | ~30 ms | 512 px |
| Result | ~8.4 FPS | serial loop, capped at 10 by the camera |

How `--timing-json` measures (`live_sprint_guavafb.py` main loop):
- `det` is the wall time of `detect()`.
- `sprint` is CUDA events on the LIVE stream around `live_raw`, so CPU gaps inside it count.
- `render` runs from the event on LIVE to the event on the default stream.
- `total` = detect + SPRINT + render + `.cpu()` copies, without display.
- `frame_interval` and `fps_observed` include display.
- `capture_to_display` is the latency.

Speed-up switches in `live_sprint_guavafb.py` (2026-11-05): `--fourcc MJPG` (now the default), `--crop center`, `--parallel`.
Replay with `--no-pace`, plugged in, each run started at 73–74 °C, GPU sampled every 2 s:

| Run | FPS | Frame time p50 / p95 | Detector | SPRINT | Render | Busy samples under thermal slowdown |
|---|---|---|---|---|---|---|
| Baseline | 10.3 | 87 / 116 ms | 30.3 | 22.0 | 31.2 | 27% |
| `--crop center --parallel` | 22.3 | 43 / 53 ms | 0.1 | 38.8 | 35.9 | 14% |
| Baseline again, laptop heat-soaked | 6.2 | 114 / 293 ms | 42.5 | 24.0 | 43.3 | 33% |

- In parallel, SPRINT and the render slow each other on the shared GPU (75 ms of stage time in 43 ms of wall time).
- Every run throttled within a minute. The CPU reached 93 °C after three runs.

Quality of the switches: `scripts/score_replays.py` on the battery-run replays of session_01 (333 frames), YOLOX as the
baseline, plus a repeat run for noise. Quality doesn't depend on the power source: the same run on battery and plugged in
rendered 55–61 dB apart. Avatar metrics are compared with the baseline scored in the same crops.

| Run | PSNR | SSIM | LPIPS | Body px | Face px | Hands px | Gate |
|---|---|---|---|---|---|---|---|
| Baseline (YOLOX), scored in the candidates' crops | 23.38 | 0.8552 | 0.1193 | 19.2 | 9.8 | 21.5 | — |
| `--crop center` | 23.65 (+0.27) | 0.8581 | 0.1172 | 18.0 (−6.1%) | 9.2 (−6.2%) | 22.0 (+2.2%) | fail: hands over 2% |
| `--crop center --parallel` | 23.66 (+0.28) | 0.8579 | 0.1173 | 17.8 (−7.3%) | 9.2 (−6.1%) | 21.9 (+1.7%) | pass |
| Baseline repeat (noise) | +0.02 | +0.0004 | −0.0004 | −0.3% | −0.5% | −0.3% | — |

- **Image and body / face pose improve.** 95% block-bootstrap intervals (10-frame blocks) exclude zero.
- **Hands are within frame noise.**
  - The mean rose 0.48 px, with a 95% interval of −0.48 to +1.59 px. The median per-frame change is 0.0 px, and hands are
    worse on 50% of frames.
  - The rise comes from side-view frames 198–203 and 450–455, where both crops already miss the hands by 60–110 px. The
    hands stay inside the square there.
  - The centre crop is 15–35 px better on frames 445 and 495.
  - The 2% rule fails `--crop center` by 0.2 points. Its noise widening covers run-to-run noise only, not which frames the
    session holds; more frames (a second session) would settle it.
- **Matched vs own-crop scores.** The baseline scores 23.06 dB in its own crop but 23.38 in the candidates' crops, because
  a larger crop holds more black background.
- **Crop size.** On session_01 the YOLOX crop is 696 px (median; 621–763) against 720 px for the centre square. With
  `--crop center`, the person is 3% smaller (up to 14%) in SPRINT's input and in the 512 render. A person standing
  further back would shrink more; YOLOX would zoom in.
- **The check is slow when the laptop is hot.** DWPose runs on the CPU, here with 6 threads. The check took 85 min: the
  CPU went from about 81 to 100 °C within one frame, 223 times, and the heat guard paused for 80 min in total.
- On battery (an earlier set of runs), everything was about 2–3× slower; ratios between runs were similar
  (centre + parallel 1.7× baseline).

Other runs:
- rec/session_01/ablation/off (replay): det 121 / SPRINT 37 / render 118 ms p50, 3.5 FPS. That is about 4× the live
  numbers; conditions weren't logged, so the cause (heat or battery) can't be told.
- `--worker uv` live: 4.1 FPS, GPU memory peak 14.26 GB (rec/session_01_timing_uv.json).

## 3. Detector and SPRINT input crop

Session rec/session_01: 333 frames, person about 633 px tall in a 1280×720 frame.

YOLOX-L on 34 frames, GPU at 69 °C. IoU is the box overlap with the live fp32 640 box (1.0 = identical):

| Config | Network | Box IoU mean / worst |
|---|---|---|
| fp32, 640 (live) | 27.56 ms | — |
| fp16, weights cast once, 640 | 15.86 ms | 0.998 / 0.991 |
| fp32, 416 | 12.01 ms | 0.974 / 0.931 |
| fp32, 320 | 9.96 ms | 0.972 / 0.939 |

- CPU pre-processing takes 0.78 ms; post-processing + NMS take 1.14 ms.
- EHM's `demo_postprocess` modifies its input in place, so copy the array before reusing it.

SPRINT input crop (`scripts/sprint_crop_test.py`). Error is SPRINT's projected keypoints vs DWPose, in frame px:

| SPRINT input | Person px in the 256 input | Body | Face | Hands | Joint rotation vs YOLOX crop |
|---|---|---|---|---|---|
| YOLOX crop (live rule: 1.1 × box, square) | 233 | 19.0 | 9.8 | 21.5 | — |
| Whole frame, padded to a square | 126 | 31.4 | 16.7 | 27.8 | 7.5° body, 8.9° hands |
| Fixed 720×720 centre square (full height) | 224 | 17.9 | 9.4 | 22.0 | 2.3°, 2.9° |

- **Whole frame: rejected.** Besides the worse pose, GUAVA draws the avatar in SPRINT's crop camera, so it comes out about
  half size (~250 px tall in the 512 window instead of ~465).
- **Centre square:** pose as accurate as the YOLOX crop on this session.
  - Only valid while the person stays inside x = 280–1000 px; the session's box centre stayed at 590–700.
  - Not yet checked on avatar quality or live.

Reusing an old YOLOX box (`scripts/box_staleness.py`, from the recorded boxes). "Crop IoU" is the overlap of the reused
crop with a fresh one:

| Box age | Crop IoU vs fresh, median | Whole body inside the reused crop |
|---|---|---|
| up to 0.36 s (~10 frames at 30 FPS) | 0.97–0.99 | 461 of 462 cases (worst 98%) |
| ~0.5 s | 0.96 | 98% of cases (worst 95%) |
| ~1 s | 0.94 | 96% of cases (worst 89.5%) |
| box from the reference frame, never updated | 0.82 | 29% of frames (worst 85.8%) |

## 4. Precision, CUDA graphs, compile

- **Cast the weights to fp16 once.** `torch.autocast` re-casts every weight on every call. In the old upper-body runner,
  PEAR's ViT took 43 ms under autocast vs 30 ms in fp32.
- **fp16 isn't always faster.** PEAR student backbone: fp32 11.60 → fp16 7.01 ms. ViTPose-S: fp32 9.51 → fp16 9.90 ms
  (slower). YOLOX-L with weights cast once: 1.74× faster. Sources: phase1/task2/backbone_latency.md, vitpose_speed.md.
- **Exact speed-ups exist.** The fastest exact ViTPose-S was fp32 + single-expert MoE + CUDA graph: 5.26 ms, bit-identical.
- **GUAVA's rasterizer is fp32-only.**
- **The server's fp16 vs fp32 results don't transfer** to this laptop. Its `phase3_dnt/sprint_fast_driving.py` uses
  autocast with `cache_enabled=False` (re-casts every call) and onnxruntime fp16 conversion.
- **CUDA graphs:**
  - A head that builds tensors from Python values (`torch.tensor(...)` constants) breaks capture.
  - A failed capture poisoned the CUDA context; restart the process after one.
- **torch.compile on torch 2.2:** `mode="reduce-overhead"` works; `max-autotune` fails with an inductor `freeze_layout`
  error.

## 5. GUAVA render components

Upper-body GUAVA, 512 px, fp32; not yet measured for GUAVA-FB:

| Component | Time |
|---|---|
| Deformation | 6.35 ms |
| Rasterisation | 5.82 ms |
| StyleUNet refiner | 17.92 ms |
| Other | 0.07 ms |
| Total per frame | 30.15 ms |

- Avatar generation takes ~199 ms.
- The refiner is the biggest piece but carries quality: a static per-Gaussian colour fix recovered only 45–80% of its PSNR
  gain. Dropping it would fail the quality gate unless something replaces it.

## 6. The old upper-body pipeline (main/live_pear_guava.py): the 27–30 FPS reference

- **Camera:** OpenCV's default 640×480, which is 30 FPS in YUYV.
- **No detector:** the whole frame, or `--input_framing centered`.
- **Models:** PEAR student fp16 (~7 ms) + GUAVA upper body (~27–30 ms).
- **Async:** pose on a side CUDA stream and GUAVA on the default stream, with a latest-frame mailbox. Result: 27–30 FPS.
- **Serial phase timing** (fp16, 256 px, AC, performance mode):
  - Original PEAR: 4.4 + 16.5 + 27.9 = 48.7 ms.
  - With the student: 4.2 + 6.8 + 27.0 = 38.0 ms.

## 7. Power and thermal

- On battery or in power-saver mode, the GPU is held to 54 W. On AC with `powerprofilesctl set performance`, it gets 80 W.
- At ~87 °C and above, the driver's software thermal slowdown drops the SM clock to 210–780 MHz.
- `nvidia-smi` shows `power.limit` as [N/A] on this GPU; use `power.draw` and `clocks_throttle_reasons.active`.
  The AC state is in `/sys/class/power_supply/AC0/online`.
- Before timing, check `nvidia-smi --query-compute-apps=pid,process_name --format=csv` for other GPU programs.
  - On 2026-11-05 RustDesk (remote desktop) held a GPU context (313 MiB). The idle GPU stayed in P0 with its memory
    clock at 6001 MHz: 35 W at 0–3% use, against 20 W earlier that day without it.
  - With the chassis heat-soaked, the idle GPU then sat at 92–93 °C under SW thermal slowdown (210 MHz), and the CPU
    package read 94 °C ("ALARM (CRIT)", limit 100 °C). It cooled about 1 °C per 1.5 min.
  - Timings taken in that state are meaningless: a 40-frame smoke replay rendered in 174–196 ms instead of ~30.
- The webcam was confirmed with `scripts/camera_fps.py`: default and YUYV 10.0 FPS, MJPG 30.0 FPS at 1280×720.

## 8. Recordings

- **rec/session_01:** 333 processed frames, 1280×720.
  - The camera ran at 10 FPS, and the loop kept every ~2nd frame.
  - Recorded with `--worker uv`; the person turns in place after an A-pose reference.
  - Has `reference.png` and `session.json` (body shape, source pose).
  - Replays: rec/session_01/ablation/{off,uv}.
- **Consecutive frames are 100–400 ms apart.** Anything temporal (box reuse, tracking, async lag) is harsher on it than at
  30 FPS.
- **Use `--no-pace` for throughput.** A paced replay waits for the recorded capture times, so it can't beat 10 FPS.
- **A 30 FPS recording** needs the MJPG fix and a loop that keeps up. The recorder saves processed frames only.

## 9. Image quality: where the error comes from

`scripts/diagnose_quality.py --session rec/session_01 --every 9` (37 live frames plus the reference, worker off, YOLOX
crop). Each frame is rendered twice: from SPRINT's live pose, and from a 150-step refined pose (`fitter.fit`, shape
locked). Both are scored against camera × matte. Output: summary.md, per_frame.json and quality_grid.jpg (camera |
live pose | refined pose | error).

| Frames | n | PSNR, live / refined pose | LPIPS, live / refined | Face PSNR | Hands PSNR | Body PSNR |
|---|---|---|---|---|---|---|
| Reference frame (the avatar's source) | 1 | 26.97 / 35.08 | 0.076 / 0.036 | 23.7 / 33.8 | 14.4 / 27.2 | 21.8 / 28.6 |
| Live, facing the camera (< 30°) | 14 | 23.42 / 25.63 | 0.119 / 0.099 | 23.8 / 24.7 | 14.8 / 20.2 | 20.2 / 20.6 |
| Live, turned 30–90° | 7 | 23.92 / 24.97 | 0.111 / 0.101 | 21.4 / 25.2 | 15.0 / 19.1 | 17.7 / 18.6 |
| Live, turned away (≥ 90°) | 16 | 22.51 / 23.14 | 0.136 / 0.130 | 16.1 / 18.9 | 14.6 / 17.6 | 16.7 / 16.9 |

- **Error budget** (mean MSE over live frames):
  - The avatar's floor, its own source frame with its refined pose: 6%.
  - SPRINT's pose error (live vs refined pose): 24%. That is 40% when facing the camera and 13% when turned away.
  - The rest: 70%. This is the avatar not matching new views and poses (below).
- **Whole-crop PSNR is confounded by silhouette area.** Side views hold more black background and score higher. Compare
  views with the region PSNRs.
- **Unseen back and sides.** The avatar is built from one front image.
  - It guesses the back pale grey-green and has no ponytail (quality_grid.jpg, frames 212 and 484).
  - From facing to turned away, face PSNR falls 23.8 → 16.1 and body 20.2 → 16.7. 16 of the 37 sampled frames are
    turned away.
  - `--worker uv` partly fixes this. On session_01 (rec/session_01/ablation): turned away +1.27 dB, LPIPS −0.005; front
    LPIPS +0.003. It ran at 4.1 FPS live.
- **Blur.** The render has 52% of the camera's fine detail (Laplacian energy, inside the person) on live frames, and
  67–69% on the reference. The floral print comes out as blotches.
- **Colour.** Inside the person, on live frames the render is +5% brighter and −15% less saturated than the camera. On the
  reference it is within 2% and 3%.
  - Part of this is the camera: auto exposure darkened the scene by 4–9% after the reference (§1).
  - Turned away, it is +11% / −17%: the pale guessed back.
- **Pose.** SPRINT's keypoint error is 23.4 px (1024 crop) against 5.9 px after refinement. Hands PSNR goes 14.7 → 18.9
  with the refined pose. The refinement takes seconds per frame, so it is not live.
- **Size.** The person is 442 px tall in the 512 render and the face 43 px; the person is 629 px tall in the camera frame.
- **Heat.** The run had 37 pauses, 685 s in total, with the process pinned to 4 cores (`taskset -c 0-3`) and 4 DWPose
  threads. The CPU still reached 100 °C on each frame.

### 9a. Why the render is blurry: the resolution chain

The avatar's appearance never sees the camera's full detail (`assets/GUAVA/config.yaml`, `fb_common.FBData`):

| Step | Pixels for the person |
|---|---|
| Webcam frame, 1280×720 | 633 px tall |
| Reference crop (1.1 × box, square) → 1024 | upscaled 1.46×; no new detail |
| Image encoder input (`feature_img_size`) | **518 px** square, person ~470 px tall; the texture features come from here |
| UV map (`uvmap_size`) | 512 |
| Render (`image_size`) and StyleUNet refiner | 512; person 442 px tall, face 43 px |

- The appearance is predicted feed-forward from one 518-px image, and the refiner was trained with image losses on a
  dataset, so where it is unsure it draws the average: fine print becomes blotches. There is no per-subject fitting.
- The reference frame itself renders at 69% of the camera's fine detail; live frames at 52%. The extra loss on live frames
  comes from pose error (misaligned Gaussians) and views the source never showed.
- Webcam noise and JPEG artefacts count as "detail" in the camera frame, so these ratios are a little pessimistic.
- A 1080p MJPG frame (30 FPS, §1) would make the person ~950 px tall and the reference crop ~1050 px: the 1024 crop
  stops being an upscale, but the encoder still sees 518 px. The gain would be limited to sharper input; measure it.

### 9b. The uv appearance update over time (rec/session_01/ablation, worker uv, 10 updates)

| Frames | n | Avatar versions | PSNR off → uv | LPIPS change |
|---|---|---|---|---|
| Turned away, first half of the session | 22 | 0–3 | 22.83 → 24.02 (+1.19) | −0.005 |
| Turned away, second half | 21 | 3–6 | 22.13 → 23.50 (+1.36) | −0.005 |
| Front, first half | 34 | 0–4 | 23.91 → 23.90 | +0.0006 |
| Front, second half | 34 | 4–9 | 23.10 → 23.32 (+0.22) | +0.003 |

- The gain arrives with the first update and then stays at about +1.2–1.5 dB; the second turn is not better than the
  first. Per version: v1 +0.9, v2 +2.6, v3 +2.0, v4 +0.7, v5 +1.5, v6 +1.5 dB on turned-away frames.
- The update only writes colour into the UV texture of the SMPL-X body. It cannot add geometry (the ponytail), cannot
  undo SPRINT's pose error, and inherits the 518-px encoder and the refiner's smoothing (§9a). It also projects live
  frames with a refined pose that is still off by ~6 px, so texels from several frames blur together.
- Fusion is capped: the source keeps prior weight 2, texels stop at confidence 4, regions stop by the held-out rule and
  ArcFace can roll an update back. On the front, updates cost LPIPS +0.003.
- The laptop ran this mode at 4.1 FPS, with 817 ms per update on the server.

### 9c. The reference frame was over-exposed (why even the front looks whitish)

rec/session_01: the reference is captured after a 5 s countdown; the first live frame comes ~3 s later (avatar build
2.75 s), in the same pose. Torso patch of the YOLOX box, HSV:

| Image | Mean RGB | V (brightness) | S (saturation) |
|---|---|---|---|
| reference.png (the avatar's source) | 107, 105, 92 | 107.9 | 40.1 |
| frame 88, first live frame | 88, 84, 70 | 88.4 (−18%) | 52.7 (+31%) |
| frames 89–103 | 85–87, 81–84, 68–70 | −18 to −20% | +24 to +33% |

- The background darkened too (bed −8.5%, wall −3.6% in V), so the camera's auto exposure moved after the reference;
  it was still converging during the countdown or adapting to the person walking in.
- The avatar reproduces the reference's colours exactly (diagnose_quality: brightness +0.4%, saturation −2% on the
  reference frame) and therefore renders every live frame too bright and too pale. This is most of the "whitish" look
  on the front; the pale guessed back adds to it when turned away.
- Fusing live frames into this source mixes two exposures; the uv worker's luminance-reopen rule (15%) exists for this
  but the source anchor keeps prior weight 2.
- Picture: scratchpad `analysis/ref_vs_first_frame.jpg`.
- Fixes with no live cost: wait for the exposure to settle (brightness stable over ~1 s) before capturing; lock
  `auto_exposure`, `exposure_time_absolute` and `white_balance_automatic` at that moment (v4l2-ctl) so the stream
  matches the source; and/or estimate a per-channel affine colour map from render to matte-masked frame on well-aligned
  frames in the worker and apply it to the published avatar.
