# Live full-body avatar on the laptop (SPRINT + GUAVA-FB)

Webcam or video → person box (YOLOX-L on the GPU, H1c subject linking) → SPRINT (k = 0, shape locked to your session) → GUAVA-FB render → display.

When a session starts, the app captures one reference frame of you. SPRINT fits it (about 1 s), and GUAVA-FB builds your avatar from it once. Every later frame only drives that avatar.

Research and non-commercial use only. Credits: GUAVA (Zhang et al.) and the SMPL-X / FLAME licences.

## 1. Requirements

Use the laptop that already runs `main/live_pear_guava.py`, with the `guava` conda env from `LIVE_PEAR_SETUP.md`:
- PyTorch 2.2 (CUDA 12.1)
- pytorch3d 0.7.7
- `diff_gaussian_rasterization_32`, built from `./submodules/`

GPU memory: the live loop with the live appearance update ran under an 8 GB cap (peak 7.6 GB; server test 2026-11-05), so a GPU
with about 10 GB or more is enough. At session start cuDNN's autotuning may briefly borrow much more if it is free; that is not
needed. `timing_*.json` reports both: `gpu_mem_session_start_peak_gb` and `gpu_mem_peak_gb` (live loop and worker).

## 2. Install (one time)

```bash
# on the laptop
scp USER@SERVER:/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/live_package/guava_live_YYYYMMDD.tar ~/
tar -xf ~/guava_live_YYYYMMDD.tar -C ~/          # creates ~/guava_live
cd ~/guava_live
conda activate guava
pip install -r requirements_live.txt
export TORCH_HOME=~/guava_live/live_assets/torch  # DINOv2 weights shipped in the package (put this in ~/.bashrc)
python -c "import torch, pytorch3d, diff_gaussian_rasterization_32, onnx2torch; print(torch.cuda.get_device_name(0))"
```

`MANIFEST.json` lists the SHA-256 checksum of every file, so you can verify the copy.

## 3. Framing (important: full body)

- **Distance:** stand about 2–3 m from the camera. Your **whole body**, head to feet, must be in view, with a little margin above your head and below your feet.
- **Lighting:** light the front of your body evenly. Avoid a bright window behind you. A plain background helps the detector and the matte.
- **Reference countdown:** stand facing the camera with your arms slightly away from your body (a relaxed A-pose). Keep still until the countdown ends. The avatar's appearance comes from this one frame.
- **One person** should be in view at the start. After that, the app tracks the same person (H1c linking).

## 4. Run

```bash
cd ~/guava_live && conda activate guava && export TORCH_HOME=~/guava_live/live_assets/torch

# webcam (index 0), side-by-side view; press q to quit
python phase3_dnt/live/live_sprint_guavafb.py --input 0

# with your webcam calibration (optional; from tools/calibrate_webcam.py)
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --calib webcam_intrinsics.json

# avatar drawn over the camera image instead of side by side
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --view overlay

# a video file instead of the webcam (every frame is processed)
python phase3_dnt/live/live_sprint_guavafb.py --input my_clip.mp4 --ref-frame 0
```

The on-screen line shows FPS (smoothed), the per-stage milliseconds (detector, SPRINT, render, total for the frame) and the worker state.

Useful options:
- `--width/--height`: webcam resolution (default 1280×720).
- `--ref-delay`: countdown length in seconds (default 5).
- `--frames N`: stop after N frames.
- `--no-display`: run without a window.

## 5. Background worker

| Flag | Meaning | Status |
|---|---|---|
| `--worker off` | live loop only (default) | deployed pipeline |
| `--worker pose` | step 1: refines one frame at a time in the background and applies the resulting pose correction to every live frame | failed its validation (PREREG phase 6); timing only |
| `--worker uv` | live appearance update (step 2): new camera frames fill in the parts of you the reference frame did not show; the live loop renders the latest avatar version | experimental, validation pending; see section 8 |
| `--worker pose+uv`, `pose+uv+geo` | combined / geometry updates | not available |

The live loop never waits for the worker.

## 6. Timing (please run this before Nov 8)

```bash
bash phase3_dnt/live/run_timing.sh my_clip.mp4      # worker off, then pose; 600 frames each, no display
```

This writes `timing_<date>/timing_off.json`, `timing_pose.json` and `summary.json`. Each file holds:
- per-stage p50 / p95 ms (`det`, `sprint`, `render`, `total`);
- `frame_interval`;
- `capture_to_display`;
- the observed FPS;
- the number of worker updates;
- the GPU name and peak memory.

For the live webcam latency, also run the webcam for about 1 minute with a timing file:

```bash
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --frames 1500 --timing-json webcam_timing_off.json
```

Please send me the JSON files.

## 7. Recording a webcam session (optional, consented "webcam set")

```bash
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --record ~/webcam_session_01
```

This saves:
- `reference.png` and `session.json` (source fit, settings);
- `frames/NNNNNN.jpg` (camera frames);
- `avatar/NNNNNN.jpg` (live renders);
- `frames.jsonl` (per-frame timestamps, box, SPRINT output, correction version, stage ms).

A background thread does the writing, and `recorder_dropped` in the timing JSON counts any frames it had to skip.

The recording is used only for offline replay and scoring as a small held-out webcam set. It is never used for tuning.

## 8. Live appearance update (experimental) and its ablation

While you move, a background worker takes some of the camera frames, computes their image features, maps them into GUAVA-FB's
UV map, fuses them with what it already has, and rebuilds the avatar. The live loop always renders the newest version, and the
on-screen `worker uv vN` counts the updates. Every 3rd frame is never used for this (`--score-every 3`), so it can be scored fairly.

Status: experimental. Its settings are still being tuned on the server; until then it runs with `--uv-prior 2 --uv-thr 0.3`.
It shares the GPU with the live loop, so the live loop is slower while it works.

**Update this package once (from your git clone of the repo):**
```bash
cd ~/guava                                   # your clone; adjust the path
git checkout driven_not_tracked && git pull
bash phase3_dnt/live/update_package.sh /data/GUAVA/guava_live
```
This copies the new code into the package and downloads the ArcFace model (261 MB, insightface antelopev2, research use only),
which checks that face updates keep your identity. The first run also downloads the LPIPS network weights (about 530 MB).

**Try it live:**
```bash
cd /data/GUAVA/guava_live && conda activate guava && export TORCH_HOME=/data/GUAVA/guava_live/live_assets/torch
python phase3_dnt/live/live_sprint_guavafb.py --input 0 --worker uv
```

**Ablation (without vs with the update, on the same recording):**
1. Record a session (section 7), for example `--record /data/GUAVA/rec/session_01`. Countdown: face the camera, whole body in
   view. Then turn slowly all the way round twice, take a few steps, raise and lower your arms, face the camera again, press q.
2. Run:
   ```bash
   bash phase3_dnt/live/run_ablation.sh /data/GUAVA/rec/session_01
   ```
   It replays the recording twice (update off, then on; the same frames at their recorded times) and scores only the frames
   the update never used. GT = your camera frame x its person matte, as in the benchmarks.
3. Results in `/data/GUAVA/rec/session_01/ablation/`: `ablation.md` (PSNR / SSIM / LPIPS without and with the update for all
   scored frames, front-facing frames and turned-away frames, plus live ms and the number of updates), `side_by_side.mp4`
   (camera | without | with), `per_frame.csv`.
