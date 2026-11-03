# Live full-body avatar on the laptop (SPRINT + GUAVA-FB)

Webcam or video → person box (YOLOX-L on the GPU, H1c subject linking) → SPRINT (k = 0, shape locked to your session) → GUAVA-FB render → display.

When a session starts, the app captures one reference frame of you. SPRINT fits it (about 1 s), and GUAVA-FB builds your avatar from it once. Every later frame only drives that avatar.

Research and non-commercial use only. Credits: GUAVA (Zhang et al.) and the SMPL-X / FLAME licences.

## 1. Requirements

Use the laptop that already runs `main/live_pear_guava.py`, with the `guava` conda env from `LIVE_PEAR_SETUP.md`:
- PyTorch 2.2 (CUDA 12.1)
- pytorch3d 0.7.7
- `diff_gaussian_rasterization_32`, built from `./submodules/`

The GPU needs about 13 GB of memory (tested peak 12.9 GB with `--worker pose`; see `gpu_mem_peak_gb` in `timing_*.json`).

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
| `--worker pose` | step 1: refines one frame at a time in the background and applies the resulting pose correction to every live frame | under validation (PREREG phase 6); use it for timing only until it passes |
| `--worker pose+uv`, `pose+uv+geo` | appearance / geometry updates (steps 2 / 3) | not available until they pass validation |

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
