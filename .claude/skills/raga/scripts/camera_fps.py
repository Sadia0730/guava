"""Webcam frame rate per pixel format at one size: shows whether the capture mode caps RAGA before any model runs.

This TensorBook's webcam gives 1280x720 at 30 FPS only in MJPG; OpenCV's default (YUYV) gives 10 FPS at that size.
"default" below is what live_sprint_guavafb.py does today (sets width / height, no pixel format).
Turns the webcam on for about 15 s; saves nothing. Close any program using the camera first.

usage: python camera_fps.py [--device 0] [--width 1280] [--height 720] [--seconds 3]
"""
import argparse
import time

import cv2

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()

for mode in ("default", "YUYV", "MJPG"):
    cap = cv2.VideoCapture(a.device)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.device} (in use by another program?)")
    if mode != "default":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    got = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    for _ in range(15):                                   # let auto-exposure settle
        cap.read()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        n += bool(cap.read()[0])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{mode:>7}: got {got} {w}x{h}  {n / a.seconds:.1f} fps", flush=True)
    cap.release()
