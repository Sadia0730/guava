"""SPRINT deployment: webcam intrinsics with an OpenCV checkerboard, for SPRINT's camera input (focal length + principal
point; real-intrinsics flag = 1).

Usage (print a checkerboard, e.g. 9 x 6 inner corners, measure the square size):
  python tools/calibrate_webcam.py --camera 0 --cols 9 --rows 6 --square-mm 25 --out webcam_intrinsics.json
Live window: SPACE = capture a view (move the board: tilt, corners of the image, near / far), ENTER = calibrate (>= 12
views), ESC = quit. --images DIR calibrates from saved photos instead. Output JSON: K (3x3), dist (k1 k2 p1 p2 k3),
image size, RMS reprojection error (px), focal in px; SPRINT uses fx (fy) and (cx, cy) at that resolution (rescale K if
the live stream uses another resolution)."""
import argparse, glob, json
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--camera", type=int, default=0); ap.add_argument("--images", default=None)
ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6); ap.add_argument("--square-mm", type=float, default=25.0)
ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
ap.add_argument("--min-views", type=int, default=12); ap.add_argument("--out", default="webcam_intrinsics.json"); a = ap.parse_args()
pattern = (a.cols, a.rows)
obj = np.zeros((a.cols * a.rows, 3), np.float32); obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm / 1000.0
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
objs, imgs, size = [], [], None


def detect(gray):
    ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit) if ok else None


if a.images:
    for f in sorted(glob.glob(a.images + "/*")):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        size = g.shape[::-1]; c = detect(g)
        if c is not None: objs.append(obj); imgs.append(c)
        print(f, "ok" if c is not None else "no board")
else:
    cap = cv2.VideoCapture(a.camera); cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    while True:
        ok, frame = cap.read()
        if not ok: raise SystemExit("camera read failed")
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); size = g.shape[::-1]; c = detect(g); vis = frame.copy()
        if c is not None: cv2.drawChessboardCorners(vis, pattern, c, True)
        cv2.putText(vis, f"views {len(imgs)} (SPACE capture, ENTER calibrate, ESC quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate", vis); k = cv2.waitKey(1) & 0xFF
        if k == 27: raise SystemExit("aborted")
        if k == 32 and c is not None: objs.append(obj); imgs.append(c)
        if k == 13 and len(imgs) >= a.min_views: break
    cap.release(); cv2.destroyAllWindows()
if len(imgs) < a.min_views:
    raise SystemExit(f"only {len(imgs)} views with a detected board (need {a.min_views})")
rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
res = {"image_size_wh": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms), "views": len(imgs),
       "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
