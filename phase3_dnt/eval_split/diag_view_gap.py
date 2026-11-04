"""Descriptive (no selection): per-frame PSNR of each finished row vs the view change from the single source frame.
View change = geodesic angle between the source frame's and the test frame's root orientation in the crop-camera frame
(base/<ds>/<clip>/smplx2020.npz, global_orient), so camera motion counts too. PSNR as main/metrics.psnr on the scored 512 PNGs."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot

O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
SETS = {"peoplesnapshot": "publicbench_v1", "neuman": "publicbench_v1", "xhumans": "publicbench_v2", "bedlam2": "avatar_test_v1", "ubody": "avatar_test_v1"}
METHODS = ["guava", "fb_gt", "fb_sprint", "gart", "gaussianavatar", "exavatar_gtface"]
BINS = [0, 30, 60, 90, 135, 181]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def clip_rows(ds, bench, clip):
    base = O / bench / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text())
    s = np.load(base / "smplx2020.npz"); idx = {str(k): i for i, k in enumerate(s["keys"])}
    R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]
    rows = []
    for k in fj["test"]:
        ang = float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec())))
        g = cv2.imread(str(O / bench / "gt512" / ds / clip / f"{k}.png"))
        r = {"ds": ds, "clip": clip, "key": k, "view": ang}
        for m in METHODS:
            p = O / bench / "renders" / m / ds / clip / f"{k}.png"
            if (O / bench / "renders" / m / ds / clip / "run.json").exists() and p.exists(): r[m] = psnr(cv2.imread(str(p)), g)
        rows.append(r)
    return rows


jobs = []
for ds, bench in SETS.items():
    for d in sorted((O / bench / "base" / ds).iterdir()):
        jobs.append((ds, bench, d.name))
with ThreadPoolExecutor(24) as ex:
    rows = [r for rs in ex.map(lambda j: clip_rows(*j), jobs) for r in rs]
out = Path(sys.argv[1]); out.write_text(json.dumps(rows))
print(len(rows), "frames ->", out)
