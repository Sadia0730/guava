"""Illustration only (fixed rule, no selection): for each People-Snapshot / X-Humans clip, the test frame with the smallest and
the largest view change from the source; columns = source image, GT, GUAVA-FB [GT], GART, GUAVA-FB [SPRINT], PSNR on each."""
import json, sys
from pathlib import Path
import cv2, numpy as np

O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
rows = json.load(open(sys.argv[1])); out = sys.argv[2]
COLS = [("gt", "GT"), ("fb_gt", "GUAVA-FB [GT pose]"), ("gart", "GART (per-person)"), ("fb_sprint", "GUAVA-FB [SPRINT]")]
BENCH = {"peoplesnapshot": "publicbench_v1", "xhumans": "publicbench_v2"}
S = 256


def lab(im, t):
    im = cv2.resize(im, (S, S)); cv2.rectangle(im, (0, 0), (S, 22), (0, 0, 0), -1)
    cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); return im


lines = []
for ds, bench in BENCH.items():
    for clip in sorted({r["clip"] for r in rows if r["ds"] == ds}):
        R = sorted([r for r in rows if r["ds"] == ds and r["clip"] == clip], key=lambda r: r["view"])
        base = O / bench / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text())
        src = cv2.imread(str(next((base / "images").glob(f"{fj['source']}.*"))))
        for r in (R[0], R[-1]):
            ims = [lab(src, f"source ({clip[:14]})")]
            for m, t in COLS:
                p = O / bench / ("gt512" if m == "gt" else f"renders/{m}") / ds / clip / f"{r['key']}.png"
                ims.append(lab(cv2.imread(str(p)), t if m == "gt" else f"{t} {r[m]:.1f}"))
            ims[1] = lab(cv2.resize(ims[1], (S, S)), f"GT, view change {r['view']:.0f} deg")
            lines.append(np.hstack(ims))
cv2.imwrite(out, np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(out)
