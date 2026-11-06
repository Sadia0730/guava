"""Side-by-side figure for the gap diagnosis: per selected frame one row = GT | renders of the methods | error heat maps (mean abs
error x 4, person mask outline). Usage: python fig_gap.py --E <root> --frames ds/clip/key[,..] --methods fb_gt,diag_fuse_oracle,gart --out fig.jpg"""
import argparse
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--frames", required=True)
ap.add_argument("--methods", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=384)
a = ap.parse_args(); E = Path(a.E); M = a.methods.split(","); S = a.size


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2; return 10 * np.log10(1.0 / max(mse, 1e-12))


def label(img, txt):
    img = img.copy(); cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA); return img


rows = []
for fr in a.frames.split(","):
    ds, clip, key = fr.split("/"); g = cv2.imread(str(E / "gt512" / ds / clip / f"{key}.png"))
    m = cv2.imread(str(E / "base" / ds / clip / "masks" / f"{key}.png"), 0); m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_AREA) > 127
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = [label(cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA), f"GT {clip} {key}")]; errs = []
    for meth in M:
        p = E / "renders" / meth / ds / clip / f"{key}.png"; r = cv2.imread(str(p)) if p.exists() else np.zeros_like(g)
        tiles.append(label(cv2.resize(r, (S, S), interpolation=cv2.INTER_AREA), f"{meth} {psnr(r, g):.2f} dB"))
        e = np.clip(np.abs(r.astype(np.float32) - g.astype(np.float32)).mean(-1) * 4, 0, 255).astype(np.uint8)
        e = cv2.applyColorMap(e, cv2.COLORMAP_INFERNO); cv2.drawContours(e, cnt, -1, (255, 255, 255), 1)
        errs.append(label(cv2.resize(e, (S, S), interpolation=cv2.INTER_AREA), f"|err| {meth}"))
    rows.append(np.concatenate(tiles + errs, 1))
cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", a.out)
