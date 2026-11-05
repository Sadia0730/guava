"""smoke figure (fixed rules, no selection by result): scored frames of the smoke clip with (1) the largest unseen-region area and
(2) the median unseen area among frames with >= 0.5 %; columns GT | static avatar (worker off) | fused avatar (worker uv) | unseen mask."""
import json, sys
from pathlib import Path
import cv2, numpy as np
E = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/phase6_val"); ds, clip = "dress4d", "00134_Outer_0004"
out = E / "reports" / "smoke_uv"; out.mkdir(parents=True, exist_ok=True)
keys = json.loads((E / "base" / ds / clip / "frames.json").read_text())["test"]
U = {k: cv2.imread(str(E / "renders/smoke_uv_on" / ds / clip / "unseen" / f"{k}.png"), 0) for k in keys}
area = {k: float((U[k] > 127).mean()) for k in keys}; elig = sorted([k for k in keys if area[k] >= 0.005], key=lambda k: area[k])
pick = [max(keys, key=lambda k: area[k]), elig[len(elig) // 2]]
run = json.loads((E / "renders/smoke_uv_on" / ds / clip / "run.json").read_text()); pf = run["per_frame"]
psnr = lambda a, b: 10 * np.log10(1.0 / max(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2, 1e-12))   # noqa: E731
S = 320
def lab(im, t):
    im = cv2.resize(im, (S, S)); cv2.rectangle(im, (0, 0), (S, 22), (0, 0, 0), -1)
    cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); return im
rows = []
for k, why in zip(pick, ("largest unseen area", "median unseen area")):
    g = cv2.imread(str(E / "gt512" / ds / clip / f"{k}.png")); a = cv2.imread(str(E / "renders/smoke_uv_off" / ds / clip / f"{k}.png"))
    b = cv2.imread(str(E / "renders/smoke_uv_on" / ds / clip / f"{k}.png")); ov = g.copy(); m = U[k] > 127; ov[m] = (0.5 * ov[m] + [0, 0, 127]).astype(np.uint8)
    v = pf[str(int(k.split("_")[1]))]["av_version"]
    rows.append(np.hstack([lab(g, f"GT {k} ({why})"), lab(a, f"static avatar, PSNR {psnr(a, g):.2f}"), lab(b, f"fused avatar v{v}, PSNR {psnr(b, g):.2f}"),
                           lab(ov, f"unseen at source: {100 * area[k]:.1f} % of image")]))
cv2.imwrite(str(out / "smoke_uv_turned_away.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out / "smoke_uv_turned_away.jpg", pick, [round(area[k], 3) for k in pick])
