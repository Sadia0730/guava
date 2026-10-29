"""Registered GUAVA-FB correctness check: re-render converted SMPL-X through GUAVA's own camera / mesh rasteriser (patched,
per-sample invtanfov) on 50 rows per exact-label source (BEDLAM2, 4D-DRESS); mean vertex reprojection error vs the source
label's projection (gt_proj.npz) must be < 1 px at 512. Clips drawn with seed 0; 10 rows max per clip."""
import json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent)); import fb_common as F
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
dev = torch.device("cuda:0"); _, inf, _, _ = F.setup_a20(dev); res = {}
for s in ("bedlam2", "dress4d"):
    cl = sorted(e["name"] for e in json.loads((O / "data" / s / "index.json").read_text())["clips"])
    random.Random(0).shuffle(cl); rows = []
    for c in cl:
        if sum(len(r.get("err_px", [])) if isinstance(r, dict) else 1 for r in rows) >= 50: break
        out = O / "reports/renderer_check" / s; out.mkdir(parents=True, exist_ok=True)
        r = F.check_render(O / "data" / s / c, n=5, dev=dev, inf=inf, out_jpg=str(out / f"{c}.jpg"))
        rows.append({"clip": c, **r}); print(s, c, {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
    res[s] = rows
(O / "reports/renderer_check/renderer_check.json").write_text(json.dumps(res, indent=1, default=float))
