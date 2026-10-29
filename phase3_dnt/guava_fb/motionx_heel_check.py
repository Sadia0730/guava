"""GUAVA-FB / Motion-X: sensitivity of the registered filter (fit_ubc.frame_filter) to the DWPose heel mapping. INFORMATION ONLY,
no data written. The DWPose-134 regressor maps the DWPose heels (rows 20 / 23) to SMPL-X joints 10 / 11 (ball of the foot); this
recomputes check (b) for every stored fit (params_dataset.npz / params_refit.npz of motionx_fit.py) with the heels mapped to the
SMPL-X heel vertices (8846 / 8635) instead, keeping (a) as stored, and reports clip pass rates (limb reading) both ways.
Note: the refits themselves were fitted with the original mapping (heel weight 0.1), so only the evaluation changes here.
Writes O/reports/motionx/heel_sensitivity.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import sprint_avatar_fit as SAF, video_fit as VF                                     # noqa: E402
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
work = Path(sys.argv[1]); dev = torch.device("cuda:0")
f = SAF.SprintFitter(VF.CKPT, dev, iters=0, dwpose_threads=2, cuda_graph=False); L = f.lean
Wv0, Sj0 = L.Wv.clone(), L.Sj.clone()
def lean(fix):
    with torch.no_grad():
        L.Wv.copy_(Wv0); L.Sj.copy_(Sj0)
        if fix:
            for r, v in {20: 8846, 23: 8635}.items(): L.Wv[r].zero_(); L.Wv[r, v] = 1.0; L.Sj[r].zero_()
res = {"a": {"orig": [], "fixed": []}, "b": {"orig": [], "fixed": []}}
z512, z1024 = np.zeros((512, 512), np.uint8), np.zeros((1024, 1024), np.float32)
for fj in sorted((O / "runs/motionx/clips").glob("*/fit.json")):
    info = json.loads(fj.read_text())
    if "a" not in info: continue
    s, c = info["subset"], info["clip"]; d = np.load(work / s / c / "prep.npz")
    for tag, fn, pre in (("a", "params_dataset.npz", "dataset"), ("b", "params_refit.npz", "refit")):
        p = fj.parent / fn
        if not p.exists(): continue
        P = np.load(p); B = len(P["keys"]); idx = list(range(len(d["video_frame"]))) if tag == "a" else info["b"]["frame_idx"]
        rots = SAF.rodrigues(torch.as_tensor(P["pose_aa"], device=dev)); beta = torch.as_tensor(P["beta"], device=dev); exp = torch.as_tensor(P["exp"], device=dev)
        Tt = torch.as_tensor(P["Tt"], device=dev); inv = torch.as_tensor(P["invtanfov"], device=dev)
        frames = info[tag]["frames"]; nall = len(d["video_frame"])
        for fix in (False, True):
            lean(fix)
            with torch.no_grad():
                _, Pk, _ = VF.lean_batch(L, rots, beta, exp); uv = VF.proj(Pk + Tt[:, None], inv).cpu().numpy()
            npass = 0
            for i, j in enumerate(idx):
                fr = frames[i]
                if "mesh_out_frac" not in fr or not np.isfinite(uv[i]).all(): continue
                kp, sc = SAF.to_openpose134(d["kp"][j], d["sc"][j]); H = P["H_o2c"][i].astype(np.float64)
                tgt = (VF.hom(H, kp) + (0.5 if tag == "a" else 0.0)) / VF.CROP
                r = VF.frame_filter(z512, z1024, uv[i], tgt, sc, d["det_box"][j], H)
                npass += bool(fr["pass_a"] and r["pass_b"])
            res[tag]["fixed" if fix else "orig"].append(npass / nall >= VF.CLIP_PASS)
out = {k: {kk: (float(np.mean(v)) if v else None) for kk, v in r.items()} | {"n": len(r["orig"])} for k, r in res.items()}
out["note"] = __doc__
(O / "reports/motionx").mkdir(parents=True, exist_ok=True); (O / "reports/motionx/heel_sensitivity.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:800])
