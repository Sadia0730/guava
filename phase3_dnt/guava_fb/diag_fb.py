"""GUAVA-FB diagnostics on the validation set (read-only; for design decisions, never for checkpoint choice): per target
PSNR / LPIPS, silhouette coverage (alpha from a second rasterisation pass with constant features vs the StyleMatte matte),
coverage near the head, brightness of the missed matte pixels, and the source-to-target view angle in the body frame."""
import argparse, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data"); ap.add_argument("--tracks", default="dress4d,bedlam2,motionx")
ap.add_argument("--dump-every", type=int, default=0, help="save GT | render | alpha | missed-matte map for every n-th target")
ap.add_argument("--val-step", type=int, default=4); ap.add_argument("--val-caps", default="dress4d:30,bedlam2:20,motionx:10")
a = ap.parse_args()

sys.path.insert(0, str(HERE))
import numpy as np, torch
import fb_common as F
from fb_common import to_dev
os.chdir(F.ROOT)
dev = torch.device("cuda:0")
from models.UbodyAvatar import Ubody_Gaussian
from utils.loss_utils import cal_psnr
from submodules.lpipsPyTorch import LPIPS

tracks = [t for t in a.tracks.split(",") if t]; jobs, meta = [], []
for s in tracks:
    for e in json.loads((Path(a.data_root) / s / "index.json").read_text())["clips"]:
        if e.get("split") == "val":
            jobs.append((str(Path(a.data_root) / s / e["name"]), None, True))
            meta.append({"source": s, "subject": str(e.get("subject", e["name"])), "pair_group": "/".join(str(e[k]) for k in ("subject", "outfit") if k in e)})
loaded = F.load_clips_parallel(jobs, workers=16); clips = []
drop = {s: {k: set(v) for k, v in json.loads((Path(a.data_root) / s / "drop_frames.json").read_text()).items()}
        for s in tracks if (Path(a.data_root) / s / "drop_frames.json").exists()}
for c, m in zip(loaded, meta):
    dk = drop.get(m["source"], {}).get(Path(c["dir"]).name)
    if dk and "keys" in c: c["keys"] = [k for k in c["keys"] if k not in dk]
    if "error" in c or len(c.get("keys", [])) < 2: continue
    c.update(m); clips.append(c)
cfg = F.get_cfg(); fbd = F.FBData(cfg, clips)
caps = {k: int(v) for k, v in (x.split(":") for x in a.val_caps.split(",") if x)}
VAL = {}
for s in tracks:                                                           # = guava_fb_train.build_val (v1b grouping)
    cis = [i for i, c in enumerate(clips) if c["source"] == s]; G = []
    if s == "dress4d":
        by = {}
        for i in cis: by.setdefault(clips[i]["pair_group"], []).append(i)
        for g, ii in sorted(by.items()):
            ii = sorted(ii, key=lambda i: clips[i]["name"]); take = ii[0]
            v0 = sorted({clips[take]["recs"][k]["view"] for k in clips[take]["keys"]})[0]
            src = (take, next(k for k in clips[take]["keys"] if clips[take]["recs"][k]["view"] == v0))
            tg = [(i, k) for i in ii for k in clips[i]["keys"] if clips[i]["recs"][k]["view"] != v0]
            step = max(1, math.ceil(len(tg) / caps[s])); G.append((src, tg[::step][:caps[s]]))
    else:
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::a.val_step]][:caps.get(s, 150)]))
    VAL[s] = G

_, inf, ren, _ = F.setup_a20(dev)
if a.ckpt != "a20":
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False); inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"])
    print("loaded", a.ckpt, "step", st.get("step")); del st
inf.eval(); ren.eval(); lp = LPIPS("vgg", "0.1").to(dev)
Jreg = inf.ehm.smplx.J_regressor.to(dev)


def body_view_dir(t):
    """unit direction from the body to the camera, in the body frame (x = body left, y = up, z = forward)."""
    V = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][0, :Jreg.shape[1]]
    J = Jreg @ V; Jh = torch.cat([J, torch.ones_like(J[:, :1])], 1); Jc = (t["w2c_cam"][0] @ Jh.T).T[:, :3]
    x = Jc[1] - Jc[2]; up = Jc[12] - Jc[0]; z = torch.linalg.cross(x, up); x = torch.linalg.cross(up, z)
    B = torch.stack([x / x.norm(), up / up.norm(), z / z.norm()])          # rows = body axes in camera coords
    return B @ (-Jc[0] / Jc[0].norm())


rows = []
with torch.no_grad():
    for s, G in VAL.items():
        for src, tg in G:
            ss = to_dev(F.collate1(fbd.target_info(*src)), dev); vs = body_view_dir(ss)
            torch.manual_seed(0); vg, ug, _ = inf(to_dev(F.collate1(fbd.source_info(*src)), dev))
            av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
            for ref in tg:
                t = to_dev(F.collate1(fbd.target_info(*ref)), dev); assets = av(t)
                torch.manual_seed(0); r = ren(assets, t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
                ones = dict(assets); ones["features_color"] = torch.ones_like(assets["features_color"])
                alpha = ren.forward_raw(ones, t["render_cam_params"], bg=0.0)["raw_features"][:, 0].clamp(0, 1)
                g = t["image"] * t["mask"]; m = t["mask"][0, 0] > 0.5; al = alpha[0] > 0.5
                hb = t["head_box"][0].long().tolist(); cx, cy = (hb[0] + hb[1]) / 2, (hb[2] + hb[3]) / 2; hs = 0.8 * max(hb[1] - hb[0], hb[3] - hb[2])
                H = torch.zeros_like(m); H[max(0, int(cy - hs)):int(cy + hs), max(0, int(cx - hs)):int(cx + hs)] = True
                lum = g[0].mean(0); miss = m & ~al
                vt = body_view_dir(t); ang = math.degrees(math.atan2(float(vs[0] * vt[2] - vs[2] * vt[0]), float(vs[0] * vt[0] + vs[2] * vt[2])))
                if a.dump_every and len(rows) % a.dump_every == 0:
                    import torchvision; dd = Path(a.out).with_suffix(""); dd.mkdir(parents=True, exist_ok=True)
                    mm = torch.stack([miss.float(), (al & ~m).float(), torch.zeros_like(alpha[0])])           # red = missed matte, green = spill
                    torchvision.utils.save_image(torch.stack([g[0], r[0], alpha[0][None].expand(3, -1, -1), 0.5 * g[0] + mm]), dd / f"{len(rows):04d}_{s}.jpg", nrow=4)
                rows.append({"track": s, "src": f"{clips[src[0]]['name']}/{src[1]}", "tgt": f"{clips[ref[0]]['name']}/{ref[1]}",
                             "psnr": float(cal_psnr(r, g)), "lpips": float(lp(r, g)),
                             "miss_frac": float(miss.sum() / m.sum().clamp(min=1)), "spill_frac": float((al & ~m).sum() / m.sum().clamp(min=1)),
                             "miss_frac_head": float((miss & H).sum() / (m & H).sum().clamp(min=1)),
                             "lum_missed": float(lum[miss].mean()) if miss.any() else None, "lum_matte": float(lum[m].mean()),
                             "rel_view_deg": abs(ang)})
        print(s, "done", len(rows), flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rows, indent=0))
def mean(x): x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None
for s in tracks:
    R = [r for r in rows if r["track"] == s]
    print(s, len(R), {k: round(mean([r[k] for r in R]), 4) for k in ("psnr", "lpips", "miss_frac", "miss_frac_head", "spill_frac", "lum_missed", "lum_matte")})
    for lo, hi in ((0, 45), (45, 135), (135, 181)):
        B = [r for r in R if lo <= r["rel_view_deg"] < hi]
        if B: print(f"   view {lo:3d}-{hi:3d} deg n={len(B):3d}", {k: round(mean([r[k] for r in B]), 4) for k in ("psnr", "lpips", "miss_frac_head")})
