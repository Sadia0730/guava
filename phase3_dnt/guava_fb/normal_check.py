"""Check of normal_prior.py before pilot C uses it (validation targets only).
(1) The SMPL-X surface is rendered as thin Gaussians lying in its faces (shortest axis = face normal): the rendered normal
    map N^ must equal the face normals, and the D-Normal from the rendered depth must agree with N^ (tests
    gaussian_normals_cam / render_geometry / depth_to_normal and the camera conventions).
(2) Sapiens' axis signs: per axis, the sign that agrees with the SMPL-X normal map on body pixels; then mean cosine
    Sapiens vs SMPL-X (clothing makes it < 1; it should be clearly positive).
(3) Optional --ckpt: the same maps for a trained GUAVA-FB avatar. Overlays for a visual check."""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data"); ap.add_argument("--sources", default="dress4d,bedlam2,motionx")
ap.add_argument("--n", type=int, default=6); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default="")
ap.add_argument("--sapiens", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/models/sapiens/sapiens_1b_normal.pt2")
a = ap.parse_args()
sys.path.insert(0, str(HERE))
import numpy as np, torch, torchvision
import fb_common as F
import normal_prior as NP
from fb_common import to_dev
os.chdir(F.ROOT); dev = torch.device("cuda:0")
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.graphics_utils import compute_face_orientation
from models.UbodyAvatar import Ubody_Gaussian

cfg = F.get_cfg(); _, inf, ren, _ = F.setup_a20(dev); NV = F.NV
if a.ckpt:
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False); inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"]); del st
inf.eval(); ren.eval()
faces = inf.ehm.smplx.faces_tensor; faces = faces[(faces < NV).all(1)]
sap = NP.SapiensNormal(a.sapiens, dev)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def mesh_assets(V, k=6):
    """k Gaussians per face (fixed barycentres), flat along the face normal (frame axis 1, as diag_offsets)."""
    Vf = V[:, faces]; R, sc = compute_face_orientation(V, faces, return_scale=True)
    bary = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6], [.45, .45, .1], [.1, .45, .45]], device=dev)[:k]
    xyz = torch.einsum("kj,bfjc->bfkc", bary, Vf).reshape(1, -1, 3)
    q = quat_xyzw_to_wxyz(rotmat_to_unitquat(R))[:, :, None].expand(-1, -1, k, -1).reshape(1, -1, 4)
    s = sc.reshape(1, -1, 1, 1).expand(-1, -1, k, 1).reshape(1, -1, 1)
    scaling = torch.cat([s * 0.6, torch.full_like(s, 1e-6), s * 0.6], -1)
    n = xyz.shape[1]
    return {"xyz": xyz, "rotation": q, "scaling": scaling, "opacity": torch.ones(1, n, 1, device=dev),
            "features_color": torch.zeros(1, n, 32, device=dev)}


def cos_mean(A, B, m):
    return float(((A * B).sum(1, keepdim=True) * m).sum() / m.sum().clamp(min=1))


rows, prods, vis = [], [], []
for s in a.sources.split(","):
    ents = [e for e in json.loads((Path(a.data_root) / s / "index.json").read_text())["clips"] if e.get("split") == "val"]
    rng = np.random.default_rng(1); pick = [ents[i] for i in rng.choice(len(ents), size=min(a.n, len(ents)), replace=False)]
    clips = [c for c in F.load_clips_parallel([(str(Path(a.data_root) / s / e["name"]), None, True) for e in pick], workers=8) if "keys" in c]
    fbd = F.FBData(cfg, clips)
    for ci, c in enumerate(clips):
        k = c["keys"][int(rng.integers(len(c["keys"])))]
        t = to_dev(F.collate1(fbd.target_info(ci, k)), dev); itf = F.batch_invtanfov(t, 1, dev)
        with torch.no_grad():
            V = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][:, :NV]
            Nm, Dm, Am = NP.render_geometry(ren, mesh_assets(V), t["render_cam_params"], t["w2c_cam"])
            Ndm = NP.depth_to_normal(Dm, itf)
            body = NP.erode((Am > 0.5).float(), 2)
            Ns = sap(t["image"]); m = body * NP.erode((t["mask"][:, :1] > 0.5).float(), 2)
            prods.append(((Ns * Nm) * m).sum((0, 2, 3)) / m.sum())
            row = {"source": s, "clip": c["name"], "key": k, "mesh_dnormal_vs_rendered_cos": cos_mean(Ndm, Nm, body), "_Ns": Ns, "_Nm": Nm, "_m": m}
            if a.ckpt:
                torch.manual_seed(0); src = to_dev(F.collate1(fbd.source_info(ci, c["keys"][0])), dev)
                vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
                Nr, Dr, Ar = NP.render_geometry(ren, av(t), t["render_cam_params"], t["w2c_cam"]); Ndr = NP.depth_to_normal(Dr, itf)
                ma = m * NP.erode((Ar > 0.5).float(), 2); row["_Nr"], row["_Ndr"], row["_ma"] = Nr, Ndr, ma
            rows.append(row)
            vis.append(torch.cat([t["image"], (Nm * 0.5 + 0.5) * body, (Ndm * 0.5 + 0.5) * body], 0))

sign = torch.sign(torch.stack(prods).mean(0)); print("per-axis <Sapiens * SMPL-X> =", torch.stack(prods).mean(0).tolist(), "-> sign", sign.tolist())
S3 = sign.view(1, 3, 1, 1)
for r, v in zip(rows, vis):
    Ns = NP.Fn.normalize(r.pop("_Ns") * S3, dim=1); Nm = r.pop("_Nm"); m = r.pop("_m")
    r["sapiens_vs_smplx_cos"] = cos_mean(Ns, Nm, m)
    extra = [(Ns * 0.5 + 0.5) * m]
    if "_Nr" in r:
        Nr, Ndr, ma = r.pop("_Nr"), r.pop("_Ndr"), r.pop("_ma")
        r["avatar_rendered_vs_sapiens_cos"] = cos_mean(Nr, Ns, ma); r["avatar_dnormal_vs_sapiens_cos"] = cos_mean(Ndr, Ns, ma)
        extra += [(Nr * 0.5 + 0.5) * ma, (Ndr * 0.5 + 0.5) * ma]
    r["_img"] = torch.cat([v] + extra, 0)
torchvision.utils.save_image(torch.cat([r.pop("_img") for r in rows]), out / "normal_check.jpg", nrow=3 + (3 if a.ckpt else 1))
summ = {"sapiens_sign": sign.tolist(), "n": len(rows)}
for key in ("mesh_dnormal_vs_rendered_cos", "sapiens_vs_smplx_cos", "avatar_rendered_vs_sapiens_cos", "avatar_dnormal_vs_sapiens_cos"):
    v = [r[key] for r in rows if key in r]
    if v: summ[key] = {"mean": float(np.mean(v)), "min": float(np.min(v))}
(out / "normal_check.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1)); print(json.dumps(summ, indent=1))
