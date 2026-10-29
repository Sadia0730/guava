"""GUAVA-FB diagnostic (read-only): how far the UV Gaussians move off the SMPL-X surface (cm), against how far the person's
matte reaches outside the projected SMPL-X body in the targets (cm at the body's depth). One 4D-DRESS validation group."""
import argparse, json, math, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--group", default="00127/Outer"); ap.add_argument("--out", required=True)
ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data"); ap.add_argument("--n-targets", type=int, default=12)
a = ap.parse_args()
sys.path.insert(0, str(HERE))
import cv2, numpy as np, torch
import fb_common as F
from fb_common import to_dev
os.chdir(F.ROOT); dev = torch.device("cuda:0")
from models.UbodyAvatar import Ubody_Gaussian
from utils.graphics_utils import compute_face_orientation

idx = json.loads((Path(a.data_root) / "dress4d/index.json").read_text())["clips"]
ents = sorted([e for e in idx if f"{e['subject']}/{e['outfit']}" == a.group], key=lambda e: e["name"])
clips = [c for c in F.load_clips_parallel([(str(Path(a.data_root) / "dress4d" / e["name"]), None, True) for e in ents], workers=8) if "keys" in c]
cfg = F.get_cfg(); fbd = F.FBData(cfg, clips)
take = 0; v0 = sorted({clips[0]["recs"][k]["view"] for k in clips[0]["keys"]})[0]
src = (0, next(k for k in clips[0]["keys"] if clips[0]["recs"][k]["view"] == v0))
tg = [(i, k) for i in range(len(clips)) for k in clips[i]["keys"] if clips[i]["recs"][k]["view"] != v0]
tg = tg[::max(1, len(tg) // a.n_targets)][:a.n_targets]

_, inf, ren, _ = F.setup_a20(dev)
st = torch.load(a.ckpt, map_location="cpu", weights_only=False); inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"])
print("loaded step", st.get("step")); del st; inf.eval(); ren.eval()
faces = inf.ehm.smplx.faces_tensor; F_np = faces.cpu().numpy()
lbs = inf.ehm.smplx.lbs_weights; part = lbs.argmax(1)                         # dominant joint per vertex
TORSO = torch.tensor([0, 3, 6, 9, 13, 14], device=lbs.device)                 # pelvis, spine1-3, collars

with torch.no_grad():
    torch.manual_seed(0); vg, ug, _ = inf(to_dev(F.collate1(fbd.source_info(*src)), dev))
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    s0 = to_dev(F.collate1(fbd.target_info(*src)), dev); av(s0)
    V = inf.ehm(s0["smplx_coeffs"], s0["flame_coeffs"])["vertices"]
    _, fscale = compute_face_orientation(V, faces, return_scale=True)
    bf = av._uv_binding_face; off_cm = (av._uv_local_xyz[0].norm(dim=-1) * fscale[0, bf].reshape(-1)) * 100
    torso_g = torch.isin(part[faces[bf, 0]], TORSO)
    nrm_cm = av._uv_local_xyz[0, :, 1] * fscale[0, bf].reshape(-1) * 100          # face frame axis 1 = normal (outward > 0)
    res = {"n_uv_gaussians": int(bf.numel()), "local_pos_norm_tri_units_pct": [float(x) for x in torch.quantile(av._uv_local_xyz[0].norm(dim=-1), torch.tensor([.5, .9, .99, 1.], device=dev))],
           "offset_cm_pct_all": [float(x) for x in torch.quantile(off_cm, torch.tensor([.5, .9, .99, 1.], device=dev))],
           "offset_cm_pct_torso": [float(x) for x in torch.quantile(off_cm[torso_g], torch.tensor([.5, .9, .99, 1.], device=dev))],
           "normal_offset_cm_pct_torso": [float(x) for x in torch.quantile(nrm_cm[torso_g], torch.tensor([.01, .1, .5, .9, .99], device=dev))],
           "face_scale_cm_torso_median": float(fscale[0, bf].reshape(-1)[torso_g].median() * 100),
           "opacity_weighted_offset_cm_torso": float((off_cm[torso_g] * av._uv_opacity[0, torso_g, 0]).sum() / av._uv_opacity[0, torso_g, 0].sum())}
    ex = []
    for ref in tg:
        t = to_dev(F.collate1(fbd.target_info(*ref)), dev)
        Vt = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][0, :lbs.shape[0]]
        Vh = torch.cat([Vt, torch.ones_like(Vt[:, :1])], 1); Vc = (t["w2c_cam"][0] @ Vh.T).T[:, :3]
        itf = float(t["invtanfov"][0]) if torch.is_tensor(t["invtanfov"]) else float(t["invtanfov"])
        p = (Vc[:, :2] * itf / Vc[:, 2:]).cpu().numpy(); px = (p + 1) * 256 - 0.5          # NDC [-1,1] -> 512 px (as GUAVA)
        body = np.zeros((512, 512), np.uint8); cv2.fillPoly(body, [np.round(px[f]).astype(np.int32) for f in F_np], 1)
        m = (t["mask"][0, 0].cpu().numpy() > 0.5); outside = m & (body == 0)
        dist = cv2.distanceTransform((body == 0).astype(np.uint8), cv2.DIST_L2, 5)
        cm_per_px = float(Vc[:, 2].median()) / (itf * 256) * 100
        d = dist[outside] * cm_per_px
        ex.append({"tgt": f"{clips[ref[0]]['name']}/{ref[1]}", "matte_outside_body_frac": float(outside.sum() / max(m.sum(), 1)),
                   "outside_cm_p50": float(np.percentile(d, 50)) if d.size else 0.0, "outside_cm_p90": float(np.percentile(d, 90)) if d.size else 0.0,
                   "outside_cm_max": float(d.max()) if d.size else 0.0})
    res["targets"] = ex
    for k in ("matte_outside_body_frac", "outside_cm_p50", "outside_cm_p90", "outside_cm_max"):
        res["mean_" + k] = float(np.mean([e[k] for e in ex]))
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=1))
