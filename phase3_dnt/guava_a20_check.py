"""GUAVA (a20) head check: on N random tracked frames (seed 0) compare the a20 mesh (guava_finetune.py's smplx_only_forward
on EHM's untouched SMPL-X 2020 buffers; shape = tracker smplx_shape, exp50 / jaw from guava_a20_conversion.py, eyes per its
decision) with EHM's own mesh (SMPL-X body + FLAME head) under variant (b)'s settings (head_scale = hand_scale = 1, joint
offsets 0; tracker FLAME expression / jaw / eyes / eyelids). Posed, model space (before the camera), mm.
Regions (SMPL-X__FLAME_vertex_ids, FLAME_masks): face, head without eyeballs, eyeballs; body = SMPL-X vertices outside the
FLAME head region; teeth excluded. Decomposition: identity only (all face params 0 in both), and expression displacement
(each mesh minus its own neutral-face mesh, same pose). Output: <out>/head_check.json."""
import argparse, ast, json, os, pickle, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c2t/tracked")
ap.add_argument("--face-conv", default="/raid/ubx858/outputs/dnt/guava_a20/conversion/face_a20.npz"); ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/guava_a20/conversion")
ap.add_argument("--n", type=int, default=300); a = ap.parse_args()
from models.modules.ehm import EHM
ehm_mod = sys.modules["models.modules.ehm.EHM"]
src = (ROOT / "phase3_dnt/guava_finetune.py").read_text(); fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "smplx_only_forward")
ns = {"ehm_mod": ehm_mod, "np": np, "torch": torch}; exec(compile(ast.Module([fn], []), "guava_finetune.py", "exec"), ns); smplx_only_forward = ns["smplx_only_forward"]
dev = torch.device("cuda:0"); ehm = EHM("assets/FLAME", "assets/SMPLX", add_teeth=True).to(dev).eval()
NV = 10475; fid = np.asarray(ehm.smplx.smplx2flame_ind)[:5023]; assert fid.max() < NV   # FLAME vertices (teeth rows appended after them excluded)
m = pickle.load(open("assets/FLAME/FLAME_masks/FLAME_masks.pkl", "rb"), encoding="latin1")
eyeb = np.zeros(len(fid), bool); eyeb[m["left_eyeball"]] = True; eyeb[m["right_eyeball"]] = True
face = np.zeros(len(fid), bool); face[m["face"]] = True
body = np.setdiff1d(np.arange(NV), fid)
W = ehm.smplx.lbs_weights[:NV].cpu().numpy(); headj = [12, 15, 22, 23, 24]
far = np.setdiff1d(np.flatnonzero(W[:, headj].sum(1) < 1e-6), fid)                 # body vertices with no neck / head / jaw / eye weight
F = np.load(a.face_conv); conv = {(str(c), str(k)): (e, j) for c, k, e, j in zip(F["clip"], F["frame_key"], F["exp50"], F["jaw"])}; eye_copy = bool(F["eye_copy"])
keys = list(conv); rng = np.random.default_rng(0); pick = [keys[i] for i in rng.choice(len(keys), a.n, replace=False)]
T = lambda x, s=None: torch.as_tensor(np.asarray(x, np.float32), device=dev)[None] if s is None else torch.as_tensor(np.asarray(x, np.float32).reshape(s), device=dev)[None]
cache, R = {}, {k: [] for k in ("face", "head_no_eyeballs", "eyeballs", "eyeballs_eye_zero", "id_face", "id_head_no_eyeballs", "exp_face", "exp_head_no_eyeballs",
                               "exp_face_src_disp", "body_all", "body_far", "joints_body", "a20_eyeball_vs_ehm_eye_zero_both")}
def run(clip, k, neutral=False, eye=True):
    if clip not in cache:
        cache[clip] = (pickle.load(open(Path(a.tracked) / clip / "optim_tracking_ehm.pkl", "rb")), pickle.load(open(Path(a.tracked) / clip / "id_share_params.pkl", "rb")))
    d, ip = cache[clip]; r = d[k]; sc, fc = r["smplx_coeffs"], r["flame_coeffs"]
    bp = {"shape": T(ip["smplx_shape"][0]), "exp": T(sc["exp"]), "global_pose": T(sc["global_pose"]), "body_pose": T(sc["body_pose"]),
          "left_hand_pose": T(sc["left_hand_pose"]), "right_hand_pose": T(sc["right_hand_pose"]), "joints_offset": T(np.zeros_like(ip["joints_offset"][0])),
          "head_scale": T(np.ones(3)), "hand_scale": T(np.ones(3))}
    z = lambda n: np.zeros(n, np.float32)
    fp = {"shape_params": T(ip["flame_shape"][0]), "expression_params": T(z(50) if neutral else fc["expression_params"], 50), "jaw_params": T(z(3) if neutral else fc["jaw_params"], 3),
          "eye_pose_params": T(z(6) if (neutral or not eye) else fc["eye_pose_params"], 6), "pose_params": T(fc["pose_params"], 3), "eyelid_params": T(z(2) if neutral else fc["eyelid_params"], 2)}
    with torch.no_grad():
        oe = ehm(bp, fp)
        e, j = conv[(clip, k)]
        b2 = {kk: v for kk, v in bp.items() if kk not in ("joints_offset", "head_scale", "hand_scale")}
        b2.update({"exp": T(z(50) if neutral else e), "jaw_pose": T(z(3) if neutral else j, (1, 3)),
                   "eye_pose": T(z(6) if (neutral or not (eye and eye_copy)) else fc["eye_pose_params"], (2, 3))})
        oa = smplx_only_forward(ehm, b2)
    return oe["vertices"][0, :NV].cpu().numpy(), oa["vertices"][0, :NV].cpu().numpy(), oe["joints_transform"][0].cpu().numpy(), oa["joints_transform"][0].cpu().numpy()
mm = lambda x, y: np.linalg.norm(x - y, axis=-1) * 1000
for clip, k in pick:
    ve, va, je, ja = run(clip, k); ne, na, _, _ = run(clip, k, neutral=True); ve0, va0, _, _ = run(clip, k, eye=False)
    he, ha = ve[fid], va[fid]
    R["face"].append(mm(he, ha)[face]); R["head_no_eyeballs"].append(mm(he, ha)[~eyeb]); R["eyeballs"].append(mm(he, ha)[eyeb])
    R["eyeballs_eye_zero"].append(mm(he, va0[fid])[eyeb]); R["a20_eyeball_vs_ehm_eye_zero_both"].append(mm(ve0[fid], va0[fid])[eyeb])
    R["id_face"].append(mm(ne[fid], na[fid])[face]); R["id_head_no_eyeballs"].append(mm(ne[fid], na[fid])[~eyeb])
    de, da = he - ne[fid], ha - na[fid]
    R["exp_face"].append(mm(de, da)[face]); R["exp_head_no_eyeballs"].append(mm(de, da)[~eyeb]); R["exp_face_src_disp"].append(np.linalg.norm(de, axis=-1)[face] * 1000)
    R["body_all"].append(mm(ve[body], va[body])); R["body_far"].append(mm(ve[far], va[far]))
    R["joints_body"].append(mm(je[:22], ja[:22]))
st = lambda x: {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}
rep = {"frames": len(pick), "eye_copy": eye_copy, "n_body_vertices": int(len(body)), "n_body_far_vertices": int(len(far)),
       **{k: st(np.concatenate([np.ravel(x) for x in v])) for k, v in R.items()},
       "body_far_bitwise_identical_frames": int(sum(float(np.max(x)) == 0.0 for x in R["body_far"]))}
Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "head_check.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
