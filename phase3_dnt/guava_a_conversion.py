"""GUAVA variant (a) ("pure SMPL-X": locked-head SMPL-X head, no FLAME, no head / hand scale, no joint offsets): convert the
EHM-tracker parameters of the 269 C2-T training clips (FLAME face, SMPL-X-2020 identity) to the locked-head SMPL-X.
1. Expression: least-squares linear map M (50 locked-head expression <- 50 FLAME expression) between EHM's FLAME expression
   basis and the locked-head SMPL-X expression basis on the FLAME-region vertices (SMPL-X__FLAME_vertex_ids). Jaw / eye
   rotations are copied (axis-angle about the jaw / eye joints). Eyelid blend shapes have no SMPL-X counterpart (dropped).
   Error: on N random tracked frames, the face-region displacement from neutral of EHM's FLAME head (expression + jaw +
   eyelids) vs the locked-head SMPL-X head (converted expression + same jaw), identity neutral; mean / p95 per-vertex mm
   and relative error.
2. Identity: per clip, locked-head betas (300) fitted by least squares to EHM's rest-pose mesh of that identity (SMPL-X
   2020 shape + FLAME head shape; head / hand scale 1, joint offsets 0, expression 0) over the 10,475 SMPL-X vertices
   except the 361 head-bun vertices; per-clip mean / max vertex error (mm).
Output: <out>/expr_map.npz (M, basis residual), <out>/identity_betas.npz (clip -> betas300, err), <out>/report.json."""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c2t/tracked"); ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/guava_a/conversion")
ap.add_argument("--n-frames", type=int, default=2000); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
from models.modules.ehm import EHM
from smplx.lbs import lbs
ehm = EHM("assets/FLAME", "assets/SMPLX", add_teeth=True).eval()
L = np.load("/home/ubx858/guava/assets/SMPLX/lockedhead/models_lockedhead/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
V11 = np.load("/raid/ubx858/datasets/models/smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
NV = 10475; fid = np.load("assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy"); NF = len(fid)
keep = np.abs(V11["v_template"] - L["v_template"]).max(1) < 1e-6                                    # drop the 361 bun vertices
# --- 1. expression map
Ef = ehm.flame.shapedirs[:NF, :, 300:350].detach().cpu().numpy().reshape(-1, 50).astype(np.float64)
El = L["shapedirs"][fid][:, :, 300:350].reshape(-1, 50).astype(np.float64)
M = np.linalg.lstsq(El, Ef, rcond=None)[0]                                                          # El @ M ~ Ef ; locked = M @ flame
basis_res = float(np.linalg.norm(El @ M - Ef) / np.linalg.norm(Ef))
clips = sorted(p.parent for p in Path(a.tracked).glob("*/optim_tracking_ehm.pkl"))
rng = np.random.default_rng(0); recs = []
for c in clips:
    d = pickle.load(open(c / "optim_tracking_ehm.pkl", "rb")); ks = sorted(d)
    for k in rng.choice(ks, min(8, len(ks)), replace=False): recs.append(d[k]["flame_coeffs"])
pick = rng.choice(len(recs), min(a.n_frames, len(recs)), replace=False); recs = [recs[i] for i in pick]
fe = np.stack([np.asarray(r["expression_params"], np.float32).reshape(50) for r in recs]); fj = np.stack([np.asarray(r["jaw_params"], np.float32).reshape(3) for r in recs])
fl = np.stack([np.asarray(r["eyelid_params"], np.float32).reshape(2) for r in recs])
fl_t = ehm.flame
def flame_disp(e, j, lid):
    B = len(e); vt = fl_t.v_template.detach().float()[None].expand(B, -1, -1); sd = fl_t.shapedirs.detach().float()[:, :, 300:350]
    pose = torch.zeros(B, 15); pose[:, 6:9] = torch.as_tensor(j)
    def run(e_, p_, lid_):
        v = lbs(torch.as_tensor(e_), p_, vt, sd, fl_t.posedirs.detach().float(), fl_t.J_regressor.detach().float(), fl_t.parents.long(), fl_t.lbs_weights.detach().float(), pose2rot=True)[0]
        v = v + fl_t.r_eyelid.float() * torch.as_tensor(lid_)[:, 1:2, None] + fl_t.l_eyelid.float() * torch.as_tensor(lid_)[:, 0:1, None]
        return v[:, :NF]
    return (run(e, pose, lid) - run(np.zeros_like(e), torch.zeros(B, 15), np.zeros_like(lid))).numpy()
vtL = torch.as_tensor(L["v_template"], dtype=torch.float32); sdL = torch.as_tensor(L["shapedirs"][:, :, 300:350], dtype=torch.float32)
pdL = torch.as_tensor(L["posedirs"], dtype=torch.float32).reshape(-1, L["posedirs"].shape[2]).T; JL = torch.as_tensor(L["J_regressor"], dtype=torch.float32)
parL = torch.as_tensor(L["kintree_table"][0].astype(np.int64)); parL[0] = -1; WL = torch.as_tensor(L["weights"], dtype=torch.float32)
def locked_disp(e50, j):
    B = len(e50); pose = torch.zeros(B, 165); pose[:, 66:69] = torch.as_tensor(j)
    run = lambda e_, p_: lbs(torch.as_tensor(e_, dtype=torch.float32), p_, vtL[None].expand(B, -1, -1), sdL, pdL, JL, parL, WL, pose2rot=True)[0][:, fid]
    return (run(e50, pose) - run(np.zeros_like(e50), torch.zeros(B, 165))).numpy()
errs, rels, mags = [], [], []
for s in range(0, len(fe), 250):
    b = slice(s, s + 250); df = flame_disp(fe[b], fj[b], fl[b]); dl = locked_disp((fe[b].astype(np.float64) @ M.T).astype(np.float32), fj[b])
    e = np.linalg.norm(df - dl, axis=2) * 1000; errs.append(e); mags.append(np.linalg.norm(df, axis=2) * 1000)
    rels.append(np.linalg.norm(df - dl, axis=(1, 2)) / np.maximum(np.linalg.norm(df, axis=(1, 2)), 1e-9))
errs, mags, rels = np.concatenate(errs), np.concatenate(mags), np.concatenate(rels)
np.savez(out / "expr_map.npz", M=M, basis_relative_residual=basis_res)
# --- 2. per-clip identity betas
SL = L["shapedirs"][:NV, :, :300][keep].reshape(-1, 300).astype(np.float64); pinv = np.linalg.pinv(SL)
ids, ierr = {}, {}
with torch.no_grad():
    for c in clips:
        sh = pickle.load(open(c / "id_share_params.pkl", "rb"))
        bp = {"shape": torch.as_tensor(sh["smplx_shape"], dtype=torch.float32), "exp": torch.zeros(1, 50), "global_pose": torch.zeros(1, 3),
              "body_pose": torch.zeros(1, 21, 3), "left_hand_pose": torch.zeros(1, 15, 3), "right_hand_pose": torch.zeros(1, 15, 3), "joints_offset": None, "head_scale": torch.ones(1, 3), "hand_scale": torch.ones(1, 3)}
        fp = {"shape_params": torch.as_tensor(sh["flame_shape"], dtype=torch.float32), "expression_params": torch.zeros(1, 50), "jaw_params": torch.zeros(1, 3),
              "eye_pose_params": torch.zeros(1, 6), "pose_params": torch.zeros(1, 3), "eyelid_params": torch.zeros(1, 2)}
        V = ehm(bp, fp, pose_type="aa")["vertices"][0, :NV].numpy().astype(np.float64)
        # compare in the rest pose: the rest mesh is posed with identity rotations (LBS = identity), so V is the shaped template
        tgt = (V - L["v_template"][:NV])[keep].reshape(-1)
        beta = pinv @ tgt; fit = (L["v_template"][:NV] + np.einsum("vcb,b->vc", L["shapedirs"][:NV, :, :300], beta))
        e = np.linalg.norm(fit - V, axis=1)[keep] * 1000
        ids[c.name] = beta.astype(np.float32); ierr[c.name] = (float(e.mean()), float(e.max()))
np.savez(out / "identity_betas.npz", clips=np.array(list(ids)), betas=np.stack(list(ids.values())), err_mean_mm=np.array([v[0] for v in ierr.values()]), err_max_mm=np.array([v[1] for v in ierr.values()]))
rep = {"expression_map": {"basis_relative_residual": basis_res, "frames": int(len(fe)), "face_vertex_err_mm": {"mean": float(errs.mean()), "median": float(np.median(errs)), "p95": float(np.percentile(errs, 95))},
       "face_displacement_mm": {"mean": float(mags.mean()), "p95": float(np.percentile(mags, 95))}, "relative_err": {"mean": float(rels.mean()), "median": float(np.median(rels))},
       "flame_jaw_deg_mean": float(np.degrees(np.linalg.norm(fj, axis=1)).mean()), "flame_expr_norm_mean": float(np.linalg.norm(fe, axis=1).mean()), "eyelid_abs_mean": float(np.abs(fl).mean())},
       "identity": {"clips": len(ids), "vertex_err_mm_mean_over_clips": float(np.mean([v[0] for v in ierr.values()])), "vertex_err_mm_max_over_clips": float(np.max([v[1] for v in ierr.values()]))}}
(out / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
