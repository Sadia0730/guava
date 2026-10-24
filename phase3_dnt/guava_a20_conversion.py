"""GUAVA variant (a20) ("pure SMPL-X" on the SMPL-X 2020 head = EHM's own SMPL-X, no FLAME): per tracked frame face params.
Model: assets/SMPLX/SMPLX_NEUTRAL_2020.npz (-> third_party/PEAR/assets/SMPLX/SMPLX_NEUTRAL_2020.npz), the one file SPRINT,
GUAVA (a20) and the evaluators use. Identity = the tracker's smplx_shape (200, already in this space; no refit).
Face: (exp50 = SMPL-X 2020 expression 0..49, jaw = SMPL-X joint-22 axis-angle) per tracked frame =
  1. the SPRINT UBody-EHM labels (labels_X/ubody_ehm_gt/<job>.labels.npz face_exp50 / face_jaw, written by
     third_party/PEAR/tools/build_face_labels_smplx2020.py with the per-row LBS refit of the tracker's FLAME expression + jaw +
     eyelids), mapped back to the tracked frames by (clip, frame_key) through the row-aligned processed/ubody_ehm_gt/<job>.npz;
  2. frames not in those labels (rows the label builder dropped as truncated): the same refit (that tool's build_models /
     lstsq_map / convert, imported unchanged, 60 LBFGS iterations), run here.
Check: the refit re-run on N covered frames vs their labels. Eyes: FLAME eye joints (EHM-aligned) vs SMPL-X 2020 eye joints
(joints 23 / 24) per clip identity -> decision copy / zero recorded in the report (the fine-tune reads it).
Output: <out>/face_a20.npz (clip, frame_key, exp50, jaw, src: 0 label / 1 refit here), <out>/report.json."""
import argparse, importlib.util, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c2t/tracked"); ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/guava_a20/conversion")
ap.add_argument("--processed", default="/raid/ubx858/datasets/processed/ubody_ehm_gt"); ap.add_argument("--labels", default="/raid/ubx858/outputs/dnt/sprint/data/labels_X/ubody_ehm_gt")
ap.add_argument("--n-check", type=int, default=1000); ap.add_argument("--eye-tol-mm", type=float, default=5.0, help="copy eye rotations if every clip's FLAME and SMPL-X eye joints are closer than this (a rotation about a centre d mm off moves the eyeball by <= d * angle: < 1.7 mm at 20 deg for 5 mm)")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0"); t0 = time.time()
spec = importlib.util.spec_from_file_location("bfl2020", str(ROOT / "third_party/PEAR/tools/build_face_labels_smplx2020.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
M = T.build_models(dev)
Bt = M["x2020"].B.cpu().numpy().astype(np.float64); Bf = M["flame"].B.cpu().numpy().astype(np.float64)
W_fl = T.lstsq_map(Bt, Bf, M["fitmask"])[0]
# ---- labels index
idx, lab = {}, {}
for f in sorted(Path(a.processed).glob("*.npz")):
    d = np.load(f, allow_pickle=True); l = np.load(Path(a.labels) / f"{f.stem}.labels.npz")
    lab[f.stem] = (l["face_exp50"], l["face_jaw"], l["face_valid"])
    for i, (c, k) in enumerate(zip(d["clip"], d["frame_key"])): idx[(str(c), str(k))] = (f.stem, i)
# ---- tracked frames
clips = sorted(p.parent for p in Path(a.tracked).glob("*/optim_tracking_ehm.pkl"))
C, K, E, J, S, FE, FJ, FL = [], [], [], [], [], [], [], []
ids = {}
for c in clips:
    d = pickle.load(open(c / "optim_tracking_ehm.pkl", "rb")); ids[c.name] = pickle.load(open(c / "id_share_params.pkl", "rb"))
    for k in sorted(d):
        fc = d[k]["flame_coeffs"]
        fe = np.asarray(fc["expression_params"], np.float32).reshape(50); fj = np.asarray(fc["jaw_params"], np.float32).reshape(3)
        fl = np.asarray(fc["eyelid_params"], np.float32).reshape(2)
        C.append(c.name); K.append(k); FE.append(fe); FJ.append(fj); FL.append(fl)
        r = idx.get((c.name, k))
        if r is not None and lab[r[0]][2][r[1]] > 0:
            E.append(lab[r[0]][0][r[1]]); J.append(lab[r[0]][1][r[1]]); S.append(0)
        else:
            E.append(np.zeros(50, np.float32)); J.append(np.zeros(3, np.float32)); S.append(1)
E, J, S, FE, FJ, FL = map(np.asarray, (E, J, S, FE, FJ, FL)); C, K = np.asarray(C), np.asarray(K)
miss = np.flatnonzero(S == 1)
rep = {"frames": int(len(S)), "from_labels": int((S == 0).sum()), "refit_here": int(len(miss)),
       "refit_here_per_clip": {str(c): int(n) for c, n in zip(*np.unique(C[miss], return_counts=True))}}
if len(miss):
    x, j = T.convert(M, "flame", W_fl, FE[miss], FJ[miss], FL[miss], "refit", dev, iters=60); E[miss], J[miss] = x, j
# ---- check: refit re-run on covered frames vs the labels
rng = np.random.default_rng(0); chk = np.sort(rng.choice(np.flatnonzero(S == 0), min(a.n_check, int((S == 0).sum())), replace=False))
x, j = T.convert(M, "flame", W_fl, FE[chk], FJ[chk], FL[chk], "refit", dev, iters=60)
ev_lab = T.evaluate(M, "ubody_ehm_gt", {"e": FE, "jaw": FJ, "lid": FL, "model": "flame"}, W_fl, chk, E[chk], J[chk], dev)[0]
ev_new = T.evaluate(M, "ubody_ehm_gt", {"e": FE, "jaw": FJ, "lid": FL, "model": "flame"}, W_fl, chk, x, j, dev)[0]
x0 = FE[chk].astype(np.float64) @ W_fl.T
ev_lin = T.evaluate(M, "ubody_ehm_gt", {"e": FE, "jaw": FJ, "lid": FL, "model": "flame"}, W_fl, chk, x0.astype(np.float32), FJ[chk], dev)[0]
rep["check_refit_vs_labels"] = {"rows": int(len(chk)), "exp50_abs_diff_mean": float(np.abs(x - E[chk]).mean()), "exp50_abs_diff_max": float(np.abs(x - E[chk]).max()),
                                "jaw_diff_deg_mean": float(np.degrees(np.linalg.norm(j - J[chk], axis=1)).mean()), "jaw_diff_deg_max": float(np.degrees(np.linalg.norm(j - J[chk], axis=1)).max()),
                                "head_err_mm_labels": ev_lab["vertex_err_mm"], "head_err_mm_refit_here": ev_new["vertex_err_mm"],
                                "head_err_mm_linear_copy_jaw": ev_lin["vertex_err_mm"], "src_disp_mm": ev_lab["src_disp_mm"]}
if len(miss):
    rep["refit_here_err_mm"] = T.evaluate(M, "ubody_ehm_gt", {"e": FE, "jaw": FJ, "lid": FL, "model": "flame"}, W_fl, miss, E[miss], J[miss], dev)[0]["vertex_err_mm"]
rep["jaw_change_deg_mean"] = float(np.degrees(np.linalg.norm(J - FJ, axis=1)).mean())
# ---- eyes: FLAME eye joints placed as EHM places the head vs SMPL-X 2020 eye joints, per clip identity (zero pose / expr)
from models.modules.ehm import EHM
from models.modules.flame.lbs import vertices2joints, blend_shapes
ehm = EHM("assets/FLAME", "assets/SMPLX", add_teeth=True).eval()
dist = []
with torch.no_grad():
    for c in sorted(ids):
        ip = ids[c]; ss = torch.as_tensor(ip["smplx_shape"], dtype=torch.float32).reshape(1, -1); fs = torch.as_tensor(ip["flame_shape"], dtype=torch.float32).reshape(1, -1)
        ss = torch.cat([ss, torch.zeros(1, ehm.smplx.n_shape - ss.shape[1])], 1); fs = torch.cat([fs, torch.zeros(1, ehm.flame.n_shape - fs.shape[1])], 1)
        tv = ehm.smplx.v_template[None] + blend_shapes(torch.cat([ss, torch.zeros(1, 50)], 1), ehm.smplx.shapedirs)
        jx = vertices2joints(ehm.smplx.J_regressor, tv)[0]
        fv = ehm.flame.v_template[None] + blend_shapes(torch.cat([fs, torch.zeros(1, 50)], 1), ehm.flame.shapedirs)
        jf = vertices2joints(ehm.flame.J_regressor, fv)[0]
        jf = jf - jf[3:5].mean(0, keepdim=True) + jx[23:25].mean(0, keepdim=True)          # EHM's alignment (eye midpoint)
        dist.append([float((jf[3] - jx[23]).norm() * 1000), float((jf[4] - jx[24]).norm() * 1000), float(jf[3, 0]), float(jx[23, 0]),
                     float((jf[2] - jx[22]).norm() * 1000)])
dist = np.asarray(dist)
eye_ok = bool(dist[:, :2].max() < a.eye_tol_mm and np.all(np.sign(dist[:, 2]) == np.sign(dist[:, 3])))
rep["eyes"] = {"flame_eye_vs_smplx_eye_joint_mm": {"mean": float(dist[:, :2].mean()), "max": float(dist[:, :2].max())},
               "same_side_order": bool(np.all(np.sign(dist[:, 2]) == np.sign(dist[:, 3]))),
               "flame_jaw_vs_smplx_jaw_joint_mm_mean": float(dist[:, 4].mean()), "tol_mm": a.eye_tol_mm,
               "decision": "copy tracker eye_pose_params (FLAME joints 3/4 -> SMPL-X joints 23/24)" if eye_ok else "zero eye pose"}
rep["eye_copy"] = eye_ok; rep["elapsed_s"] = time.time() - t0
np.savez(out / "face_a20.npz", clip=C, frame_key=K, exp50=E.astype(np.float32), jaw=J.astype(np.float32), src=S.astype(np.int8),
         eye_copy=np.array(eye_ok), note=np.array("SMPL-X 2020 exp50 (shapedirs[:,:,300:350]) + SMPL-X jaw_pose per tracked frame; src 0 = SPRINT labels_X ubody_ehm_gt (refit), 1 = same refit run here"))
(out / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
