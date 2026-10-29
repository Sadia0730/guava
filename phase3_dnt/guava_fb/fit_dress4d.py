"""GUAVA-FB / 4D-DRESS step 1: refit the dataset's SMPL-X labels to SMPLX_NEUTRAL_2020 (200 shape, 50 exp, full axis-angle hands).

Source labels (established by check_model(), see report): SMPL-X v1.1 GENDERED model (models_smplx_v1_1 SMPLX_{MALE,FEMALE}.npz,
gender from basic_info.pkl), 10 betas, 10 expression, hand pose = 12 PCA components with flat_hand_mean=False (the smplx package's
default hand mean added). This reproduces the provided SMPLX/mesh-f*_smplx.ply to < 1e-6 m. The ply / parameter vertices are in
the world frame of Capture/cameras.pkl (extrinsics = world -> OpenCV camera, x_cam = R x + t); basic_info 'rotation' is NOT
applied (it is a viewing rotation for the visualiser; offset is None in every take).

Refit (per subject, on the GPU, all frames used by the converter = every 5th scan frame of every take of both outfits):
  ONE shape (200 betas) shared by all takes / outfits of the subject; per frame: global orient, 21 body, jaw, 2 eyes, 2 x 15 hand
  joints (axis-angle, flat hand mean = GUAVA smplx_only_forward convention), exp 50, translation. Loss = mean squared vertex
  distance to the gendered mesh (+ 1e-6 |beta|^2). Init: gendered poses (hands: PCA -> full rotations incl. the hand mean),
  translation shifted by the pelvis difference, beta 0, exp 0. Adam: 300 steps (root / transl / shape) + 1500 steps (all).
  Registered check (PREREG GUAVA-FB): per subject median over frames of the mean vertex distance < 5 mm; failing subjects dropped.
Output: <out>/<subject>.npz {take, frame, global_orient (N,3), body_pose (N,21,3), jaw_pose (N,3), leye_pose, reye_pose,
  left_hand_pose (N,15,3), right_hand_pose (N,15,3), exp (N,50), transl (N,3), betas (200,), err_mean_m (N,), err_max_m (N,)}
  and <out>/summary.json.
Usage: python fit_dress4d.py --out DIR [--subjects 00122 ...] [--device cuda:0] [--check-model]"""
import argparse, glob, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch
from smplx.lbs import lbs

DATA = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/4d_dress/extracted")
V11 = Path("/raid/ubx858/datasets/models/smplx_v1_1/models/smplx")
N2020 = Path("/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz")
STRIDE = 5
SUBJECTS = lambda: sorted(s for s in os.listdir(DATA) if s.isdigit())


class Model(torch.nn.Module):
    """SMPL-X from an npz (smplx package lbs), n_shape / n_exp columns taken as in GUAVA's SMPLX (shape [:ns], exp [300:300+ne])."""
    def __init__(self, path, ns, ne, dev):
        super().__init__()
        d = np.load(path, allow_pickle=True)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        self.vt = f(d["v_template"]); sd = np.asarray(d["shapedirs"])
        self.sd = f(np.concatenate([sd[:, :, :ns], sd[:, :, 300:300 + ne]], 2)); self.ns, self.ne = ns, ne
        self.pd = f(np.asarray(d["posedirs"]).reshape(-1, 486).T); self.J = f(d["J_regressor"]); self.W = f(d["weights"])
        par = torch.as_tensor(np.asarray(d["kintree_table"])[0].astype(np.int64), device=dev); par[0] = -1; self.par = par
        self.hmean = {s: f(d[f"hands_mean{s}"]) for s in "lr"}; self.hcomp = {s: f(d[f"hands_components{s}"]) for s in "lr"}
        self.faces = np.asarray(d["f"]).astype(np.int64)

    def forward(self, beta, exp, pose, transl):
        """beta (B, ns) or (1, ns), exp (B, ne), pose (B, 55, 3) axis-angle (flat hands), transl (B, 3) -> verts, joints."""
        B = pose.shape[0]
        if beta.shape[0] != B: beta = beta.expand(B, -1)
        v, j = lbs(torch.cat([beta, exp], 1), pose.reshape(B, -1), self.vt, self.sd, self.pd, self.J, self.par, self.W, pose2rot=True)
        return v + transl[:, None], j + transl[:, None]


def load_take(T, stride=STRIDE):
    b = pickle.load(open(T / "basic_info.pkl", "rb"))
    frames = list(b["scan_frames"])[::stride]
    P = [pickle.load(open(T / "SMPLX" / f"mesh-f{fr}_smplx.pkl", "rb")) for fr in frames]
    return b["gender"], frames, {k: np.stack([p[k] for p in P]).astype(np.float32) for k in P[0]}


def gendered_pose(m, p, dev):
    """4D-DRESS params -> (B, 55, 3) full axis-angle pose (hands: mean + PCA12 @ components)."""
    t = lambda k: torch.as_tensor(p[k], device=dev)
    lh = m.hmean["l"] + t("left_hand_pose") @ m.hcomp["l"][:12]; rh = m.hmean["r"] + t("right_hand_pose") @ m.hcomp["r"][:12]
    return torch.cat([t("global_orient"), t("body_pose"), t("jaw_pose"), t("leye_pose"), t("reye_pose"), lh, rh], 1).reshape(-1, 55, 3)


def gendered_verts(m, p, dev):
    t = lambda k: torch.as_tensor(p[k], device=dev)
    exp = torch.cat([t("expression"), torch.zeros(len(p["expression"]), m.ne - 10, device=dev)], 1) if m.ne > 10 else t("expression")
    return m(t("betas")[:, :m.ns], exp, gendered_pose(m, p, dev), t("transl"))


def read_ply_vertices(path, n=10475):
    b = open(path, "rb").read(); i = b.index(b"end_header\n") + 11
    return np.frombuffer(b[i:i + n * 12], np.float32).reshape(n, 3)


def check_model(dev):
    """Gendered v1.1, PCA 12, flat_hand_mean False vs the provided ply on the first frame of one take per subject."""
    out = {}
    for s in SUBJECTS():
        T = sorted(glob.glob(str(DATA / s / "*" / "Take*")))[0]; T = Path(T)
        g, frames, p = load_take(T, 10 ** 9)
        m = Model(V11 / f"SMPLX_{g.upper()}.npz", 10, 10, dev)
        v, _ = gendered_verts(m, p, dev)
        ply = read_ply_vertices(T / "SMPLX" / f"mesh-f{frames[0]}_smplx.ply")
        out[s] = {"take": str(T.relative_to(DATA)), "gender": g, "max_abs_m": float(np.abs(v[0].cpu().numpy() - ply).max())}
    return out


def fit_subject(subj, dev, n2020, it_a=300, it_b=1500, log=print):
    takes = sorted(Path(p) for p in glob.glob(str(DATA / subj / "*" / "Take*")))
    names, frames, P, gender = [], [], [], None
    for T in takes:
        g, fr, p = load_take(T); gender = g
        names += [f"{subj}_{T.parent.name}_{T.name}"] * len(fr); frames += fr; P.append(p)
    p = {k: np.concatenate([q[k] for q in P]) for k in P[0]}
    mg = Model(V11 / f"SMPLX_{gender.upper()}.npz", 10, 10, dev)
    with torch.no_grad():
        tgt, jg = gendered_verts(mg, p, dev); pose0 = gendered_pose(mg, p, dev)
    B = tgt.shape[0]
    m = n2020
    with torch.no_grad():
        _, j0 = m(torch.zeros(1, 200, device=dev), torch.zeros(B, 50, device=dev), torch.zeros(B, 55, 3, device=dev), torch.zeros(B, 3, device=dev))
        _, jg0 = mg(torch.as_tensor(p["betas"], device=dev), torch.zeros(B, 10, device=dev), torch.zeros(B, 55, 3, device=dev), torch.zeros(B, 3, device=dev))
    pose = pose0.clone().requires_grad_(True)
    transl = (torch.as_tensor(p["transl"], device=dev) + jg0[:, 0] - j0[:, 0]).requires_grad_(True)
    beta = torch.zeros(1, 200, device=dev, requires_grad=True); exp = torch.zeros(B, 50, device=dev, requires_grad=True)

    def loss_fn():
        v, _ = m(beta, exp, pose, transl)
        d2 = ((v - tgt) ** 2).sum(-1)
        return d2.mean() + 1e-6 * (beta ** 2).sum(), d2

    t0 = time.time()
    opt = torch.optim.Adam([{"params": [beta], "lr": 0.05}, {"params": [transl], "lr": 0.005}, {"params": [pose], "lr": 0.0}], lr=0.01)
    root_mask = torch.zeros(1, 55, 1, device=dev); root_mask[:, 0] = 1
    for i in range(it_a):          # stage A: shape, translation, root
        opt.zero_grad(); L, _ = loss_fn(); L.backward(); pose.grad *= root_mask
        opt.param_groups[2]["lr"] = 0.005; opt.step()
    opt = torch.optim.Adam([{"params": [beta], "lr": 0.02}, {"params": [transl], "lr": 0.002}, {"params": [pose], "lr": 0.005},
                            {"params": [exp], "lr": 0.02}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it_b, eta_min=1e-2 * 0.005)
    for i in range(it_b):          # stage B: everything
        opt.zero_grad(); L, _ = loss_fn(); L.backward(); opt.step(); sch.step()
        if i % 500 == 0: log(f"  {subj} it {i} rmse {float(L) ** 0.5 * 1000:.2f} mm")
    with torch.no_grad():
        _, d2 = loss_fn(); d = d2.sqrt()
    err_mean = d.mean(1).cpu().numpy(); err_max = d.max(1)[0].cpu().numpy()
    po = pose.detach().cpu().numpy().astype(np.float32)
    res = {"take": np.array(names), "frame": np.array(frames), "gender": gender,
           "global_orient": po[:, 0], "body_pose": po[:, 1:22], "jaw_pose": po[:, 22], "leye_pose": po[:, 23], "reye_pose": po[:, 24],
           "left_hand_pose": po[:, 25:40], "right_hand_pose": po[:, 40:55], "exp": exp.detach().cpu().numpy().astype(np.float32),
           "transl": transl.detach().cpu().numpy().astype(np.float32), "betas": beta.detach().cpu().numpy()[0].astype(np.float32),
           "err_mean_m": err_mean.astype(np.float32), "err_max_m": err_max.astype(np.float32),
           "src_betas_unique": np.unique(p["betas"].round(5), axis=0)}
    summ = {"subject": subj, "gender": gender, "n_frames": int(B), "n_takes": len(takes), "median_mean_mm": float(np.median(err_mean) * 1000),
            "p90_mean_mm": float(np.percentile(err_mean, 90) * 1000), "max_mean_mm": float(err_mean.max() * 1000),
            "median_max_mm": float(np.median(err_max) * 1000), "n_source_betas": int(len(res["src_betas_unique"])),
            "pass": bool(np.median(err_mean) * 1000 < 5.0), "fit_s": time.time() - t0}
    return res, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--check-model", action="store_true")
    ap.add_argument("--it-a", type=int, default=300); ap.add_argument("--it-b", type=int, default=1500)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device)
    if a.check_model:
        r = check_model(dev); (a.out / "model_check.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r, indent=1)); return
    n2020 = Model(N2020, 200, 50, dev)
    subs = a.subjects or SUBJECTS()
    for s in subs:
        if (a.out / f"{s}.json").exists(): print("skip", s); continue
        try:                                   # several fitters share the subject list: one lock file per subject
            os.close(os.open(a.out / f"{s}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            print("locked", s); continue
        res, summ = fit_subject(s, dev, n2020, a.it_a, a.it_b)
        np.savez(a.out / f"{s}.npz", **res); (a.out / f"{s}.json").write_text(json.dumps(summ, indent=1))
        os.remove(a.out / f"{s}.lock")
        print(json.dumps(summ), flush=True)
    allj = {p.stem: json.loads(p.read_text()) for p in sorted(a.out.glob("0*.json"))}
    (a.out / "summary.json").write_text(json.dumps(allj, indent=1))


if __name__ == "__main__":
    main()
