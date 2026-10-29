"""GUAVA-FB training (GUAVA extended to the full body; all credit for the architecture, losses and recipe goes to GUAVA,
Zhang et al.). Implements the registered "GUAVA-FB" section of phase3_dnt/PREREG.md; derived from guava_finetune.py --variant a20.

- init: GUAVA (a20) best.pt, SMPL-X 2020 template reset, smplx_only_forward; per-sample camera patch + empty-box guard (fb_common)
- data: O/data/<source>/index.json (+ clip dirs) for bedlam2 / motionx / dress4d / ubc (split "train"), plus UBody = a20's
  training clips (same seed-0 draw, same exclusion list, a20 face conversion). Mix 30/20/20/15/15: the source of every
  (source image, 3 targets) group is drawn by weight; within a source, the group (clip; 4D-DRESS: subject) is drawn in
  proportion to its frame count (GUAVA / a20 sample frames uniformly). A source missing at run time is an error unless
  --allow-missing (the mix is then renormalised over the present sources and recorded in run_info.json).
- pairs: 4D-DRESS: source and targets from DIFFERENT "view" values of the same subject (any take / outfit); other sources:
  different frames of the same clip. Targets drawn without replacement when possible.
- batch = 2 sources x 3 targets: inferer once on the 2 sources, the Gaussian assets repeated per target, Ubody_Gaussian once,
  6 renders. Losses = GUAVA's Optimization_Loss (lambda_head_crop 0.5, others GUAVA's) + ArcFace identity loss (weight 0.05):
  1 - cos(ArcFace(render crop), ArcFace(GT crop).detach()), 112 x 112 crops aligned (similarity, grid_sample, differentiable)
  by the projected SMPL-X 2020 eye centres / nose tip / mouth corners of the TARGET params; only targets whose GT face box
  (square box of the projected head vertices) is >= 48 px, whose head forward axis is within 60 deg of the camera axis and
  whose 5 landmarks are inside the image.
- optimiser: Adam with GUAVA's param groups (configure_optimizers, betas (0, 0.99)), lr 5e-5 -> 0.1x linear over --iters,
  loss iteration offset 160k, NaN-zeroed grads, seed 0.
- validation every 2k (iteration 0 = a20): dress4d val (per subject: source = frame 0 of the first view of its first take,
  targets = every Nth frame of the other views, <= 150 / subject), bedlam2 / motionx val clips (source = first frame, targets
  = every 4th frame from the 2nd, <= 150 / clip), UBody a20 val clips (exact a20 draw and frame rule). Metrics PSNR / SSIM /
  LPIPS(vgg) / L1 on the full 512 image (GT x mask on black) + face-crop LPIPS (square box of the projected head vertices of
  the target params, resized to 128; frames whose box is < 50 % inside the image are skipped). Every inferer / render call is
  preceded by torch.manual_seed(0) (StyleUNet noise) so validation is deterministic.
- best.pt: eligible iff UBody SSIM >= a20 SSIM - 0.002 AND UBody face-crop LPIPS <= a20 + 0.002; best = max mean SSIM over
  dress4d / bedlam2 / motionx among eligible trained checkpoints. latest.pt (with optimiser state, for --resume) during the
  run; deleted at a successful end.
- --smoke: 200 iterations, then 10 renders [GT | render | 50/50 blend] to <run>/smoke/ and summary.json, then stop.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
REG_MIX = "bedlam2=30,motionx=20,dress4d=20,ubc=15,ubody=15"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True, help="run name (-> O/runs/<name>) or absolute path")
ap.add_argument("--iters", type=int, default=30000); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr-end", type=float, default=0.1)
ap.add_argument("--n-src", type=int, default=2); ap.add_argument("--n-tgt", type=int, default=3)
ap.add_argument("--mix", default=REG_MIX); ap.add_argument("--allow-missing", action="store_true")
ap.add_argument("--data-root", default=str(O / "data"))
ap.add_argument("--val-tracks", default="dress4d,bedlam2,motionx,ubody"); ap.add_argument("--val-every", type=int, default=2000)
ap.add_argument("--val-step", type=int, default=4); ap.add_argument("--val-cap", type=int, default=150)
ap.add_argument("--val-caps", default="dress4d:30,bedlam2:20,motionx:10", help="per-track target caps (per pairing group (subject+outfit) for cross-view, per clip otherwise); UBody uses a20's rule")
ap.add_argument("--vis-every", type=int, default=1000); ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lambda-head-crop", type=float, default=0.5); ap.add_argument("--arcface-weight", type=float, default=0.05)
ap.add_argument("--face-min-px", type=float, default=48.0); ap.add_argument("--face-max-deg", type=float, default=60.0)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke-iters", type=int, default=200)
ap.add_argument("--max-minutes", type=float, default=0, help="stop training after this wall time (timing runs only)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--stop-iter", type=int, default=0, help="stop at this iteration; the lr schedule still runs over --iters (pilots)")
ap.add_argument("--self-target-p", type=float, default=0.0, help="prob. that one target of a non-cross-view source is the source frame itself")
ap.add_argument("--same-frame-p", type=float, default=0.0, help="cross-view sources: prob. per target of the same moment from another camera")
ap.add_argument("--zoom-sources", default="", help="sources with true 2x zoomed targets (warped 1024 crop + rotated camera)")
ap.add_argument("--zoom-p", type=float, default=0.0); ap.add_argument("--zoom-factor", type=float, default=2.0); ap.add_argument("--zoom-head-p", type=float, default=0.25)
ap.add_argument("--edge-weight", type=float, default=0.0, help="Sobel-gradient L1 on source-frame (exactly aligned) targets only")
ap.add_argument("--cloth-crop", type=float, default=0.0, help="clothing-box L1 + LPIPS weight (as GUAVA's head crop), box = torso / hips / thighs / upper arms")
ap.add_argument("--sil-weight", type=float, default=0.0, help="L1 between the Gaussians' coverage (constant-feature render) and the matte")
ap.add_argument("--fullbody-tracks", default="dress4d,bedlam2,motionx", help="tracks of the best.pt mean SSIM (registered: dress4d,bedlam2,motionx)")
ap.add_argument("--cross-view-sources", default="dress4d", help="sources paired across different views (registered: dress4d)")
ap.add_argument("--cross-view-group", default="subject+outfit", help="index fields that define a cross-view pairing group (v1: subject; v1b registered: subject+outfit, so source and target wear the same clothes)")
a = ap.parse_args()
out = Path(a.out) if os.path.isabs(a.out) else O / "runs" / a.out
assert not str(out.resolve()).startswith("/raid"), "write nothing on /raid"
(out / "checkpoints").mkdir(parents=True, exist_ok=True); (out / "visuals").mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
import numpy as np, random, torch, torchvision
import fb_common as F
from fb_common import NV, to_dev
os.chdir(F.ROOT)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
from models.UbodyAvatar import Ubody_Gaussian
from models.UbodyAvatar.ubody_gaussian import configure_optimizers
from utils.loss_utils import Optimization_Loss, cal_ssim, cal_psnr
from submodules.lpipsPyTorch import LPIPS
import onnx_arcface_torch as AF

PAIRING = {s: "cross_view" for s in a.cross_view_sources.split(",") if s}
ZOOM_SOURCES = {s for s in a.zoom_sources.split(",") if s} if a.zoom_p > 0 else set()
EXTRAS = a.self_target_p > 0 or a.same_frame_p > 0 or bool(ZOOM_SOURCES)
FULLBODY_TRACKS = tuple(t for t in a.fullbody_tracks.split(",") if t)
OFF = 160000


def log_print(*x): print(time.strftime("%H:%M:%S"), *x, flush=True)


# ------------------------------------------------------------------------------------------------- data assembly
def parse_mix(s):
    m = {}
    for kv in s.split(","):
        if kv.strip():
            k, v = kv.split("="); m[k.strip()] = float(v)
    return m


mix_req = parse_mix(a.mix)
val_tracks = [t for t in a.val_tracks.split(",") if t.strip()] if not a.smoke else []
need = sorted(set(mix_req) | set(val_tracks))
face_conv, eye_copy = F.load_face_conv()
jobs, meta, problems = [], [], {}
for s in need:
    if s == "ubody":
        trn, val = F.a20_split()
        for split, cl in (("train", trn), ("val", val)):
            for c in cl:
                jobs.append((str(c), face_conv.get(c.name, {}), eye_copy)); meta.append({"source": s, "split": split, "subject": c.name})
        continue
    ip = Path(a.data_root) / s / "index.json"
    if not ip.exists():
        problems[s] = f"missing {ip}"; continue
    idx = json.loads(ip.read_text())
    for e in idx["clips"]:
        if e.get("split") not in ("train", "val"): continue          # test clips are never loaded
        jobs.append((str(Path(a.data_root) / s / e["name"]), None, True)); meta.append({"source": s, "split": e["split"], "subject": str(e.get("subject", e["name"])),
                     "pair_group": "/".join(str(e[k]) for k in a.cross_view_group.split("+") if k in e) or str(e.get("subject", e["name"]))})
log_print(f"loading {len(jobs)} clips from {need}")
t_load = time.time(); loaded = F.load_clips_parallel(jobs, workers=24)
clips, skipped = [], {}
drop_frames = {}                                                   # O/data/<source>/drop_frames.json {clip: [frame_key]} (e.g. UBC front/back flips)
for s in need:
    dp = Path(a.data_root) / s / "drop_frames.json"
    if dp.exists(): drop_frames[s] = {k: set(v) for k, v in json.loads(dp.read_text()).items()}
for c, m in zip(loaded, meta):
    dk = drop_frames.get(m["source"], {}).get(Path(c["dir"]).name)
    if dk and "keys" in c: c["keys"] = [k for k in c["keys"] if k not in dk]
    if "error" in c or len(c.get("keys", [])) < 2:
        skipped.setdefault(m["source"], []).append({"dir": c["dir"], "reason": c.get("error", "fewer than 2 usable frames")}); continue
    c.update(m); clips.append(c)
log_print(f"loaded {len(clips)} clips in {time.time() - t_load:.0f} s; skipped {sum(len(v) for v in skipped.values())}")
from omegaconf import OmegaConf
cfg = F.get_cfg(); OmegaConf.set_readonly(cfg, False); cfg.OPTIMIZE.lambda_head_crop = a.lambda_head_crop; OmegaConf.set_readonly(cfg, True)
fbd = F.FBData(cfg, clips)

# training groups
groups = {}
for s in mix_req:
    cis = [i for i, c in enumerate(clips) if c["source"] == s and c["split"] == "train"]
    G = []
    if PAIRING.get(s) == "cross_view":
        by_subj = {}
        for i in cis: by_subj.setdefault(clips[i].get("pair_group", clips[i]["subject"]), []).append(i)
        for subj, ii in sorted(by_subj.items()):
            fr = [(i, k) for i in ii for k in clips[i]["keys"]]; views = np.array([clips[i]["recs"][k]["view"] for i, k in fr])
            if len(set(views)) >= 2: G.append({"name": subj, "frames": fr, "views": views})
            else: skipped.setdefault(s, []).append({"subject": subj, "reason": "single view"})
    else:
        for i in cis: G.append({"name": clips[i]["name"], "frames": [(i, k) for k in clips[i]["keys"]], "views": None})
    if G: groups[s] = G
    elif s not in problems: problems[s] = "no usable training clips"
for s in mix_req:
    if s not in groups: problems.setdefault(s, "no usable training clips")
missing = [s for s in mix_req if s not in groups]
if missing and not a.allow_missing:
    raise SystemExit(f"sources missing at run time: { {s: problems.get(s) for s in missing} } (use --allow-missing to renormalise the mix)")
mix = {s: w for s, w in mix_req.items() if s in groups}; tot = sum(mix.values()); mix = {s: w / tot for s, w in mix.items()}
src_names = sorted(mix); src_p = np.array([mix[s] for s in src_names])
gw = {s: np.array([len(g["frames"]) for g in groups[s]], np.float64) for s in src_names}; gw = {s: w / w.sum() for s, w in gw.items()}
log_print("effective mix", mix, {s: (len(groups[s]), int(sum(len(g["frames"]) for g in groups[s]))) for s in src_names})


class GroupSampler(torch.utils.data.Dataset):
    """item i -> one source frame + n_tgt targets, from np.random.default_rng([seed, i]) (deterministic, resume-safe)."""
    def __len__(self): return 1 << 40

    def draw(self, i):
        rng = np.random.default_rng([a.seed, i])
        s = src_names[rng.choice(len(src_names), p=src_p)]; g = groups[s][rng.choice(len(groups[s]), p=gw[s])]
        n = len(g["frames"]); si = int(rng.integers(n))
        cand = np.flatnonzero(g["views"] != g["views"][si]) if g["views"] is not None else np.delete(np.arange(n), si)
        ti = rng.choice(cand, size=a.n_tgt, replace=len(cand) < a.n_tgt)
        return s, g, si, ti

    def extras(self, i, s, g, si, ti):
        """pilot options (amendment 3): self target, same-moment cross-view targets, zoomed targets; RNG [seed, i, 7]."""
        ti = np.array(ti).copy(); selfs = [False] * len(ti); zooms = [None] * len(ti)
        if not EXTRAS: return ti, selfs, zooms
        rng = np.random.default_rng([a.seed, i, 7])
        if s not in PAIRING and a.self_target_p > 0 and rng.random() < a.self_target_p:
            ti[0] = si; selfs[0] = True
        if s in PAIRING and a.same_frame_p > 0:
            ci, k = g["frames"][si]; fid = k.split("_", 1)[-1]
            if "_sf" not in g:
                g["_sf"] = {}
                for p_, (cj, kj) in enumerate(g["frames"]): g["_sf"].setdefault((cj, kj.split("_", 1)[-1]), []).append(p_)
            same = [p_ for p_ in g["_sf"].get((ci, fid), []) if g["views"][p_] != g["views"][si]]
            for j in range(len(ti)):
                if same and rng.random() < a.same_frame_p:
                    p_ = same[int(rng.integers(len(same)))]; ti[j] = p_
        if s in ZOOM_SOURCES:
            for j in range(len(ti)):
                if rng.random() < a.zoom_p: zooms[j] = int(rng.integers(1 << 31))
        return ti, selfs, zooms

    def __getitem__(self, i):
        s, g, si, ti = self.draw(i)
        ti, selfs, zooms = self.extras(i, s, g, si, ti)
        src = fbd.source_info(*g["frames"][si]); tl = []
        for j, sf, z in zip(ti, selfs, zooms):
            t = fbd.target_info_zoom(*g["frames"][j], np.random.default_rng(z), a.zoom_factor, a.zoom_head_p) if z is not None else None
            zf = t is not None
            if t is None: t = fbd.target_info(*g["frames"][j])
            t["is_self"] = torch.tensor(float(sf)); t["is_zoom"] = torch.tensor(float(zf)); tl.append(t)
        tg = torch.utils.data.default_collate(tl)
        m = {"source": s, "group": g["name"], "src": "/".join(map(str, (clips[g["frames"][si][0]]["name"], g["frames"][si][1]))),
             "tgt": ["/".join(map(str, (clips[g["frames"][j][0]]["name"], g["frames"][j][1]))) for j in ti],
             "views": None if g["views"] is None else [str(g["views"][si])] + [str(g["views"][j]) for j in ti],
             "self": [bool(x) for x in selfs], "zoom": [bool(x) for x in tg["is_zoom"].tolist()]}
        return {"source": src, "target": tg, "meta": json.dumps(m)}


class FromSampler(torch.utils.data.Sampler):
    def __init__(self, start): self.start = start
    def __iter__(self):
        i = self.start
        while True: yield i; i += 1


# validation tracks: list of (src (ci, k), [targets (ci, k)])
def build_val():
    T = {}; caps = {k: int(v) for k, v in (x.split(":") for x in a.val_caps.split(",") if x)}
    for s in val_tracks:
        cis = [i for i, c in enumerate(clips) if c["source"] == s and c["split"] == "val"]
        if not cis: continue
        G = []
        if s == "ubody":                                                     # exact a20 rule
            for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::4]]))
        elif PAIRING.get(s) == "cross_view":
            by_subj = {}
            for i in cis: by_subj.setdefault(clips[i].get("pair_group", clips[i]["subject"]), []).append(i)
            for subj, ii in sorted(by_subj.items()):
                ii = sorted(ii, key=lambda i: clips[i]["name"]); take = ii[0]
                v0 = sorted({clips[take]["recs"][k]["view"] for k in clips[take]["keys"]})[0]
                src = (take, next(k for k in clips[take]["keys"] if clips[take]["recs"][k]["view"] == v0))
                tg = [(i, k) for i in ii for k in clips[i]["keys"] if clips[i]["recs"][k]["view"] != v0]
                if not tg: continue
                cap = caps.get(s, a.val_cap); step = max(1, math.ceil(len(tg) / cap)); G.append((src, tg[::step][:cap]))
        else:
            for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::a.val_step]][:caps.get(s, a.val_cap)]))
        if G: T[s] = G
    return T


VAL = build_val()
for s in val_tracks:
    if s not in VAL and not a.allow_missing and not a.smoke:
        raise SystemExit(f"validation track {s} has no data (use --allow-missing)")
log_print("val tracks", {s: (len(G), sum(len(t) for _, t in G)) for s, G in VAL.items()})

# ------------------------------------------------------------------------------------------------- model
cfg_, inf, ren, ck_meta = F.setup_a20(dev)
F.install_box_guard()
fg = F.FaceGeom(inf, dev)
lossm = Optimization_Loss(cfg).to(dev); lp = LPIPS("vgg", "0.1").to(dev)
arc = AF.OnnxArcFace().to(dev).eval()
arc_fwd = AF.OnnxArcFace.forward.__wrapped__                    # forward without @torch.no_grad (gradients to the render crop)
import types
ocfg = types.SimpleNamespace(learning_rate=a.lr, lr_decay_rate=a.lr_end, lr_decay_iter=a.iters)
opt, sch = configure_optimizers(inf, ocfg, ren)
COS_MAX = math.cos(math.radians(a.face_max_deg))


def git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", str(F.ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(F.ROOT), "diff", "--quiet", "HEAD", "--", "phase3_dnt"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception as e:
        return repr(e)


run_info = {"what": "GUAVA-FB training (GUAVA extended to the full body; credit GUAVA)", "git": git_sha(), "args": vars(a), "out": str(out),
            "init": {k: v for k, v in ck_meta.items() if k not in ("missing", "unexpected")} | {"missing_n": [len(x) for x in ck_meta["missing"]], "unexpected_n": [len(x) for x in ck_meta["unexpected"]]},
            "mix_requested": mix_req, "mix_effective": mix, "missing_sources": {s: problems.get(s) for s in missing},
            "train": {s: {"groups": len(groups[s]), "frames": int(sum(len(g["frames"]) for g in groups[s])),
                          "clips": sorted({clips[i]["name"] for g in groups[s] for i, _ in g["frames"]})} for s in src_names},
            "val": {s: [{"source": "/".join((clips[src[0]]["name"], src[1])), "n_targets": len(tg)} for src, tg in G] for s, G in VAL.items()},
            "skipped": skipped, "lambda_head_crop": cfg.OPTIMIZE.lambda_head_crop, "arcface_weight": a.arcface_weight}
(out / "run_info.json").write_text(json.dumps(run_info, indent=1, default=str))


# ------------------------------------------------------------------------------------------------- helpers
def flat(x):
    if torch.is_tensor(x): return x.reshape(-1, *x.shape[2:])
    if isinstance(x, dict): return {k: flat(v) for k, v in x.items()}
    return x


def rep(d, n, B):
    return {k: (v.repeat_interleave(n, 0) if torch.is_tensor(v) and v.shape[0] == B else v) for k, v in d.items()}


@torch.no_grad()
def target_geometry(t):
    o = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"]); V = o["vertices"][:, :NV]; B = V.shape[0]
    itf = F.batch_invtanfov(t, B, dev)
    hp, _ = F.project_px(o["vertices"][:, fg.head_vid], t["w2c_cam"], itf); box, side, inside = F.face_boxes(hp)   # full array (EHM appends teeth)
    kp, _ = F.project_px(fg.kps5(V), t["w2c_cam"], itf)
    fwd = F.head_forward_cam(o["joint_transform_mat"], t["w2c_cam"]); cosang = -fwd[:, 2] / fwd.norm(dim=-1).clamp(min=1e-8)
    return box, side, inside, kp, cosang


SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dev) / 8
CLOTH_JOINTS = torch.tensor([0, 1, 2, 3, 6, 9, 13, 14, 16, 17], device=dev)          # pelvis, hips (thighs), spine1-3, collars, shoulders (upper arms)
CLOTH_VID = torch.nonzero(torch.isin(inf.ehm.smplx.lbs_weights[:NV].argmax(1), CLOTH_JOINTS)).flatten()


def grad_img(x):
    w = torch.stack([SOBEL, SOBEL.T])[:, None].repeat(x.shape[1], 1, 1, 1)
    return torch.nn.functional.conv2d(x, w, padding=1, groups=x.shape[1])


def pilot_losses(rr, t, assets):
    """amendment 3 (pilot B): clothing-box L1 + LPIPS, edge loss on source-frame targets, Gaussian-coverage silhouette loss."""
    out_ = {}; r = rr["renders"]; g = t["image"] * t["mask"]; B = r.shape[0]
    if a.cloth_crop > 0:
        with torch.no_grad():
            V = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][:, CLOTH_VID]
            px, _ = F.project_px(V, t["w2c_cam"], F.batch_invtanfov(t, B, dev))
            lo = torch.quantile(px, 0.02, dim=1); hi = torch.quantile(px, 0.98, dim=1)
            box = torch.stack([lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]], 1).clamp(0, r.shape[-1] - 1).long()
        lam = [lossm.cfg.lambda_l1 * a.cloth_crop, lossm.cfg.lambda_perpetual_high * a.cloth_crop]
        out_["cloth_loss"] = lossm.cal_box_loss(r, g, box, [lossm.l1_loss_f, lossm.perpetual_loss_f], lam)
    if a.edge_weight > 0:
        sm = t["is_self"] > 0.5
        out_["edge_loss"] = (a.edge_weight * (grad_img(r[sm]) - grad_img(g[sm])).abs().mean()) if bool(sm.any()) else r.sum() * 0.0
    if a.sil_weight > 0:
        ones = dict(assets); ones["features_color"] = torch.ones_like(assets["features_color"])
        alpha = ren.forward_raw(ones, t["render_cam_params"], bg=0.0)["raw_features"][:, :1]
        out_["sil_loss"] = a.sil_weight * (alpha - t["mask"][:, :1]).abs().mean()
    return out_


def arcface_loss(render, gt, t):
    box, side, inside, kp, cosang = target_geometry(t)
    kin = ((kp >= 0) & (kp <= 511)).all(-1).all(-1)
    ok = (side >= a.face_min_px) & (cosang >= COS_MAX) & kin
    if not bool(ok.any()): return render.sum() * 0.0, 0
    ii = torch.nonzero(ok).flatten()
    th = torch.stack([AF.align_theta(kp[i].double().cpu().numpy(), render.shape[-1]) for i in ii.tolist()]).to(dev)
    er = arc_fwd(arc, arc.preprocess(AF.crops(render[ii], th)))
    with torch.no_grad(): eg = arc_fwd(arc, arc.preprocess(AF.crops(gt[ii], th)))
    return (1 - (er * eg).sum(-1)).mean(), int(ok.sum())


@torch.no_grad()
def face_lpips(r, g, t):
    box, side, inside, _, _ = target_geometry(t)
    if float(inside[0]) < 0.5 or float(side[0]) < 4: return None
    b = box[0].clamp(0, r.shape[-1]).round().long().tolist(); x0, y0, x1, y1 = b
    if x1 - x0 < 2 or y1 - y0 < 2: return None
    rc = torch.nn.functional.interpolate(r[:, :, y0:y1, x0:x1], (128, 128), mode="bilinear", antialias=True, align_corners=False)
    gc = torch.nn.functional.interpolate(g[:, :, y0:y1, x0:x1], (128, 128), mode="bilinear", antialias=True, align_corners=False)
    return float(lp(rc, gc))


@torch.no_grad()
def build_avatar(src):
    torch.manual_seed(0); vg, ug, _ = inf(to_dev(F.collate1(fbd.source_info(*src)), dev))
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); return av


@torch.no_grad()
def validate():
    inf.eval(); ren.eval(); res = {}; t0 = time.time()
    for s, G in VAL.items():
        m = {"psnr": [], "ssim": [], "lpips": [], "l1": [], "face_lpips": []}
        for src, tg in G:
            av = build_avatar(src)
            for ref in tg:
                t = to_dev(F.collate1(fbd.target_info(*ref)), dev)
                torch.manual_seed(0); r = ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
                g = t["image"] * t["mask"]
                m["psnr"].append(float(cal_psnr(r, g))); m["ssim"].append(float(cal_ssim(r, g))); m["lpips"].append(float(lp(r, g)))
                m["l1"].append(float((r - g).abs().mean())); fl = face_lpips(r, g, t)
                if fl is not None: m["face_lpips"].append(fl)
        res[s] = {k: float(np.mean(v)) if v else None for k, v in m.items()} | {"frames": len(m["psnr"]), "face_frames": len(m["face_lpips"])}
    inf.train(); ren.train()
    fb = [res[s]["ssim"] for s in FULLBODY_TRACKS if s in res]
    res["fullbody_mean_ssim"] = float(np.mean(fb)) if fb else None; res["fullbody_tracks"] = [s for s in FULLBODY_TRACKS if s in res]
    res["seconds"] = time.time() - t0
    return res


def eligible(v, base):
    if "ubody" not in v or "ubody" not in base: return False
    return (v["ubody"]["ssim"] >= base["ubody"]["ssim"] - 0.002) and (v["ubody"]["face_lpips"] <= base["ubody"]["face_lpips"] + 0.002)


# ------------------------------------------------------------------------------------------------- loop
log = open(out / "train_log.jsonl", "a"); step, best, base = 0, None, None
if a.resume and (out / "checkpoints/latest.pt").exists():
    ck = torch.load(out / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    inf.load_state_dict(ck["model"]); ren.load_state_dict(ck["render_model"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["scheduler"])
    step, best, base = ck["step"], ck["best"], ck["a20_val"]; del ck
    log_print("resumed at", step)
ds = GroupSampler()
dl = torch.utils.data.DataLoader(ds, batch_size=a.n_src, sampler=FromSampler(step * a.n_src), num_workers=a.workers,
                                 persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
it = iter(dl)                                                        # fork the workers before the main process opens LMDBs
if base is None and VAL:
    base = validate(); log.write(json.dumps({"split": "val", "iter": 0, **base}) + "\n"); log.flush(); log_print("val iter 0 (a20)", json.dumps(base))
inf.train(); ren.train(); t0 = time.time(); t_data = 0.0; nonfinite = []; torch.cuda.reset_peak_memory_stats(); n0 = step
stop_iter = a.smoke_iters if a.smoke else (min(a.stop_iter, a.iters) if a.stop_iter else a.iters)
while step < stop_iter:
    td = time.time(); batch = next(it); t_data += time.time() - td
    meta_b = [json.loads(m) for m in batch.pop("meta")]
    batch = to_dev(batch, dev); src = batch["source"]; tgt = flat(batch["target"])
    vg, ug, ex = inf(src)
    av = Ubody_Gaussian(cfg.MODEL, rep(vg, a.n_tgt, a.n_src), rep(ug, a.n_tgt, a.n_src), pruning=False); av.init_ehm(inf.ehm)
    assets = av(tgt); rr = ren(assets, tgt["render_cam_params"], bg=0.0)
    extra = {"uv_point_xyz": ug["local_pos"], "uv_point_scale": ug["scales"], "vertices": inf.smplx_deform_res["vertices"], "uv_point_opacity": ug["opacities"],
             "vertex_opacity": vg["opacities"], "vertex_scale": vg["scales"]}
    lm, show = lossm(rr, tgt, extra, OFF + step)
    n_arc = 0
    if a.arcface_weight > 0:
        la, n_arc = arcface_loss(rr["renders"], tgt["image"] * tgt["mask"], tgt); lm["arcface_loss"] = la * a.arcface_weight
    if a.cloth_crop > 0 or a.edge_weight > 0 or a.sil_weight > 0:
        lm.update(pilot_losses(rr, tgt, assets))
    loss = sum(lm.values())
    opt.zero_grad(set_to_none=True); loss.backward()
    for p_ in list(inf.parameters()) + list(ren.parameters()):
        if p_.grad is not None: p_.grad.nan_to_num_()
    opt.step(); sch.step(); step += 1
    vals = {k: float(v) for k, v in lm.items()}
    if not all(math.isfinite(v) for v in vals.values()) or not math.isfinite(float(loss)): nonfinite.append({"iter": step, **vals})
    if step % a.log_every == 0 or step == 1:
        el = time.time() - t0
        rec = {"split": "train", "iter": step, "loss": float(loss), "it_per_s": (step - n0) / el, "data_wait_frac": t_data / el, "lr": sch.get_last_lr()[0],
               "mem_gb": torch.cuda.max_memory_allocated() / 2**30, "mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30, "n_arcface": n_arc,
               "sources": [m["source"] for m in meta_b], "pairs": meta_b, **vals}
        log.write(json.dumps(rec) + "\n"); log.flush(); log_print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    if step % a.vis_every == 0:
        with torch.no_grad():
            torchvision.utils.save_image(torch.cat([tgt["image"] * tgt["mask"], rr["renders"].clamp(0, 1)]), out / "visuals" / f"train_{step:06d}.jpg", nrow=a.n_src * a.n_tgt)
    if not a.smoke and VAL and (step % a.val_every == 0 or step == a.iters):
        v = validate(); el = eligible(v, base)
        state = {"model": inf.state_dict(), "render_model": ren.state_dict(), "global_iter": OFF + step, "step": step, "variant": "guava_fb", "val": v, "eligible": el}
        is_best = el and v["fullbody_mean_ssim"] is not None and (best is None or v["fullbody_mean_ssim"] > best["fullbody_mean_ssim"])
        if is_best:
            best = {"iter": step, "fullbody_mean_ssim": v["fullbody_mean_ssim"]}; torch.save(state, out / "checkpoints/best.pt")
        log.write(json.dumps({"split": "val", "iter": step, "eligible": el, "is_best": is_best, **v}) + "\n"); log.flush()
        log_print("val", step, "eligible", el, "best", best, json.dumps(v))
        torch.save(state | {"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "best": best, "a20_val": base}, out / "checkpoints/latest.pt")
    if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
        log_print("max-minutes reached at", step); break

summary = {"iters_done": step, "it_per_s": (step - n0) / (time.time() - t0), "data_wait_frac": t_data / (time.time() - t0),
           "peak_mem_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "nonfinite_iters": nonfinite[:20], "n_nonfinite": len(nonfinite), "all_losses_finite": not nonfinite}
if a.smoke:
    sd = out / "smoke"; sd.mkdir(exist_ok=True); inf.eval(); ren.eval()
    with torch.no_grad():
        for j in range(10):
            s, g, si, ti = ds.draw(10**9 + j)
            av = build_avatar(g["frames"][si]); ref = g["frames"][ti[0]]
            t = to_dev(F.collate1(fbd.target_info(*ref)), dev)
            torch.manual_seed(0); r = ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1); gt = t["image"] * t["mask"]
            torchvision.utils.save_image(torch.cat([gt, r, 0.5 * gt + 0.5 * r], 3), sd / f"smoke_{j:02d}_{s}_{clips[ref[0]]['name']}_{ref[1]}.jpg")
    (sd / "summary.json").write_text(json.dumps(summary, indent=1)); log_print("SMOKE", json.dumps(summary)); sys.exit(0)
(out / "summary.json").write_text(json.dumps(summary | {"best": best, "a20_val": base}, indent=1))
if step >= a.iters:
    if best is None: log_print("NO ELIGIBLE CHECKPOINT: registered best.pt criterion failed")
    lt = out / "checkpoints/latest.pt"
    if lt.exists(): lt.unlink()
    log_print("DONE", best)
