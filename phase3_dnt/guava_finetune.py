"""Fine-tune GUAVA (released best_160000.pt) on the EHM-tracked UBody TRAINING clips (C2-T phase 1) as one of two variants:
  b  "FLAME face": EHM unchanged (SMPL-X body + FLAME head) but the EHM-only identity extras switched off for every frame:
     head_scale = hand_scale = 1, joints_offset = 0 (the values plain SMPL-X implicitly has).
  a  "pure SMPL-X": no FLAME. The head is the locked-head SMPL-X head (template / shape / 50 expression blend shapes of the
     locked-head model replace the SMPL-X 2020 rows in EHM's buffers; teeth rows kept), driven by SMPL-X expression + jaw +
     eye rotations. Tracker FLAME expression -> locked-head expression via phase3_dnt/guava_a_conversion.py's map; jaw / eyes
     copied; per-clip identity = locked-head betas fitted to the tracker's identity; no head / hand scale, no joint offsets.
Training = GUAVA's own recipe (Optimization_Loss, configure_optimizers param groups, Adam betas (0, 0.99), NaN-zeroed grads),
batch 6, iteration counter offset by 160k so the loss schedule is the late-training one. Source frame: a random other frame
of the same clip (GUAVA's _choose_image); target: the real frame. 10 clips held out (seed 0) for validation: frame 0 as
source, every 4th frame as target, GUAVA's metrics (full image, GT x mask on black); best.pt by validation SSIM (as
GUAVA's trainer) and latest.pt only.
  a20 "pure SMPL-X on the SMPL-X 2020 head" (added 2026-10-23): no FLAME; EHM's own SMPL-X 2020 model (assets/SMPLX/
     SMPLX_NEUTRAL_2020.npz, the file SPRINT and the evaluators use) with its own head: EHM's FLAME-inserted template rows are
     reset to the SMPL-X 2020 template, smplx_only_forward. Shape = the tracker's smplx_shape (200, same space); exp50 + jaw
     per tracked frame from phase3_dnt/guava_a20_conversion.py (SPRINT UBody-EHM labels_X = per-row LBS refit of the tracker's
     FLAME expression + jaw + eyelids; same refit for the frames those labels lack); eye rotations copied (eye joints agree,
     see the conversion report) ; no eyelids, head / hand scale, joint offsets. Use with --exclude-clips (the 56 clips dropped
     by the UBody-EHM label filter) -- excluded from training and validation.
--exclude-clips JSON (list of clip names, any variant): removed from training; validation keeps the seed-0 draw over all
clips minus the excluded ones, and replaces each excluded draw by continuing the same seeded RNG over the remaining clips."""
import argparse, json, os, pickle, random, sys, time, types
from pathlib import Path
import numpy as np, torch, torchvision
from copy import deepcopy
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--variant", required=True, choices=("a", "b", "a20")); ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c2t/tracked")
ap.add_argument("--conversion", default="/raid/ubx858/outputs/dnt/guava_a/conversion"); ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr-end", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=6); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--val-clips", type=int, default=10)
ap.add_argument("--val-every", type=int, default=2000); ap.add_argument("--vis-every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-clips", type=int, default=0)
ap.add_argument("--exclude-clips", default=None, help="JSON list of clip names to exclude from training and validation")
ap.add_argument("--face-conv", default="/raid/ubx858/outputs/dnt/guava_a20/conversion/face_a20.npz", help="a20: per-frame SMPL-X 2020 exp50 / jaw")
a = ap.parse_args(); out = Path(a.out); (out / "checkpoints").mkdir(parents=True, exist_ok=True); (out / "visuals").mkdir(exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
from dataset.data_loader import TrackedData, data_to_tensor, squeeze_params
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
from models.UbodyAvatar.ubody_gaussian import configure_optimizers
from models.modules.ehm import EHM as _EHM
ehm_mod = sys.modules["models.modules.ehm.EHM"]   # the module (the package re-exports the class under the same name)
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.graphics_utils import get_full_proj_matrix
from utils.lmdb import LMDBEngine
from utils.loss_utils import Optimization_Loss, cal_ssim, cal_psnr
from submodules.lpipsPyTorch import LPIPS
cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "configs/train/ubody_512.yaml")))
NV = 10475
conv = None
if a.variant == "a":
    em = np.load(Path(a.conversion) / "expr_map.npz"); idb = np.load(Path(a.conversion) / "identity_betas.npz")
    conv = {"M": em["M"].astype(np.float32), "betas": {str(c): b for c, b in zip(idb["clips"], idb["betas"])}}
if a.variant == "a20":
    fcv = np.load(a.face_conv)
    conv = {"face": {(str(c), str(k)): (e, j) for c, k, e, j in zip(fcv["clip"], fcv["frame_key"], fcv["exp50"], fcv["jaw"])}, "eye_copy": bool(fcv["eye_copy"])}

def finite_crops(r):
    try:
        return all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for k in ("body_crop", "head_crop", "left_hand_crop", "right_hand_crop") for v in r[k].values())
    except Exception:
        return False

class MultiClip(TrackedData):
    """TrackedData over many single-clip EHM-tracker directories (flat optim_tracking_ehm.pkl / id_share_params.pkl / LMDB
    keys '<frame>/body_image'), with the variant's parameter transform applied to every source and target record."""
    def __init__(self, cfg, clip_dirs, split):
        torch.utils.data.Dataset.__init__(self)
        self.cfg = cfg; self.bg_color = 0.0; self.feature_img_size = cfg.MODEL.feature_img_size; self.feature_part_size = cfg.MODEL.feature_part_size
        self.image_size = cfg.MODEL.image_size; self.tanfov, self.focal = 1 / cfg.MODEL.invtanfov, cfg.MODEL.invtanfov; self.split = split
        self.clips = [str(c) for c in clip_dirs]; self.tracked, self.ids, self.keys, self.frames = [], [], [], []
        for ci, c in enumerate(self.clips):
            d = load_pkl(Path(c) / "optim_tracking_ehm.pkl"); ks = [k for k in sorted(d) if finite_crops(d[k])]
            self.tracked.append(d); self.ids.append(load_pkl(Path(c) / "id_share_params.pkl")); self.keys.append(ks)
            sel = ks if split == "train" else ks[1::4]
            self.frames += [(ci, k) for k in sel]
        self._engines = {}
    def _eng(self, ci):
        if ci not in self._engines: self._engines[ci] = LMDBEngine(os.path.join(self.clips[ci], "img_lmdb"), write=False)
        return self._engines[ci]
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        ci, k = self.frames[i]
        if self.split == "train":
            src = random.choice([x for x in self.keys[ci] if x != k]) if len(self.keys[ci]) > 1 else k
        else:
            src = self.keys[ci][0]
        st, si, sm = self._info(ci, src); si = si * sm
        si = torchvision.transforms.functional.resize(si, (self.feature_img_size, self.feature_img_size), antialias=True)
        tt, ti, tm = self._info(ci, k)
        ti = torchvision.transforms.functional.resize(ti, (self.image_size, self.image_size), antialias=True)
        tm = torchvision.transforms.functional.resize(tm, (self.image_size, self.image_size), antialias=True)
        view, full = get_full_proj_matrix(tt["w2c_cam"], self.tanfov)
        tt["render_cam_params"] = {"world_view_transform": view, "full_proj_transform": full, "tanfovx": self.tanfov, "tanfovy": self.tanfov,
                                   "image_height": self.image_size, "image_width": self.image_size, "camera_center": tt["c2w_cam"][:3, 3]}
        rec = {"source": {"image": si}, "target": {"image": ti, "mask": tm}}; rec["source"].update(st); rec["target"].update(tt); return rec
    def _info(self, ci, k):
        e = self._eng(ci); img = e[f"{k}/body_image"].float() / 255.0; msk = e[f"{k}/body_mask"].float() / 255.0
        r = deepcopy(self.tracked[ci][k]); ids = self.ids[ci]
        r["smplx_coeffs"].update({"shape": ids["smplx_shape"][0], "joints_offset": np.zeros_like(ids["joints_offset"][0]),
                                  "head_scale": np.ones(3, np.float32), "hand_scale": np.ones(3, np.float32)})   # both variants: extras off
        r["flame_coeffs"].update({"shape_params": ids["flame_shape"][0]})
        if a.variant == "a":
            fe = np.asarray(r["flame_coeffs"]["expression_params"], np.float32).reshape(50)
            r["smplx_coeffs"]["exp"] = (conv["M"] @ fe).astype(np.float32)
            r["smplx_coeffs"]["shape"] = conv["betas"][Path(self.clips[ci]).name]
            r["smplx_coeffs"]["jaw_pose"] = np.asarray(r["flame_coeffs"]["jaw_params"], np.float32).reshape(1, 3)
            r["smplx_coeffs"]["eye_pose"] = np.asarray(r["flame_coeffs"]["eye_pose_params"], np.float32).reshape(2, 3)
        if a.variant == "a20":                                     # shape stays the tracker's smplx_shape (SMPL-X 2020 space)
            e50, jaw = conv["face"][(Path(self.clips[ci]).name, k)]
            r["smplx_coeffs"]["exp"] = np.asarray(e50, np.float32).reshape(50)
            r["smplx_coeffs"]["jaw_pose"] = np.asarray(jaw, np.float32).reshape(1, 3)
            r["smplx_coeffs"]["eye_pose"] = (np.asarray(r["flame_coeffs"]["eye_pose_params"], np.float32).reshape(2, 3) if conv["eye_copy"]
                                             else np.zeros((2, 3), np.float32))
        r = squeeze_params(data_to_tensor(r))
        RT = r["smplx_coeffs"]["camera_RT_params"]; c2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])); M4 = torch.eye(4); M4[:3, :4] = RT
        w2c = c2c @ M4; r["w2c_cam"], r["c2w_cam"] = w2c, torch.linalg.inv(w2c)
        r["head_box"], r["left_hand_box"], r["right_hand_box"] = self._load_box(r)
        for drop in ("dwpose_raw", "dwpose_rlt", "head_lmk_203", "head_lmk_70", "head_lmk_mp", "left_mano_coeffs", "right_mano_coeffs"):
            r.pop(drop, None)
        return r, img, msk

def load_pkl(p):
    with open(p, "rb") as f: return pickle.load(f)

# ---- model
inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev); ren = GaussianRenderer(cfg.MODEL).to(dev)
ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
inf.load_state_dict(ck["model"], strict=False); ren.load_state_dict(ck["render_model"], strict=False); del ck

def smplx_only_forward(self, body_param_dict, flame_param_dict=None, mano_param_dict=None, static_offset=None, pose_type="rotmat", **kw):
    """Variant (a): EHM's body path only (no FLAME head insertion), SMPL-X jaw / eye rotations from the body params."""
    bp = body_param_dict; shape_params = bp["shape"]; B = shape_params.shape[0]; dv = shape_params.device
    exp = bp.get("exp"); exp = self.expression_params.expand(B, -1) if exp is None else exp
    gp = bp.get("global_pose"); gp = gp.unsqueeze(1) if gp.dim() == 2 else gp
    jaw = bp.get("jaw_pose"); jaw = torch.zeros(B, 1, 3, device=dv) if jaw is None else jaw.reshape(B, 1, 3)
    eye = bp.get("eye_pose"); eye = torch.zeros(B, 2, 3, device=dv) if eye is None else eye.reshape(B, 2, 3)
    if shape_params.shape[-1] < self.smplx.n_shape:
        shape_params = torch.cat([shape_params, torch.zeros(B, self.smplx.n_shape - shape_params.shape[1], device=dv)], 1)
    comps = torch.cat([shape_params[:, :self.smplx.n_shape], exp], 1)
    full_pose = torch.cat([gp, bp["body_pose"], jaw, eye, bp["left_hand_pose"], bp["right_hand_pose"]], 1)
    tv = self.smplx.v_template.clone().unsqueeze(0).expand(B, -1, -1) + ehm_mod.blend_shapes(comps, self.smplx.shapedirs)
    if static_offset is not None: tv = tv + static_offset
    verts, jt, joints, vtm, jtm = ehm_mod.lbs_wobeta(full_pose, tv, self.smplx.posedirs, self.smplx.J_regressor, self.smplx.parents,
                                                     self.smplx.lbs_weights, joints_offset=None, dtype=self.smplx.dtype)
    if not hasattr(self, "head_index"): self.head_index = np.unique(self.flame.head_index)
    return {"vertices": verts, "joints": joints, "joints_transform": jt, "ver_transform_mat": vtm, "joint_transform_mat": jtm,
            "head_vertices": verts[:, self.smplx.smplx2flame_ind][:, self.head_index], "head_ref_joint": joints[:, 23:25].mean(dim=1, keepdim=True),
            "left_hand_vertices": verts[:, self.smplx.smplx2mano_ind["left_hand"]], "left_hand_ref_joint": joints[:, 20:21],
            "right_hand_vertices": verts[:, self.smplx.smplx2mano_ind["right_hand"]], "right_hand_ref_joint": joints[:, 21:22]}

if a.variant == "a":
    L = np.load("/home/ubx858/guava/assets/SMPLX/lockedhead/models_lockedhead/smplx/SMPLX_NEUTRAL.npz")
    with torch.no_grad():
        s = inf.ehm.smplx
        s.v_template[:NV] = torch.as_tensor(L["v_template"], dtype=s.v_template.dtype, device=s.v_template.device)
        s.shapedirs[:NV, :, :300] = torch.as_tensor(L["shapedirs"][:, :, :300], dtype=s.shapedirs.dtype, device=s.shapedirs.device)
        s.shapedirs[:NV, :, 300:350] = torch.as_tensor(L["shapedirs"][:, :, 300:350], dtype=s.shapedirs.dtype, device=s.shapedirs.device)
        inf.ehm.v_template[:NV] = s.v_template[:NV]; inf.v_template.data[:NV] = s.v_template[:NV]
    inf.ehm.forward = types.MethodType(smplx_only_forward, inf.ehm)
if a.variant == "a20":                                             # SMPL-X 2020 head: undo EHM's FLAME-template insertion
    with torch.no_grad():
        s = inf.ehm.smplx
        inf.ehm.v_template[:NV] = s.v_template[:NV]; inf.v_template.data[:NV] = s.v_template[:NV]
    inf.ehm.forward = types.MethodType(smplx_only_forward, inf.ehm)

# ---- data
clips = sorted(p.parent for p in Path(a.tracked).glob("*/optim_tracking_ehm.pkl"))
if a.max_clips: clips = clips[:a.max_clips]
rng = random.Random(a.seed); val = sorted(rng.sample(clips, min(a.val_clips, max(1, len(clips) // 10))))
trn = [c for c in clips if c not in val]
excl_info = None
if a.exclude_clips:                                                # same seed-0 draw; excluded draws replaced from the same RNG
    excl = set(json.loads(Path(a.exclude_clips).read_text()))
    v_drop = [c for c in val if c.name in excl]; keep = [c for c in val if c.name not in excl]
    pool = [c for c in clips if c.name not in excl and c not in val]
    repl = rng.sample(pool, min(len(v_drop), len(pool))) if v_drop else []
    val = sorted(keep + repl); trn = [c for c in clips if c not in val and c.name not in excl]
    excl_info = {"exclude_clips": a.exclude_clips, "n_excluded_present": sum(c.name in excl for c in clips),
                 "val_dropped": [c.name for c in v_drop], "val_replacements": [c.name for c in repl]}
    print("exclude-clips:", json.dumps(excl_info), flush=True)
dtr = MultiClip(cfg, trn, "train"); dva = MultiClip(cfg, val, "valid")
dl = torch.utils.data.DataLoader(dtr, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0)
print(f"variant {a.variant}: {len(trn)} train clips / {len(dtr)} frames; {len(val)} val clips / {len(dva)} frames", flush=True)
ocfg = types.SimpleNamespace(learning_rate=a.lr, lr_decay_rate=a.lr_end, lr_decay_iter=a.iters)
opt, sch = configure_optimizers(inf, ocfg, ren)
lossm = Optimization_Loss(cfg).to(dev); lp = LPIPS("vgg", "0.1").to(dev)
OFF = 160000
info = {"variant": a.variant, "args": vars(a), "train_clips": [c.name for c in trn], "val_clips": [c.name for c in val]}
if excl_info: info["exclusion"] = excl_info
(out / "run_info.json").write_text(json.dumps(info, indent=1))
def to_dev(x):
    if torch.is_tensor(x): return x.to(dev, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    return x
@torch.no_grad()
def validate():
    inf.eval(); ren.eval(); ps, ss, ls, l1 = [], [], [], []
    for ci in range(len(dva.clips)):
        idx = [i for i, (c, _) in enumerate(dva.frames) if c == ci]
        b0 = to_dev(torch.utils.data.default_collate([dva[idx[0]]]))
        vg, ug, _ = inf(b0["source"]); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
        for i in idx:
            b = to_dev(torch.utils.data.default_collate([dva[i]])); t = b["target"]
            torch.manual_seed(0); r = ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
            g = t["image"] * t["mask"]
            ps.append(float(cal_psnr(r, g))); ss.append(float(cal_ssim(r, g))); ls.append(float(lp(r, g))); l1.append(float((r - g).abs().mean()))
    inf.train(); ren.train()
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls)), "l1": float(np.mean(l1)), "frames": len(ps)}
log = open(out / "train_log.jsonl", "a"); best = -1.0
v0 = validate(); log.write(json.dumps({"split": "val", "iter": 0, **v0}) + "\n"); log.flush(); print("val iter 0", v0, flush=True)
inf.train(); ren.train(); it = iter(dl); t0 = time.time(); step = 0
while step < a.iters:
    try: batch = next(it)
    except StopIteration: it = iter(dl); batch = next(it)
    batch = to_dev(batch)
    vg, ug, ex = inf(batch["source"]); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=False); av.init_ehm(inf.ehm)
    rr = ren(av(batch["target"]), batch["target"]["render_cam_params"], bg=0.0)
    extra = {"uv_point_xyz": ug["local_pos"], "uv_point_scale": ug["scales"], "vertices": inf.smplx_deform_res["vertices"], "uv_point_opacity": ug["opacities"],
             "vertex_opacity": vg["opacities"], "vertex_scale": vg["scales"]}
    lm, show = lossm(rr, batch["target"], extra, OFF + step); loss = sum(lm.values())
    opt.zero_grad(set_to_none=True); loss.backward()
    for p_ in list(inf.parameters()) + list(ren.parameters()):
        if p_.grad is not None: p_.grad.nan_to_num_()
    opt.step(); sch.step(); step += 1
    if step % 50 == 0 or step == 1:
        el = time.time() - t0
        rec = {"split": "train", "iter": step, "loss": float(loss), "it_per_s": step / el, "lr": sch.get_last_lr()[0], "mem_gb": torch.cuda.max_memory_allocated() / 2**30,
               **{k: float(v) for k, v in lm.items()}}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    if step % a.vis_every == 0 or step == a.iters:
        with torch.no_grad():
            g = batch["target"]["image"] * batch["target"]["mask"]
            torchvision.utils.save_image(torch.cat([g, rr["renders"].clamp(0, 1)]), out / "visuals" / f"train_{step:06d}.jpg", nrow=a.batch)
    if step % a.val_every == 0 or step == a.iters:
        v = validate(); log.write(json.dumps({"split": "val", "iter": step, **v}) + "\n"); log.flush(); print("val", step, v, flush=True)
        state = {"model": inf.state_dict(), "render_model": ren.state_dict(), "global_iter": OFF + step, "variant": a.variant, "val": v}
        torch.save(state, out / "checkpoints" / "latest.pt")
        if v["ssim"] > best:
            best = v["ssim"]; torch.save(state, out / "checkpoints" / "best.pt")
print("DONE", flush=True)
