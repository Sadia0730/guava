"""GUAVA-FB fine-tuned on one subject (PREREG extended benchmark amendment 6 (b), (c)). All credit for the model, losses and
optimiser to GUAVA (Zhang et al.); B28k = our full-body training of it.

From B28k, fine-tune on ONE clip's training frames (frames.json "recon", GT records; never a test frame). Source = the clip's
first training frame (as every GUAVA-FB row). Step: inferer on the source -> Gaussians -> 4 target renders (training frames drawn
without replacement, RNG seeded by the clip id) -> GUAVA's Optimization_Loss (lambda_head_crop 0.5, local-offset threshold 3.0,
loss iteration 188k + step) + 0.05 x ArcFace identity loss (B28k's face rule, code from guava_fb_train.py) -> Adam with GUAVA's
parameter groups (configure_optimizers, betas (0, 0.99)), constant lr 5e-5, NaN-zeroed gradients, seed 0.
  --pilot: fit on the first 80 % of the training frames (time order), validate on the last 20 % at --eval-at steps (PSNR / SSIM /
           LPIPS-VGG with score.py's functions, 8-bit render vs GT x mask at 512); nothing saved but the JSON log.
           The torch RNG state is saved / restored around each validation, so step n of a long run equals the end of an n-step run.
  default: fit on all training frames for --steps, write {"model", "render_model", "step", "variant"} to --out (/dev/shm only;
           the job deletes it after the clip's three rows), + a JSON log next to it.
Usage: CUDA_VISIBLE_DEVICES=4 python subject_ft.py --E <root> --clip ubc/91-3003CN5S --steps 1000 --out /dev/shm/.../ft.pt
       python subject_ft.py --E <root> --clip ... --pilot --eval-at 0,250,500,1000,1500,2000 --log pilot.json"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True); ap.add_argument("--clip", required=True, help="dataset/clip")
ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--out", default=""); ap.add_argument("--log", default="")
ap.add_argument("--pilot", action="store_true"); ap.add_argument("--eval-at", default="0,250,500,1000,1500,2000")
ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--n-tgt", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arcface-weight", type=float, default=0.05); ap.add_argument("--face-min-px", type=float, default=48.0)
ap.add_argument("--face-max-deg", type=float, default=60.0)
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
a = ap.parse_args()
assert a.pilot or a.out.startswith("/dev/shm/"), "fine-tuned weights go to /dev/shm only (deleted after the clip's rows)"
E = Path(a.E); ds, name = a.clip.split("/")

import numpy as np                                                                                   # noqa: E402
import torch                                                                                         # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import fb_common as F                                                                                # noqa: E402
from fb_common import NV, to_dev                                                                     # noqa: E402
os.chdir(F.ROOT)
import random                                                                                        # noqa: E402
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
from omegaconf import OmegaConf                                                                      # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                        # noqa: E402
from models.UbodyAvatar.ubody_gaussian import configure_optimizers                                   # noqa: E402
from utils.loss_utils import Optimization_Loss                                                       # noqa: E402
from main.metrics import psnr, ssim                                                                  # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
import onnx_arcface_torch as AF                                                                      # noqa: E402

OFF = 160000 + 28000                                                                                 # B28k's last loss iteration
cfg = F.get_cfg(); OmegaConf.set_readonly(cfg, False); cfg.OPTIMIZE.lambda_head_crop = 0.5
cfg.OPTIMIZE.threshold_local_xyz = 3.0; OmegaConf.set_readonly(cfg, True)
_, inf, ren, _ = F.setup_a20(dev)
st = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
r1 = inf.load_state_dict(st["model"], strict=False); r2 = ren.load_state_dict(st["render_model"], strict=False)
CK = {"ckpt": a.fb_ckpt, "step": st.get("step"), "missing": len(r1.missing_keys) + len(r2.missing_keys)}; del st
F.install_box_guard(); fg = F.FaceGeom(inf, dev)
lossm = Optimization_Loss(cfg).to(dev); lp = LPIPS("vgg", "0.1").to(dev).eval()
arc = AF.OnnxArcFace().to(dev).eval(); arc_fwd = AF.OnnxArcFace.forward.__wrapped__
opt, sch = configure_optimizers(inf, types.SimpleNamespace(learning_rate=a.lr, lr_decay_rate=1.0, lr_decay_iter=max(1, a.steps)), ren)
COS_MAX = math.cos(math.radians(a.face_max_deg))

# ------------------------------------------------------------------------------------------------ data (training frames only)
fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
cl = F.load_clips_parallel([(str(E / "data" / ds / name), None, True)], workers=1); fbd = F.FBData(cfg, cl); recs = set(cl[0]["recs"])
train = [k for k in fj["recon"] if k in recs]
assert fj["source"] == train[0] and not set(train) & set(fj["test"])
if a.pilot:
    n_fit = math.ceil((1 - a.val_frac) * len(train)); fit, val = train[:n_fit], train[n_fit:]
else:
    fit, val = train, []
src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev)
rng = np.random.default_rng([a.seed, zlib.crc32(a.clip.encode())])
CACHE = {} if len(fit) <= 400 else None                                                              # targets on CPU (X-Humans: lazy)


def tinfo(k):
    if CACHE is None: return fbd.target_info(0, k)
    if k not in CACHE: CACHE[k] = fbd.target_info(0, k)
    return CACHE[k]




def rep(d, n, B):
    return {k: (v.repeat_interleave(n, 0) if torch.is_tensor(v) and v.shape[0] == B else v) for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ ArcFace (guava_fb_train.py)
@torch.no_grad()
def target_geometry(t):
    o = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"]); V = o["vertices"][:, :NV]; B = V.shape[0]
    itf = F.batch_invtanfov(t, B, dev)
    hp, _ = F.project_px(o["vertices"][:, fg.head_vid], t["w2c_cam"], itf); box, side, inside = F.face_boxes(hp)
    kp, _ = F.project_px(fg.kps5(V), t["w2c_cam"], itf)
    fwd = F.head_forward_cam(o["joint_transform_mat"], t["w2c_cam"]); cosang = -fwd[:, 2] / fwd.norm(dim=-1).clamp(min=1e-8)
    return box, side, inside, kp, cosang


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


# ------------------------------------------------------------------------------------------------ validation (pilot)
@torch.no_grad()
def validate():
    cpu_s, cuda_s = torch.get_rng_state(), torch.cuda.get_rng_state()
    inf.eval(); ren.eval(); torch.manual_seed(0); vg, ug, _ = inf(src)
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); m = []
    for k in val:
        t = to_dev(F.collate1(fbd.target_info(0, k)), dev); torch.manual_seed(0)
        r = (ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1) * 255).round() / 255
        g = ((t["image"] * t["mask"]).clamp(0, 1) * 255).round() / 255
        m.append([float(psnr(r, g).mean()), float(ssim(r, g)), float(lp(r, g).mean())])
    inf.train(); ren.train(); torch.set_rng_state(cpu_s); torch.cuda.set_rng_state(cuda_s)
    m = np.asarray(m).mean(0); return {"psnr": float(m[0]), "ssim": float(m[1]), "lpips": float(m[2]), "n": len(val)}


# ------------------------------------------------------------------------------------------------ loop
eval_at = sorted({int(x) for x in a.eval_at.split(",")}) if a.pilot else []
n_steps = max(eval_at) if a.pilot else a.steps
log = {"clip": a.clip, **CK, "pilot": a.pilot, "n_train": len(train), "n_fit": len(fit), "n_val": len(val), "steps": n_steps,
       "lr": a.lr, "n_tgt": a.n_tgt, "arcface_weight": a.arcface_weight, "gpu": torch.cuda.get_device_name(0), "val": {}, "train": []}
inf.train(); ren.train(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); t_val = 0.0
if 0 in eval_at:
    tv = time.time(); log["val"]["0"] = validate(); t_val += time.time() - tv; print("val 0", log["val"]["0"], flush=True)
for step in range(1, n_steps + 1):
    ks = [fit[i] for i in rng.choice(len(fit), size=min(a.n_tgt, len(fit)), replace=False)]
    tgt = to_dev(torch.utils.data.default_collate([tinfo(k) for k in ks]), dev)
    vg, ug, _ = inf(src); n = len(ks)
    av = Ubody_Gaussian(cfg.MODEL, rep(vg, n, 1), rep(ug, n, 1), pruning=False); av.init_ehm(inf.ehm)
    rr = ren(av(tgt), tgt["render_cam_params"], bg=0.0)
    extra = {"uv_point_xyz": ug["local_pos"], "uv_point_scale": ug["scales"], "vertices": inf.smplx_deform_res["vertices"], "uv_point_opacity": ug["opacities"],
             "vertex_opacity": vg["opacities"], "vertex_scale": vg["scales"]}
    lm, _ = lossm(rr, tgt, extra, OFF + step); n_arc = 0
    if a.arcface_weight > 0:
        la, n_arc = arcface_loss(rr["renders"], tgt["image"] * tgt["mask"], tgt); lm["arcface_loss"] = la * a.arcface_weight
    loss = sum(lm.values())
    opt.zero_grad(set_to_none=True); loss.backward()
    for p_ in list(inf.parameters()) + list(ren.parameters()):
        if p_.grad is not None: p_.grad.nan_to_num_()
    opt.step(); sch.step()
    if step % 50 == 0 or step == 1:
        el = time.time() - t0 - t_val
        log["train"].append({"step": step, "loss": float(loss), **{k: round(float(v), 5) for k, v in lm.items()}, "n_arcface": n_arc,
                             "s_per_step": el / step, "mem_gb": torch.cuda.max_memory_allocated() / 2**30})
        print(json.dumps(log["train"][-1]), flush=True)
    if step in eval_at:
        tv = time.time(); log["val"][str(step)] = validate(); t_val += time.time() - tv; print("val", step, log["val"][str(step)], flush=True)
log["train_s"] = time.time() - t0 - t_val; log["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
if not a.pilot:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": inf.state_dict(), "render_model": ren.state_dict(), "step": f"ft{a.steps}@{CK['step']}", "variant": "guava_fb",
                "ft": {k: v for k, v in log.items() if k != "train"}}, a.out)
lpath = Path(a.log) if a.log else (Path(a.out).with_suffix(".json") if a.out else None)
if lpath: lpath.parent.mkdir(parents=True, exist_ok=True); lpath.write_text(json.dumps(log, indent=1))
print("done", a.clip, f"{log['train_s'] / 60:.1f} min", flush=True)
