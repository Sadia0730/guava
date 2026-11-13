"""LHM++ (aigc3d/LHM-plusplus @ 6fa576c, LHMPP-700M) timing for the laptop timing kit (phase3_dnt/timing).

Times the evaluated pipeline of phase3_dnt/baselines/lhmpp/run_lhmpp.py (row "LHM++" = `lhmpp`, source frame only), with the
same inputs, conversions, render crop, renderer and composite, but batch 1 and phase brackets (phase_timer.Phases). The LHM++
code is unchanged; phases are bracketed by wrapping its methods at runtime (see NOTES below / the JSON "notes").

Usage (normally through run_lhmpp.sh, which sets the env exactly as baselines/lhmpp/lhmpp_run.sh):
  python time_lhmpp.py --root SUBSET_ROOT --out OUT_DIR/lhmpp.json --pretrained .../LHM-plusplus_data/pretrained_models

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

NOTES (what each phase contains; also written to the JSON)
Reconstruction, once per person (inputs decoded from PNG / npz before timing; disk reads not timed):
  preprocess   run_lhmpp.py's source preparation: package crop x StyleMatte alpha (> 0.5) on white, crop to the alpha box,
               SrcImagePipeline(PadRatioWithScale 5/3, 840) -> 840 x 504; package betas (200, SMPL-X 2020) -> LHM++'s 10 by least
               squares; tensors to the GPU.
  segmentation null: the evaluated pipeline used the package's StyleMatte masks (LHM++'s rembg / BiRefNet not run).
  pose         null: GT SMPL-X shape from the package (LHM++'s own pose / shape estimator not run).
  network      ModelHumanA4OLRM.infer_single_view (renderer.get_query_points with the betas, forward_latent_points = DINOv2
               encoder + multimodal transformer (eager: torch._dynamo disabled, as evaluated), renderer.forward_gs = Gaussian
               attribute heads).
Animation, per driving frame (GT pose given, batch 1, test keys cycled):
  preprocess   run_lhmpp.py's per-frame inputs: GT SMPL-X 2020 forward of the frame (projected vertex box -> LHM++'s evaluation
               crop via _process_mask_frame_for_eval, 420 px wide 5:3 render), expression 50 -> FLAME 100 map, SMPL-X param
               dict / camera tensors to the GPU.
  deformation  renderer.animate_gs_model (SMPL-X voxel skinning: _prepare_smplx_data, _transform_points = LBS of the 160k
               Gaussians, _compute_rotations, _create_gaussian_models).
  rendering    from the end of animate_gs_model to the 512 output image: renderer._render_views (gsplat rasterisation of RGB +
               128-d features at 700 x 420, = rendering_raster), neural_renderer (patch_4dptonly DPT decoder -> RGB + alpha,
               = rendering_refiner), animation_infer's resize / paste into the 512 board (CPU) and torch.cuda.empty_cache(), and
               run_lhmpp.py's composite onto black (rgb - (1 - alpha)) to uint8. PNG writing is not timed.
  total        end to end per frame (preprocess + ModelHumanA4OLRM.animation_infer + composite).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _pick_openblas_coretype():
    # Must run before numpy (and phase_timer, which imports numpy) is imported.
    if os.environ.get("OPENBLAS_CORETYPE"): return os.environ["OPENBLAS_CORETYPE"] + " (preset)"
    flags = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"): flags = line; break
    except OSError:
        pass
    ct = "SkylakeX" if " avx512f" in flags else ("Haswell" if " avx2" in flags else "")
    if ct: os.environ["OPENBLAS_CORETYPE"] = ct
    return ct or "unset"


OPENBLAS = _pick_openblas_coretype()
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase_timer import PROTOCOL, Phases, sync, write  # noqa: E402

REPO = HERE.parent.parent
SNAP = "huggingface/models--3DAIGC--LHMPP-700M/snapshots/fc9f73664b9bfcc457e96210ddc06fe7caf0d559"

ap = argparse.ArgumentParser(description="LHM++ reconstruction / animation timing (laptop timing kit)")
ap.add_argument("--root", required=True, help="timing subset root (manifest.json, <E>/base/<ds>/<clip>/)")
ap.add_argument("--clips", default="", help="comma list of clip ids (ds/clip); default all clips of the manifest")
ap.add_argument("--out", required=True, help="output JSON path (e.g. OUT_DIR/lhmpp.json)")
ap.add_argument("--warmup", type=int, default=PROTOCOL["warmup"], help="animation warm-up frames (not recorded)")
ap.add_argument("--iters", type=int, default=PROTOCOL["iters"], help="animation timed frames")
ap.add_argument("--recon-warmup", type=int, default=PROTOCOL["recon_warmup"], help="reconstruction warm-up builds")
ap.add_argument("--recon-reps", type=int, default=PROTOCOL["recon_reps"], help="reconstruction timed builds")
ap.add_argument("--pretrained", default=os.environ.get("LHMPP_PRETRAINED", str(REPO / "pretrained_models")),
                help="LHM++ pretrained_models dir (from lhmpp_pretrained.tar: LHM-plusplus_data/pretrained_models; the parent "
                     "LHM-plusplus_data is accepted too)")
ap.add_argument("--lhm", default=os.environ.get("LHMPP_CODE", str(Path.home() / "LHM-plusplus")), help="LHM++ checkout (commit 6fa576c)")
ap.add_argument("--smplx2020", default=str(REPO / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), help="package SMPL-X 2020 model")
ap.add_argument("--views", type=int, default=1, help="reference views as run_lhmpp.py (1 = row LHM++, source frame only)")
ap.add_argument("--render-size", type=int, default=420, help="as run_lhmpp.py (LHM++ render_image.high)")
ap.add_argument("--composite", default="unpremul_white", choices=["unpremul_white", "mul_alpha"], help="as run_lhmpp.py")
ap.add_argument("--save-png", type=int, default=0, help="sanity only: after timing, render this many test keys per clip to "
                                                        "<out dir>/lhmpp_png/<ds>/<clip>/<key>.png")
ap.add_argument("--compare-eval", default="", help="sanity only: eval root holding <E>/renders/lhmpp/<ds>/<clip>/<key>.png "
                                                   "(read only); the --save-png frames are compared with it (PSNR)")
ap.add_argument("--power-note", default="", help="e.g. 'laptop on AC, performance mode' (written to env.power_note)")
a = ap.parse_args()

OUT = Path(a.out).resolve(); OUTDIR = OUT.parent; OUTDIR.mkdir(parents=True, exist_ok=True)
ROOT = Path(a.root).resolve()
LHM = Path(a.lhm).resolve()
PRE = Path(a.pretrained).resolve()
if not (PRE / "human_model_files").exists() and (PRE / "pretrained_models" / "human_model_files").exists():
    PRE = PRE / "pretrained_models"
assert (PRE / "human_model_files").exists(), f"no human_model_files under {PRE} (--pretrained)"
MODEL = PRE / SNAP
if not MODEL.exists():
    cand = sorted((PRE / "huggingface/models--3DAIGC--LHMPP-700M/snapshots").glob("*/config.json"))
    assert cand, f"LHMPP-700M snapshot not found under {PRE}"
    MODEL = cand[0].parent
SMPLX2020 = Path(a.smplx2020)
assert SMPLX2020.exists(), f"{SMPLX2020} missing (--smplx2020: the package's SMPLX_NEUTRAL_2020.npz)"

# LHM++ reads its assets through relative paths ("./pretrained_models/..."), so it runs from its checkout (as run_lhmpp.py).
# When --pretrained is not the checkout's own pretrained_models, run from a throw-away directory under the output dir that
# mirrors the checkout with symlinks (code and inputs only, read through) plus pretrained_models -> --pretrained.
SHADOW = None
if (LHM / "pretrained_models").exists() and (LHM / "pretrained_models").resolve() == PRE and not os.environ.get("LHMPP_TIMING_SHADOW"):
    CWD = LHM
else:
    SHADOW = Path(tempfile.mkdtemp(prefix=".lhmpp_cwd_", dir=OUTDIR))
    for e in LHM.iterdir():
        if e.name != "pretrained_models": (SHADOW / e.name).symlink_to(e)
    (SHADOW / "pretrained_models").symlink_to(PRE)
    CWD = SHADOW


def _cleanup_shadow():
    if SHADOW is None or not SHADOW.exists(): return
    for e in SHADOW.iterdir():
        if e.is_symlink(): e.unlink()          # unlink only the links; never follow them
    try:
        SHADOW.rmdir()
    except OSError as ex:
        print("note: could not remove", SHADOW, ex, flush=True)


import atexit  # noqa: E402
atexit.register(_cleanup_shadow)
os.chdir(CWD); sys.path.insert(0, str(LHM))

import cv2  # noqa: E402
import smplx  # noqa: E402
import torch  # noqa: E402
torch._dynamo.config.disable = True        # as run_lhmpp.py: forward_latent_points' @torch.compile runs eager
from core.datasets.data_utils import SrcImagePipeline  # noqa: E402
from core.models import model_dict  # noqa: E402
from core.runners.infer.utils import _process_mask_frame_for_eval  # noqa: E402
from core.utils.hf_hub import wrap_model_hub  # noqa: E402

dev = "cuda"
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, "numpy BLAS wrong (OPENBLAS_CORETYPE)"
assert np.abs(np.linalg.lstsq(_A, _A @ _X, rcond=None)[0] - _X).max() < 1e-9, "numpy lstsq wrong (OPENBLAS_CORETYPE)"
import subprocess  # noqa: E402
COMMIT = subprocess.run(["git", "-C", str(LHM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
GPU = torch.cuda.get_device_name(0)

# ---------------- fixed linear conversions (person independent; built once per process like the model, not timed) --------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)                  # (10475, 3, 400)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
lhm_sx = np.load(PRE / "human_model_files/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
assert np.abs(lhm_sx["v_template"] - Z["v_template"]).max() == 0 and np.abs(lhm_sx["shapedirs"][:, :, :10] - Z["shapedirs"][:, :, :10]).max() == 0
face_idx = np.load(PRE / "human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
flame = smplx.create(str(PRE / "human_model_files"), "flame", gender="neutral", num_betas=10, num_expression_coeffs=100)
EF = flame.expr_dirs.detach().cpu().numpy().astype(np.float64).reshape(-1, 100)                    # (5023*3, 100)
ES = SD[face_idx, :, 300:350].reshape(-1, 50)                                                      # (5023*3, 50)
MEXP, *_ = np.linalg.lstsq(EF, ES, rcond=None)                                                     # (100, 50): e100 = MEXP @ e50
body2020 = smplx.SMPLX(str(SMPLX2020), num_betas=200, num_expression_coeffs=50, use_pca=False, flat_hand_mean=True).to(dev)


def betas10(b200):                                   # run_lhmpp.py betas10
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    r = (S10 @ b - S200 @ b200).reshape(-1, 3); n = np.linalg.norm(r, axis=1) * 1000
    return b.astype(np.float32), float(n.mean()), float(n.max())


def expr100(e50):                                    # run_lhmpp.py expr100 without the residual diagnostic
    return (MEXP @ e50.T.astype(np.float64)).T.astype(np.float32)


# ---------------- model (loaded once per process; not part of any timed phase) ----------------
from accelerate import Accelerator  # noqa: E402
Accelerator()
t0 = time.time()
model = wrap_model_hub(model_dict["human_lrm_a4o"]).from_pretrained(str(MODEL)); model.to(dev); model.eval()
sync(); LOAD_S = time.time() - t0
PARAM_DTYPES = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
try:
    import core.models.encoders.sonata.model as _son
    FLASH = _son.flash_attn is not None
except Exception:                                                                              # noqa: BLE001
    FLASH = None
print(f"model loaded in {LOAD_S:.1f} s ({COMMIT[:7]}, {GPU}, params {PARAM_DTYPES}, flash_attn {FLASH}, OPENBLAS_CORETYPE {OPENBLAS})", flush=True)
pipe = SrcImagePipeline(dict(name="PadRatioWithScale", target_ratio=5 / 3, tgt_max_size_list=[840], val=True))

# ---------------- runtime phase wrappers on the official methods (inactive unless ST["on"]) ----------------
ST = {"P": None, "on": False, "t_def_end": None}
R = model.renderer


def _wrap(obj, name, phase, mark_end=False):
    orig = getattr(obj, name)

    def w(*args, **kw):
        if not ST["on"]: return orig(*args, **kw)
        with ST["P"](phase):
            out = orig(*args, **kw)
        if mark_end: ST["t_def_end"] = time.perf_counter()   # Phases synchronised at the end of the bracket
        return out
    setattr(obj, name, w)


_wrap(R, "animate_gs_model", "deformation", mark_end=True)      # GSPlatFeatRenderer.forward_animate_gs -> self.animate_gs_model
_wrap(R, "_render_views", "rendering_raster")                   # GSPlatFeatRenderer.forward_animate_gs -> self._render_views
_wrap(model.neural_renderer, "forward", "rendering_refiner")    # animation_infer -> self.neural_renderer(...)


def prep_source(img, alpha):                         # run_lhmpp.py prep_source
    m = (alpha.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    rgb = img.astype(np.float32) / 255.0 * m[..., None] + (1 - m[..., None])
    ys, xs = np.where(m > 0)
    crop = rgb[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    out = pipe([crop])[0]
    return (out * 255).astype(np.uint8)


def rd(p, flag=cv2.IMREAD_COLOR):
    x = cv2.imread(str(p), flag)
    if x is None: raise FileNotFoundError(p)
    return x[..., ::-1].copy() if flag == cv2.IMREAD_COLOR else x


PKEYS = ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose", "trans", "expr", "focal", "princpt", "img_size_wh")


def frame_inputs(D, k, betas_t, tmat):
    """run_lhmpp.py's per-test-frame preparation for one frame -> animation_infer kwargs (batch 1, one view)."""
    sp, cams, i = D["sp"], D["cams"], D["sk"][k]
    K = cams["K"][D["ck"][k]].astype(np.float64)
    with torch.no_grad():
        g = lambda n: torch.from_numpy(sp[n][i: i + 1].reshape(1, -1)).float().to(dev)   # noqa: E731
        V = body2020(betas=D["betas200_t"], global_orient=g("global_orient"), body_pose=g("body_pose"), jaw_pose=g("jaw_pose"),
                     leye_pose=g("leye_pose"), reye_pose=g("reye_pose"), left_hand_pose=g("left_hand_pose"),
                     right_hand_pose=g("right_hand_pose"), expression=g("expression"), transl=g("transl")).vertices[0].cpu().numpy()
    uv = (V @ K.T)[:, :2] / (V @ K.T)[:, 2:3]
    x0, y0 = np.clip(np.floor(uv.min(0)), 0, 511).astype(int); x1, y1 = np.clip(np.ceil(uv.max(0)), 0, 511).astype(int)
    boxm = np.zeros((512, 512), np.uint8); boxm[y0: y1 + 1, x0: x1 + 1] = 255
    I4 = torch.eye(4); I4[0, 0], I4[1, 1], I4[0, 2], I4[1, 2] = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr, cm, off, _ = _process_mask_frame_for_eval(boxm, None, 0.0, I4, a.render_size, 5 / 3.0, 14, res_scale=1.0,
                                                    resize_to_1024=False, mask_already_padded=False, rendering_with_pad=False)
    e100 = expr100(sp["expression"][i: i + 1])[0]
    Pf = {"root_pose": sp["global_orient"][i], "body_pose": sp["body_pose"][i], "jaw_pose": sp["jaw_pose"][i],
          "leye_pose": sp["leye_pose"][i], "reye_pose": sp["reye_pose"][i], "lhand_pose": sp["left_hand_pose"][i],
          "rhand_pose": sp["right_hand_pose"][i], "trans": sp["transl"][i], "expr": e100,
          "focal": np.array([K[0, 0], K[1, 1]], np.float32), "princpt": np.array([K[0, 2], K[1, 2]], np.float32),
          "img_size_wh": np.array([512, 512], np.float32)}
    bp = {"betas": betas_t, "transform_mat_neutral_pose": tmat}
    bp.update({q: torch.from_numpy(np.stack([Pf[q]]).astype(np.float32))[None].to(dev) for q in PKEYS})   # (1, 1, ...)
    return dict(query_points=D["build"]["qp"], gs_model_list=D["build"]["gs"], smplx_params=bp,
                render_c2ws=torch.eye(4)[None, None].to(dev), render_intrs=intr[None, None].to(dev),
                render_bg_colors=torch.ones(1, 1, 3).to(dev), gs_hidden_features=D["build"]["feat"],
                image_latents=D["build"]["lat"], motion_emb=D["build"]["mot"], offset_list=[off], mask_seqs=[cm],
                output_rgb=torch.zeros(512, 512, 3), **({"pos_emb": D["build"]["pos"]} if D["build"]["pos"] is not None else {}))


def composite(rgb, al):                              # run_lhmpp.py output image (before cv2.imwrite)
    r = rgb[0].clamp(0, 1).numpy(); m = al[0].clamp(0, 1).numpy()[..., None]
    o = np.clip(r - (1 - m), 0, 1) * (m > 0) if a.composite == "unpremul_white" else r * m
    return (o * 255 + 0.5).astype(np.uint8)


def build(D, P):
    """One avatar build as run_lhmpp.py (inputs already decoded in D)."""
    with P("preprocess"):
        ref_imgs = np.stack([prep_source(*D["src"][q]) for q in D["refs"]])
        b10, sh_mean, sh_max = betas10(D["sp"]["betas"])
        refs_t = torch.from_numpy(ref_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)[None].to(dev)
        betas_t = torch.from_numpy(b10)[None].to(dev)
    with P("network"):
        with torch.no_grad():
            outs = model.infer_single_view(refs_t, None, None, render_c2ws=D["c2w1"], render_intrs=D["intr1"],
                                           render_bg_colors=D["bg1"], smplx_params={"betas": betas_t},
                                           ref_imgs_bool=torch.ones(1, len(D["refs"]), dtype=torch.bool, device=dev))
    gs, qp, tmat, feat, lat, mot = outs[:6]
    return dict(gs=gs, qp=qp, tmat=tmat, feat=feat, lat=lat, mot=mot, pos=outs[6] if len(outs) == 7 else None,
                betas_t=betas_t, shape_fit_mm=(round(sh_mean, 3), round(sh_max, 3)), src_img=ref_imgs[0])


man = json.load(open(ROOT / "manifest.json"))
want = [c for c in a.clips.split(",") if c]
todo = [c for c in man["clips"] if not want or c["id"] in want]
assert todo, f"no clips selected from {[c['id'] for c in man['clips']]}"
missing = set(want) - {c["id"] for c in todo}
assert not missing, f"unknown clips {missing}"
protocol = dict(PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=a.recon_warmup, recon_reps=a.recon_reps,
                precision="fp32", render_px=512)
NOTES = (
    "LHM++ (aigc3d/LHM-plusplus @ %s, LHMPP-700M, row `lhmpp` = source frame only), pipeline of phase3_dnt/baselines/lhmpp/"
    "run_lhmpp.py at batch 1, LHM++ code unchanged, phases bracketed by wrapping its methods at runtime. Precision fp32 (params %s, "
    "no autocast; flash_attn %s, as evaluated; torch._dynamo disabled as evaluated). Model load %.1f s (once per process, not "
    "timed). Inputs decoded from PNG / npz before timing (disk reads and PNG writes not timed). "
    "RECONSTRUCTION preprocess = prep_source (StyleMatte alpha > 0.5 on white, crop to alpha box, PadRatioWithScale 5/3 840 -> "
    "840x504) + betas 200 -> 10 least squares + upload; segmentation null (package StyleMatte masks used, LHM++ rembg/BiRefNet "
    "not run); pose null (GT SMPL-X shape given); network = ModelHumanA4OLRM.infer_single_view (renderer.get_query_points, "
    "forward_latent_points = DINOv2-L encoder + multimodal transformer, renderer.forward_gs = Gaussian heads). "
    "ANIMATION preprocess = GT SMPL-X 2020 forward of the frame -> projected box -> LHM++ eval crop (_process_mask_frame_for_eval, "
    "%d px wide, 5:3 -> 700x420 render) + expression 50 -> FLAME 100 + param / camera upload; deformation = "
    "renderer.animate_gs_model (voxel-skinning LBS of the Gaussians, rotations); rendering = end of animate_gs_model -> 512 uint8 "
    "image: renderer._render_views (gsplat RGB + 128-d feature splat, = rendering_raster) + neural_renderer.forward "
    "(patch_4dptonly DPT decoder, = rendering_refiner) + animation_infer's CPU resize / paste into the 512 board and "
    "torch.cuda.empty_cache() + run_lhmpp.py composite onto black (%s); total = end to end per frame. Differences from the "
    "evaluation run: one frame per animation_infer call (evaluation: 20 views per call, which loops views one by one inside, and "
    "one batched SMPL-X forward for all test frames); the build is given only the betas (the evaluation also passed all test-frame "
    "params, which infer_single_view does not read)." % (COMMIT[:7], PARAM_DTYPES, FLASH, LOAD_S, a.render_size, a.composite))

clips_out = {}; peak_all = (0, 0)
torch.cuda.reset_peak_memory_stats()
for c in todo:
    ds, clip, cid = c["dataset"], c["clip"], c["id"]
    B = ROOT / c["E"] / "base" / ds / clip
    fr = json.load(open(B / "frames.json")); test = list(fr["test"]); src = fr["source"]; recon = list(fr["recon"])
    cams = {k: v for k, v in np.load(B / "cameras.npz").items()}; sp = {k: v for k, v in np.load(B / "smplx2020.npz").items()}
    if a.views == 1: refs = [src]
    else:
        pool = [src] + [k for k in recon if k != src]
        refs = [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, a.views).round().astype(int))]
    D = {"sp": sp, "cams": cams, "refs": refs, "ck": {k: i for i, k in enumerate(cams["keys"].tolist())},
         "sk": {k: i for i, k in enumerate(sp["keys"].tolist())},
         "src": {q: (rd(B / "images" / f"{q}.png"), rd(B / "masks" / f"{q}.png", cv2.IMREAD_GRAYSCALE)) for q in refs},
         "betas200_t": torch.from_numpy(sp["betas"][None]).float().to(dev)}
    torch.cuda.reset_peak_memory_stats()
    # infer_single_view reads the render camera only for an unused render size; one 420 x 700 crop camera, built untimed
    _I4 = torch.eye(4); _I4[0, 2], _I4[1, 2] = a.render_size / 2, a.render_size * 5 / 6
    D["c2w1"] = torch.eye(4)[None, None].to(dev); D["intr1"] = _I4[None, None].to(dev); D["bg1"] = torch.ones(1, 1, 3).to(dev)
    D["build"] = None

    # ---------------- reconstruction ----------------
    Pr = Phases()
    for n in range(a.recon_warmup + a.recon_reps):
        D["build"] = None                                   # free the previous avatar before the next build (untimed)
        Pr.start_run(record=n >= a.recon_warmup)
        D["build"] = build(D, Pr)
        Pr.end_run()
    peak_build = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    rs = Pr.summary()

    # ---------------- animation ----------------
    torch.cuda.reset_peak_memory_stats()
    Pa = Phases(); ST.update(P=Pa, on=True)
    bt = D["build"]
    for n in range(a.warmup + a.iters):
        k = test[n % len(test)]
        Pa.start_run(record=n >= a.warmup)
        with Pa("preprocess"):
            kw = frame_inputs(D, k, bt["betas_t"], bt["tmat"])
        rgb, al = model.animation_infer(**kw)
        img = composite(rgb, al)
        sync(); Pa.add("rendering", (time.perf_counter() - ST["t_def_end"]) * 1000)
        Pa.end_run()
    ST["on"] = False
    peak_anim = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    an = Pa.summary()
    peak_all = (max(peak_all[0], peak_build[0], peak_anim[0]), max(peak_all[1], peak_build[1], peak_anim[1]))

    # ---------------- optional sanity renders (after timing; not timed) ----------------
    sanity = ""
    if a.save_png > 0:
        PD = OUTDIR / "lhmpp_png" / ds / clip; PD.mkdir(parents=True, exist_ok=True)
        ps = []
        for k in test[: a.save_png]:
            img = composite(*model.animation_infer(**frame_inputs(D, k, bt["betas_t"], bt["tmat"])))
            cv2.imwrite(str(PD / f"{k}.png"), img[..., ::-1])
            if a.compare_eval:
                ref = Path(a.compare_eval) / c["E"] / "renders" / "lhmpp" / ds / clip / f"{k}.png"
                if ref.exists():
                    rr = rd(ref).astype(np.float64); mse = np.mean((rr - img.astype(np.float64)) ** 2)
                    ps.append((k, round(float(10 * np.log10(255 ** 2 / max(mse, 1e-10))), 2), int(np.abs(rr - img).max())))
        cv2.imwrite(str(PD / "source_input.png"), bt["src_img"][..., ::-1])
        sanity = f" sanity PNGs {PD}"
        if ps: sanity += "; PSNR / max abs diff vs evaluated render: " + ", ".join(f"{k} {p} dB / {m}" for k, p, m in ps)
        print(sanity.strip(), flush=True)

    gb = lambda x: round(x / 2 ** 30, 2)  # noqa: E731
    clips_out[cid] = {
        "reconstruction": {"preprocess": rs.get("preprocess"), "segmentation": None, "pose": None, "network": rs.get("network"),
                           "training": None, "total": rs.get("total")},
        "animation": {"pose": None, "segmentation": None, "preprocess": an.get("preprocess"), "deformation": an.get("deformation"),
                      "rendering": an.get("rendering"), "rendering_raster": an.get("rendering_raster"),
                      "rendering_refiner": an.get("rendering_refiner"), "update": None, "total": an.get("total")},
        "notes": (f"E={c['E']}; source {src}, refs {refs}; {len(test)} test keys cycled ({test[0]}..{test[-1]}); render crop "
                  f"{kw['mask_seqs'][0].shape[1]}x{kw['mask_seqs'][0].shape[0]} (w x h); betas fit residual mean/max "
                  f"{bt['shape_fit_mm'][0]}/{bt['shape_fit_mm'][1]} mm; peak GPU memory allocated/reserved: build "
                  f"{gb(peak_build[0])}/{gb(peak_build[1])} GB, animation {gb(peak_anim[0])}/{gb(peak_anim[1])} GB.{sanity}")}
    r_, n_ = rs.get("total") or {}, an.get("total") or {}
    print(f"{cid}: build total {r_.get('median_ms')} ms (network {(rs.get('network') or {}).get('median_ms')}), frame total "
          f"{n_.get('median_ms')} ms (pre {(an.get('preprocess') or {}).get('median_ms')}, def {(an.get('deformation') or {}).get('median_ms')}, "
          f"render {(an.get('rendering') or {}).get('median_ms')} = raster {(an.get('rendering_raster') or {}).get('median_ms')} + refiner "
          f"{(an.get('rendering_refiner') or {}).get('median_ms')} + paste/composite); peak alloc build {gb(peak_build[0])} GB / anim "
          f"{gb(peak_anim[0])} GB", flush=True)
    D = None; bt = None; kw = None
    write(OUT, method="lhmpp", row="LHM++", clips=clips_out,
          notes=NOTES + f" Peak GPU memory over the run (torch, allocated/reserved): {gb(peak_all[0])}/{gb(peak_all[1])} GB.",
          protocol=protocol, power_note=a.power_note)
print("LHMPP_TIMING_DONE", flush=True)
