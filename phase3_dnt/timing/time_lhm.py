"""LHM (aigc3d/LHM @ 4f88aae, LHM-1B-HF) timing for the laptop kit: reconstruction (source photo -> Gaussian avatar, once per
person) and animation (one GT SMPL-X driving frame -> 512 x 512 render), batch 1, the evaluated setting of
phase3_dnt/baselines/lhm/run_lhm.py with --clamp-face --face-on-matte (reported row `lhm_facefix`). Output schema: phase_timer.py.

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

Usage (inside the env, see run_lhm.sh):  python time_lhm.py --root SUBSET_ROOT --out OUT/lhm.json [--clips ubody/ConductMusic_S15,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _blas_core():
    if os.environ.get("OPENBLAS_CORETYPE"): return
    try: flags = Path("/proc/cpuinfo").read_text()
    except OSError: flags = ""
    os.environ["OPENBLAS_CORETYPE"] = "SkylakeX" if " avx512f" in flags else "Haswell"


_blas_core()                                            # before numpy is imported (phase_timer imports numpy)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import phase_timer as PT  # noqa: E402

LHM_DEF = os.environ.get("LHM_CODE", "/dev/shm/ubx858_baselines/lhm/src")                   # aigc3d/LHM @ 4f88aae (unchanged)
MODEL_DEF = os.environ.get("LHM_MODEL", "/dev/shm/ubx858_baselines/weights/lhm_1b_hf")       # 3DAIGC/LHM-1B-HF @ 92372582
MODEL_REV = "3DAIGC/LHM-1B-HF@92372582f660066b9f1b9513860744357265b3d5"
SMPLX2020_DEF = str(HERE.parent.parent / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="/dev/shm/ubx858_timing/timing_subset_v1", help="timing subset root (manifest.json)")
ap.add_argument("--clips", default="", help="comma list of clip ids (ds/clip); default all clips in the manifest")
ap.add_argument("--out", required=True, help="output JSON path")
ap.add_argument("--warmup", type=int, default=PT.PROTOCOL["warmup"], help="animation warm-up frames (not recorded)")
ap.add_argument("--iters", type=int, default=PT.PROTOCOL["iters"], help="animation timed frames")
ap.add_argument("--recon-warmup", type=int, default=PT.PROTOCOL["recon_warmup"])
ap.add_argument("--recon-reps", type=int, default=PT.PROTOCOL["recon_reps"])
ap.add_argument("--lhm-src", default=LHM_DEF)
ap.add_argument("--model", default=MODEL_DEF, help="LHM-1B-HF weights directory")
ap.add_argument("--smplx2020", default=SMPLX2020_DEF)
ap.add_argument("--no-facefix", action="store_true", help="time the unfixed face crop (row `lhm`) instead of `lhm_facefix`")
ap.add_argument("--save-png", type=int, default=2, help="after timing, write this many renders per clip under <out dir>/lhm_png")
ap.add_argument("--check-eval", default="", help="optional (read only): eval root holding <E>/renders/<name>/<ds>/<clip>/<key>.png;"
                                                 " the saved PNGs are compared with the evaluated renders")
ap.add_argument("--check-name", default="", help="render name under --check-eval (default lhm_facefix, or lhm with --no-facefix)")
ap.add_argument("--power-note", default="", help="e.g. 'AC power, performance mode' (also TIMING_POWER_NOTE)")
a = ap.parse_args()
ROOT = Path(a.root).resolve(); OUTJ = Path(a.out).resolve(); PNG = OUTJ.parent / "lhm_png"
FACEFIX = not a.no_facefix; CHECK_NAME = a.check_name or ("lhm_facefix" if FACEFIX else "lhm")
SMPLX2020 = Path(a.smplx2020).resolve(); LHM = Path(a.lhm_src).resolve()
if not SMPLX2020.exists(): sys.exit(f"missing {SMPLX2020} (SMPL-X 2020 neutral, git-ignored): copy it there or pass --smplx2020")

os.chdir(LHM); sys.path.insert(0, str(LHM))                        # LHM loads its priors by relative paths
import cv2  # noqa: E402
import smplx  # noqa: E402
import torch  # noqa: E402
torch._dynamo.config.disable = True                                 # as evaluated (run_lhm.py): no torch.compile
from engine.SegmentAPI.base import Bbox  # noqa: E402
from LHM.models import model_dict  # noqa: E402
from LHM.utils.hf_hub import wrap_model_hub  # noqa: E402
from LHM.utils.face_detector import FaceDetector  # noqa: E402
from LHM.runners.infer.utils import calc_new_tgt_size_by_aspect, center_crop_according_to_mask, resize_image_keepaspect_np  # noqa: E402

dev = "cuda"
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, "numpy BLAS wrong (OPENBLAS_CORETYPE)"
assert np.abs(np.linalg.lstsq(_A, _A @ _X, rcond=None)[0] - _X).max() < 1e-9, "numpy lstsq wrong (OPENBLAS_CORETYPE)"

# ---------------- fixed linear conversions, as run_lhm.py (shape 200 -> 10, expression SMPL-X 2020 50 -> FLAME 100) ----------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
lhm_sx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
assert np.abs(lhm_sx["v_template"] - Z["v_template"]).max() == 0 and np.abs(lhm_sx["shapedirs"][:, :, :10] - Z["shapedirs"][:, :, :10]).max() == 0
face_idx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
flame = smplx.create(str(LHM / "pretrained_models/human_model_files"), "flame", gender="neutral", num_betas=10, num_expression_coeffs=100)
EF = flame.expr_dirs.detach().cpu().numpy().astype(np.float64).reshape(-1, 100)
ES = SD[face_idx, :, 300:350].reshape(-1, 50)
MEXP, *_ = np.linalg.lstsq(EF, ES, rcond=None)


def betas10(b200):                                                  # run_lhm.betas10 without the logged residual
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    return b.astype(np.float32)


def expr100(e50):                                                   # run_lhm.expr100 without the logged residual
    return (MEXP @ e50.T.astype(np.float64)).T.astype(np.float32)


# ---------------- LHM's own source preprocessing (copied from run_lhm.py = human_lrm.py minus `assert w < h`) ----------------
def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0; pha[pha < 0.5] = 0.0; pha[pha >= 0.5] = 1.0
    _h, _w = np.where(pha == 1)
    return Bbox([_w.min().item(), _h.min().item(), _w.max().item(), _h.max().item()]).scale(1.1, width=width, height=height)


def infer_preprocess_image(rgb, mask, bg_color=1.0, max_tgt_size=896, aspect_standard=5.0 / 3, enlarge_ratio=(1.0, 1.0), render_tgt_size=1024, multiply=14):
    bb = get_bbox(mask).get_box()
    rgb = rgb[bb[1]: bb[3], bb[0]: bb[2]]; mask = mask[bb[1]: bb[3], bb[0]: bb[2]]
    h, w, _ = rgb.shape
    scale_ratio = (h / w) / aspect_standard
    target_w = int(min(w * scale_ratio, h))
    if target_w - w > 0:
        ow = (target_w - w) // 2
        rgb = np.pad(rgb, ((0, 0), (ow, ow), (0, 0)), mode="constant", constant_values=255); mask = np.pad(mask, ((0, 0), (ow, ow)), mode="constant", constant_values=0)
    else:
        oh = int(w * aspect_standard - h)
        rgb = np.pad(rgb, ((oh, 0), (0, 0), (0, 0)), mode="constant", constant_values=255); mask = np.pad(mask, ((oh, 0), (0, 0)), mode="constant", constant_values=0)
    rgb = rgb / 255.0; mask = (mask / 255.0 > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size); mask = resize_image_keepaspect_np(mask, max_tgt_size)
    rgb, mask, _, _ = center_crop_according_to_mask(rgb, mask, aspect_standard, list(enlarge_ratio))
    tgt_hw, _, _ = calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], aspect_standard=aspect_standard, tgt_size=render_tgt_size, multiply=multiply)
    rgb = cv2.resize(rgb, dsize=(tgt_hw[1], tgt_hw[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)          # [1, 3, H, W]


def rd(p, flag=cv2.IMREAD_COLOR):
    x = cv2.imread(str(p), flag)
    if x is None: raise FileNotFoundError(p)
    return x[..., ::-1].copy() if flag == cv2.IMREAD_COLOR else x


# ---------------- model (as run_lhm.py: fp32 weights; Sapiens runs under its own bf16 autocast inside LHM) ----------------
from accelerate import Accelerator  # noqa: E402
Accelerator()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"]).from_pretrained(a.model); model.to(dev); model.eval(); model.to(torch.float32)
facedetect = FaceDetector("./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd", device=dev)
torch.cuda.synchronize(); LOAD_S = time.time() - t0; LOAD_PEAK = torch.cuda.max_memory_allocated() / 2**30
print(f"model loaded in {LOAD_S:.1f} s, {LOAD_PEAK:.2f} GB allocated ({MODEL_REV}, {torch.cuda.get_device_name(0)})", flush=True)

# ---------------- runtime wrappers at the official function boundaries (timing only; no behaviour change) ----------------
CUR = {"P": None}


def wrap(obj, attr, name):
    orig = getattr(obj, attr)

    def f(*args, **kw):
        P = CUR["P"]
        if P is None: return orig(*args, **kw)
        with P(name): return orig(*args, **kw)
    setattr(obj, attr, f)


R = model.renderer
wrap(R, "animate_gs_model", "deformation")              # GS3DRenderer.animate_gs_model: SMPL-X voxel-skinning LBS + rotations
wrap(R, "forward_single_batch", "rendering")            # GS3DRenderer.forward_single_batch: Camera.from_c2w + 3DGS rasteriser
wrap(model, "obtain_facesr", "net:face_sr")             # Real-ESRGAN x4 + GFPGAN v1.3 on the 112 head crop
wrap(R, "get_query_points", "net:query_points")         # SMPL-X dense query points in the neutral pose (betas)
wrap(model, "forward_latent_points", "net:encoders_transformer")   # Sapiens-1B + DINOv2 encoders, SD3 MM transformer
wrap(R, "forward_gs", "net:gs_decode")                  # per-point Gaussian attribute decoder


def recon_once(P, B, src, betas200, k0, p0):
    """One avatar build, as timed in run_lhm.py (t_build): read source, LHM preprocessing, head crop, betas fit, infer_single_view."""
    with P("preprocess"):
        img = rd(B / "images" / f"{src}.png"); alpha = rd(B / "masks" / f"{src}.png", cv2.IMREAD_GRAYSCALE)
        image = infer_preprocess_image(img, alpha)
        try:
            fimg = (img * (alpha[..., None] > 127) + 255 * (alpha[..., None] <= 127)).astype(np.uint8) if FACEFIX else img
            box = [int(v) for v in facedetect(torch.from_numpy(fimg).permute(2, 0, 1))]
            y0, x0 = (max(0, box[1]), max(0, box[0])) if FACEFIX else (box[1], box[0])
            head = cv2.resize(fimg[y0: box[3], x0: box[2]], dsize=(112, 112), interpolation=cv2.INTER_AREA); found = True
        except Exception:                                                                      # noqa: BLE001  (LHM's own fallback)
            head = np.zeros((112, 112, 3), np.uint8); found = False
        head_t = torch.from_numpy(head / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        b10 = torch.from_numpy(betas10(betas200))[None].to(dev)
        smp = frame_params(k0, p0); smp["betas"] = b10
        image_d, head_d = image.unsqueeze(0).to(dev), head_t.unsqueeze(0).to(dev)
    with P("network"), torch.no_grad():
        gs_list, qp, tmat = model.infer_single_view(image_d, head_d, None, None, render_c2ws=smp["c2ws"], render_intrs=smp["intrs"],
                                                    render_bg_colors=smp["bgs"], smplx_params={k: v for k, v in smp.items() if k not in CAM})
    return gs_list, qp, tmat, b10, found


CAM = ("c2ws", "intrs", "bgs")


def frame_params(key, p):
    """GT SMPL-X 2020 params + K of one frame -> LHM inputs [1, 1, ...] on the GPU (run_lhm.py's per-frame conversion)."""
    i, K = p["sk"][key], p["K"][p["ck"][key]].astype(np.float64)
    I4 = torch.eye(4); I4[0, 0], I4[1, 1], I4[0, 2], I4[1, 2] = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    sp = p["sp"]; f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
    d = {"root_pose": f32(sp["global_orient"][i]).view(1, 1, 3), "body_pose": f32(sp["body_pose"][i]).view(1, 1, 21, 3),
         "jaw_pose": f32(sp["jaw_pose"][i]).view(1, 1, 3), "leye_pose": f32(sp["leye_pose"][i]).view(1, 1, 3),
         "reye_pose": f32(sp["reye_pose"][i]).view(1, 1, 3), "lhand_pose": f32(sp["left_hand_pose"][i]).view(1, 1, 15, 3),
         "rhand_pose": f32(sp["right_hand_pose"][i]).view(1, 1, 15, 3), "trans": f32(sp["transl"][i]).view(1, 1, 3),
         "expr": f32(expr100(sp["expression"][i: i + 1])).view(1, 1, 100),
         "focal": torch.tensor([[[K[0, 0], K[1, 1]]]], dtype=torch.float32), "princpt": torch.tensor([[[K[0, 2], K[1, 2]]]], dtype=torch.float32),
         "img_size_wh": torch.tensor([[[512.0, 512.0]]]), "c2ws": torch.eye(4)[None, None], "intrs": I4[None, None], "bgs": torch.zeros(1, 1, 3)}
    return {k: v.to(dev) for k, v in d.items()}


def anim_once(P, key, p, gs_list, qp, tmat, b10):
    """One driving frame, batch 1: params -> model.animation_infer (deformation + rendering) -> 512 RGB on the GPU."""
    with P("preprocess"):
        f = frame_params(key, p)
        bp = {k: v for k, v in f.items() if k not in CAM}; bp["betas"] = b10; bp["transform_mat_neutral_pose"] = tmat
    with torch.no_grad():
        res = model.animation_infer(gs_list, qp, bp, render_c2ws=f["c2ws"], render_intrs=f["intrs"], render_bg_colors=f["bgs"])
        return res["comp_rgb"].clamp(0, 1)                                                     # [1, 512, 512, 3]


def to_u8(rgb):
    return (rgb[0].float().cpu().numpy() * 255 + 0.5).astype(np.uint8)                       # run_lhm.py's PNG quantisation


man = json.loads((ROOT / "manifest.json").read_text())
want = [c for c in a.clips.split(",") if c]
todo = [c for c in man["clips"] if not want or c["id"] in want]
assert todo and (not want or len(todo) == len(want)), f"clips {want} not all in the manifest"
clips_out = {}; peak_all = LOAD_PEAK; reserved_all = torch.cuda.max_memory_reserved() / 2**30
for c in todo:
    cid, ds, name = c["id"], c["dataset"], c["clip"]; B = ROOT / c["E"] / "base" / ds / name
    fr = json.loads((B / "frames.json").read_text()); test = list(fr["test"]); src = fr["source"]
    cams = np.load(B / "cameras.npz"); spz = np.load(B / "smplx2020.npz")
    p = {"K": cams["K"], "ck": {k: i for i, k in enumerate(cams["keys"].tolist())},
         "sp": {k: spz[k] for k in spz.files}, "sk": {k: i for i, k in enumerate(spz["keys"].tolist())}}
    torch.cuda.reset_peak_memory_stats()

    # reconstruction: recon_warmup + recon_reps builds
    PR = PT.Phases(); CUR["P"] = PR; built = None
    for r in range(a.recon_warmup + a.recon_reps):
        built = None; PR.start_run(record=r >= a.recon_warmup)
        built = recon_once(PR, B, src, p["sp"]["betas"], test[0], p)
        PR.end_run()
    gs_list, qp, tmat, b10, face_found = built
    rs = PR.summary(); sub = {k[4:]: rs.pop(k) for k in list(rs) if k.startswith("net:")}
    recon = {"preprocess": rs.get("preprocess"), "segmentation": None, "pose": None, "network": rs.get("network"),
             "training": None, "total": rs.get("total")}

    # animation: warmup + iters frames, batch 1, cycling over the clip's test keys
    PA = PT.Phases(); CUR["P"] = PA
    for it in range(a.warmup + a.iters):
        PA.start_run(record=it >= a.warmup)
        out = anim_once(PA, test[it % len(test)], p, gs_list, qp, tmat, b10)
        PA.end_run(); del out
    CUR["P"] = None
    asum = PA.summary()
    anim = {"pose": None, "segmentation": None, "preprocess": asum.get("preprocess"), "deformation": asum.get("deformation"),
            "rendering": asum.get("rendering"), "rendering_raster": None, "rendering_refiner": None, "update": None, "total": asum.get("total")}
    peak = torch.cuda.max_memory_allocated() / 2**30; resv = torch.cuda.max_memory_reserved() / 2**30
    peak_all = max(peak_all, peak); reserved_all = max(reserved_all, resv)

    # sanity PNGs (untimed) + optional comparison with the evaluated renders (read only)
    chk = []
    if a.save_png > 0:
        D = PNG / ds / name; D.mkdir(parents=True, exist_ok=True)
        for k in test[: a.save_png]:
            u8 = to_u8(anim_once(PT.Phases(), k, p, gs_list, qp, tmat, b10)); cv2.imwrite(str(D / f"{k}.png"), u8[..., ::-1])
            ref = Path(a.check_eval) / c["E"] / "renders" / CHECK_NAME / ds / name / f"{k}.png" if a.check_eval else None
            if ref is not None and ref.exists():
                g = rd(ref).astype(np.float64); mse = ((u8.astype(np.float64) - g) ** 2).mean()
                chk.append(f"{k}: max|diff| {int(np.abs(u8.astype(np.int16) - g.astype(np.int16)).max())}, "
                           f"PSNR {'inf' if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)} dB vs {CHECK_NAME}")
    med = lambda s: s["median_ms"] if s else None  # noqa: E731
    notes = (f"{qp.shape[1]} Gaussians; face_found {face_found}; network breakdown, median ms: "
             + ", ".join(f"{k} {med(v)}" for k, v in sub.items())
             + f"; peak GPU memory this clip {peak:.2f} GB allocated / {resv:.2f} GB reserved"
             + (f"; check vs evaluated renders: {'; '.join(chk)}" if chk else ""))
    clips_out[cid] = {"reconstruction": recon, "animation": anim, "notes": notes}
    print(f"{cid}: recon total {med(recon['total'])} ms (pre {med(recon['preprocess'])}, net {med(recon['network'])}); "
          f"anim total {med(anim['total'])} ms (pre {med(anim['preprocess'])}, deform {med(anim['deformation'])}, "
          f"render {med(anim['rendering'])}); peak {peak:.2f} GB; {notes}", flush=True)
    del gs_list, qp, tmat, built; torch.cuda.empty_cache()

NOTES = (
    f"LHM-1B-HF ({MODEL_REV}), aigc3d/LHM @ 4f88aae unchanged, adapter as phase3_dnt/baselines/lhm/run_lhm.py"
    f"{' --clamp-face --face-on-matte (reported row lhm_facefix)' if FACEFIX else ' without the face fix (row lhm)'}. "
    "Precision as evaluated: fp32 weights; the Sapiens-1B encoder runs under LHM's own bf16 autocast; torch.compile disabled "
    "(run_lhm.py sets torch._dynamo.config.disable). Batch 1, 512 x 512 render with the frame's K, identity extrinsics, black bg. "
    "Reconstruction (from the clip's source frame): preprocess = read source image + StyleMatte mask PNG, LHM infer_preprocess_image "
    "(alpha box x1.1, pad 5:3, matte on white, resize 896, centre crop by mask, resize to 1024, multiple of 14), VGGHead face "
    "detection (LHM FaceDetector) on the matted source + 112 px head crop, least-squares 200 -> 10 betas, tensors to GPU; "
    "segmentation = null: the package's StyleMatte masks are used instead of LHM's SAM2 / rembg (as in the evaluation); "
    "pose = null: GT SMPL-X given; network = ModelHumanLRMSapdinoBodyHeadSD3_5.infer_single_view (obtain_facesr: Real-ESRGAN x4 + "
    "GFPGAN v1.3 on the head crop, partly on the CPU; GS3DRenderer.get_query_points: dense SMPL-X points; forward_latent_points: "
    "Sapiens-1B + DINOv2 ViT-L encoders and the 15-layer SD3 MM transformer; GS3DRenderer.forward_gs: Gaussian decoder), sub-times "
    "in each clip's notes; total = preprocess + network end to end. Animation per driving frame (cycling over the clip's test "
    "keys): preprocess = GT SMPL-X 2020 params of the frame -> LHM params (expression 50 -> FLAME 100 linear map, K -> 4x4 "
    "intrinsics, identity c2w, black bg), tensors to GPU; deformation = GS3DRenderer.animate_gs_model (SMPL-X voxel-skinning LBS "
    "of the Gaussians to the target pose, together with LHM's A-pose canonical copy, and the Gaussian rotation update); rendering = "
    "GS3DRenderer.forward_single_batch (Camera.from_c2w + diff_gaussian_rasterization, no refiner / upsampler); total = preprocess "
    "+ ModelHumanLRMSapdinoBodyHeadSD3_5.animation_infer + clamp, ending with the 512 RGB image on the GPU (no GPU->CPU copy or PNG "
    "write; the evaluation's run.json render_s_per_frame includes both and ran batches of 20 frames). "
    f"Model load {LOAD_S:.1f} s (not timed). Peak GPU memory over the run: {peak_all:.2f} GB allocated "
    f"(model load {LOAD_PEAK:.2f} GB), {reserved_all:.2f} GB reserved."
)
proto = dict(PT.PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=a.recon_warmup, recon_reps=a.recon_reps,
             precision="fp32 (Sapiens encoder under LHM's own bf16 autocast), as evaluated", render_px=512)
PT.write(OUTJ, method="lhm", row="LHM", clips=clips_out, notes=NOTES, protocol=proto, power_note=a.power_note)
print("LHM_TIMING_DONE", flush=True)
