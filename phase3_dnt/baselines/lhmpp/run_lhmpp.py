"""LHM++ (aigc3d/LHM-plusplus, LHMPP-700M) as a feed-forward baseline under the AvatarTest-v1 contract
(phase3_dnt/baselines/CONTRACT.md): avatar from the clip's source frame only (--views 1) or from N recon frames (--views N,
uniformly spaced over the recon list, source first), driven by the GT SMPL-X 2020 params of each test frame, rendered with the
512 crop camera (K[key], identity extrinsics), person on black, test keys only. No per-person optimisation, no test-time fitting.

Conversions (no fitting to images):
- shape: LHM++ uses the same SMPL-X 2020 model file as the package (template, joint regressor, skinning, first 10 shape dirs
  identical); its 10 betas are the least-squares fit of the package's 200-beta shape displacement (residual logged).
- expression: LHM++ replaces the SMPL-X face expression basis by FLAME's 100 (on the SMPL-X face vertices); the package's 50
  SMPL-X 2020 expression coefficients are mapped by a fixed least-squares linear map between the two bases (residual logged).
  Jaw / eyes / hands / body / root / transl are copied (both flat_hand_mean = True).
- source image: 512 crop x StyleMatte alpha (> 0.5) on white, cropped to the alpha box, padded / scaled to 840 x 504 with the
  app's PadRatioWithScale(5/3, [840], val=True) (the format of the repo's example inputs). LHM++'s own pose estimator (shape)
  and rembg are not used: shape is the GT's, matte is the package's (the same inputs every GT row gets).
- render: LHM++'s own evaluation path (scripts/inference/infer_eval_cano.py / runners/infer/utils._process_mask_frame_for_eval):
  a crop symmetric about the image centre covering 1.2 x the box of the projected GT SMPL-X vertices (no GT mask used), rendered
  at its native 420-px width, 5:3, then pasted into the 512 board. Output = the neural renderer's RGB composited onto black with
  its own predicted alpha (see --composite).
Run with the lhmpp env from any cwd (the script chdirs into the LHM++ checkout, which uses relative asset paths).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

# The lhmpp env's numpy 1.23 (bundled OpenBLAS64) auto-selects a kernel that returns WRONG matrix products / lstsq on the
# Xeon 8480+ (measured 2026-11-06: A @ X off by ~75 vs torch, lstsq off by 1e5); forcing the SkylakeX kernel is exact.
os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"
import numpy as np

LHM = Path("/home/ubx858/LHM-plusplus")
MODEL = "/raid/ubx858/LHM-plusplus_data/pretrained_models/huggingface/models--3DAIGC--LHMPP-700M/snapshots/fc9f73664b9bfcc457e96210ddc06fe7caf0d559"
SMPLX2020 = "/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True, help="eval root with base/<ds>/<clip>/ (package) and renders/")
ap.add_argument("--split", required=True, help="split JSON (clip list)")
ap.add_argument("--name", required=True, help="method name -> E/renders/<name>/<ds>/<clip>/<key>.png")
ap.add_argument("--views", type=int, default=1, help="1 = source frame only; N > 1 = N recon frames (source first)")
ap.add_argument("--clips", default="", help="comma list of clip ids (ds/clip) to run; default all of the split")
ap.add_argument("--max-test", type=int, default=0, help="smoke only: render at most this many test keys per clip")
ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--render-size", type=int, default=420, help="LHM++ config render_image.high")
ap.add_argument("--composite", default="unpremul_white", choices=["unpremul_white", "mul_alpha"],
                help="unpremul_white: out = rgb - (1 - a) (exact if the renderer composites over white, its training bg); mul_alpha: out = rgb * a")
ap.add_argument("--debug-dir", default="", help="smoke only: also save raw rgb / alpha boards here")
a = ap.parse_args()
E = Path(a.E); OUTROOT = E / "renders" / a.name

os.chdir(LHM); sys.path.insert(0, str(LHM))
import cv2, torch, smplx  # noqa: E402
torch._dynamo.config.disable = True
from core.datasets.data_utils import SrcImagePipeline  # noqa: E402
from core.models import model_dict  # noqa: E402
from core.utils.hf_hub import wrap_model_hub  # noqa: E402
from core.runners.infer.utils import _process_mask_frame_for_eval  # noqa: E402

dev = "cuda"
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, "numpy BLAS wrong (OPENBLAS_CORETYPE)"
assert np.abs(np.linalg.lstsq(_A, _A @ _X, rcond=None)[0] - _X).max() < 1e-9, "numpy lstsq wrong (OPENBLAS_CORETYPE)"
COMMIT = subprocess.run(["git", "-C", str(LHM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
GUAVA_SHA = subprocess.run(["git", "-C", "/home/ubx858/guava", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
GPU = torch.cuda.get_device_name(0)

# ---------------- fixed linear conversions (shape 200 -> 10, expression SMPL-X 2020 50 -> FLAME 100) ----------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)                  # (10475, 3, 400)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
lhm_sx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
assert np.abs(lhm_sx["v_template"] - Z["v_template"]).max() == 0 and np.abs(lhm_sx["shapedirs"][:, :, :10] - Z["shapedirs"][:, :, :10]).max() == 0
face_idx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
flame = smplx.create(str(LHM / "pretrained_models/human_model_files"), "flame", gender="neutral", num_betas=10, num_expression_coeffs=100)
EF = flame.expr_dirs.detach().cpu().numpy().astype(np.float64).reshape(-1, 100)                    # (5023*3, 100)
ES = SD[face_idx, :, 300:350].reshape(-1, 50)                                                      # (5023*3, 50)
MEXP, *_ = np.linalg.lstsq(EF, ES, rcond=None)                                                     # (100, 50): e100 = MEXP @ e50
body2020 = smplx.SMPLX(SMPLX2020, num_betas=200, num_expression_coeffs=50, use_pca=False, flat_hand_mean=True).to(dev)


def betas10(b200):
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    r = (S10 @ b - S200 @ b200).reshape(-1, 3); n = np.linalg.norm(r, axis=1) * 1000
    return b.astype(np.float32), float(n.mean()), float(n.max())


def expr100(e50):
    e = (MEXP @ e50.T.astype(np.float64)).T                                                         # (T, 100)
    r = (EF @ e.T - ES @ e50.T.astype(np.float64)).T.reshape(len(e50), -1, 3); n = np.linalg.norm(r, axis=2) * 1000
    return e.astype(np.float32), float(n.mean()) if n.size else 0.0, float(n.max()) if n.size else 0.0


# ---------------- model ----------------
from accelerate import Accelerator  # noqa: E402
Accelerator()                       # LHM++ logs through accelerate (the app scripts create one too)
t0 = time.time()
model = wrap_model_hub(model_dict["human_lrm_a4o"]).from_pretrained(MODEL); model.to(dev); model.eval()   # its .to / .eval do not return self
print(f"model loaded in {time.time() - t0:.1f} s ({COMMIT[:7]}, {GPU})", flush=True)
pipe = SrcImagePipeline(dict(name="PadRatioWithScale", target_ratio=5 / 3, tgt_max_size_list=[840], val=True))


def prep_source(img, alpha):
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


split = json.load(open(a.split)); clips = split["clips"]
want = set(c for c in a.clips.split(",") if c)
todo = [c for c in clips if not want or c["id"] in want]
print(f"{len(todo)} clips -> {OUTROOT}", flush=True)

for c in todo:
    ds, clip = c["dataset"], c["clip"]; B = E / "base" / ds / clip; OUT = OUTROOT / ds / clip
    fr = json.load(open(B / "frames.json")); test = list(fr["test"])[: a.max_test] if a.max_test else list(fr["test"])
    if OUT.exists() and all((OUT / f"{k}.png").exists() for k in test) and (OUT / "run.json").exists():
        print(f"skip {c['id']} (complete)", flush=True); continue
    cams = np.load(B / "cameras.npz"); sp = np.load(B / "smplx2020.npz")
    ck = {k: i for i, k in enumerate(cams["keys"].tolist())}; sk = {k: i for i, k in enumerate(sp["keys"].tolist())}
    src = fr["source"]; recon = list(fr["recon"])
    if a.views == 1: refs = [src]
    else:
        pool = [src] + [k for k in recon if k != src]
        refs = [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, a.views).round().astype(int))]
    t_build = time.time()
    ref_imgs = np.stack([prep_source(rd(B / "images" / f"{k}.png"), rd(B / "masks" / f"{k}.png", cv2.IMREAD_GRAYSCALE)) for k in refs])
    b10, sh_mean, sh_max = betas10(sp["betas"])
    ti = [sk[k] for k in test]; T = len(ti)
    e100, ex_mean, ex_max = expr100(sp["expression"][ti])
    # projected GT SMPL-X box per test frame (render crop only; no GT mask)
    with torch.no_grad():
        g = lambda n: torch.from_numpy(sp[n][ti].reshape(T, -1)).float().to(dev)
        V = body2020(betas=torch.from_numpy(sp["betas"][None]).float().to(dev).expand(T, -1), global_orient=g("global_orient"), body_pose=g("body_pose"),
                     jaw_pose=g("jaw_pose"), leye_pose=g("leye_pose"), reye_pose=g("reye_pose"), left_hand_pose=g("left_hand_pose"),
                     right_hand_pose=g("right_hand_pose"), expression=g("expression"), transl=g("transl")).vertices.cpu().numpy()
    intrs, masks, offs, P = [], [], [], {k: [] for k in ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose", "trans", "expr", "focal", "princpt", "img_size_wh")}
    for j, (k, i) in enumerate(zip(test, ti)):
        K = cams["K"][ck[k]].astype(np.float64)
        uv = (V[j] @ K.T)[:, :2] / (V[j] @ K.T)[:, 2:3]
        x0, y0 = np.clip(np.floor(uv.min(0)), 0, 511).astype(int); x1, y1 = np.clip(np.ceil(uv.max(0)), 0, 511).astype(int)
        boxm = np.zeros((512, 512), np.uint8); boxm[y0: y1 + 1, x0: x1 + 1] = 255
        I4 = torch.eye(4); I4[0, 0], I4[1, 1], I4[0, 2], I4[1, 2] = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        intr, cm, off, _ = _process_mask_frame_for_eval(boxm, None, 0.0, I4, a.render_size, 5 / 3.0, 14, res_scale=1.0,
                                                        resize_to_1024=False, mask_already_padded=False, rendering_with_pad=False)
        intrs.append(intr); masks.append(cm); offs.append(off)
        P["root_pose"].append(sp["global_orient"][i]); P["body_pose"].append(sp["body_pose"][i]); P["jaw_pose"].append(sp["jaw_pose"][i])
        P["leye_pose"].append(sp["leye_pose"][i]); P["reye_pose"].append(sp["reye_pose"][i]); P["lhand_pose"].append(sp["left_hand_pose"][i])
        P["rhand_pose"].append(sp["right_hand_pose"][i]); P["trans"].append(sp["transl"][i]); P["expr"].append(e100[j])
        P["focal"].append(np.array([K[0, 0], K[1, 1]], np.float32)); P["princpt"].append(np.array([K[0, 2], K[1, 2]], np.float32)); P["img_size_wh"].append(np.array([512, 512], np.float32))
    smp = {k: torch.from_numpy(np.stack(v).astype(np.float32))[None] for k, v in P.items()}
    smp["betas"] = torch.from_numpy(b10)[None]
    c2ws = torch.eye(4)[None, None].repeat(1, T, 1, 1); intrs = torch.stack(intrs)[None]; bgs = torch.ones(1, T, 3)
    with torch.no_grad():
        refs_t = torch.from_numpy(ref_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)[None].to(dev)
        outs = model.infer_single_view(refs_t, None, None, render_c2ws=c2ws.to(dev), render_intrs=intrs.to(dev), render_bg_colors=bgs.to(dev),
                                       smplx_params={k: v.to(dev) for k, v in smp.items()}, ref_imgs_bool=torch.ones(1, len(refs), dtype=torch.bool, device=dev))
        gs_list, qp, tmat, gs_feat, img_lat, mot_emb = outs[:6]; pos_emb = outs[6] if len(outs) == 7 else None
        torch.cuda.synchronize(); build_s = time.time() - t_build
        OUT.mkdir(parents=True, exist_ok=True); t_r = time.time(); n_done = 0; bg_med = []
        for s in range(0, T, a.batch):
            bp = {"betas": smp["betas"].to(dev), "transform_mat_neutral_pose": tmat}
            bp.update({k: smp[k][:, s: s + a.batch].to(dev) for k in P})
            kw = dict(gs_model_list=gs_list, query_points=qp, smplx_params=bp, render_c2ws=c2ws[:, s: s + a.batch].to(dev),
                      render_intrs=intrs[:, s: s + a.batch].to(dev), render_bg_colors=bgs[:, s: s + a.batch].to(dev), gs_hidden_features=gs_feat,
                      image_latents=img_lat, motion_emb=mot_emb, offset_list=offs[s: s + a.batch], mask_seqs=masks[s: s + a.batch],
                      output_rgb=torch.zeros(512, 512, 3))
            if pos_emb is not None: kw["pos_emb"] = pos_emb
            rgb, al = model.animation_infer(**kw)
            rgb = rgb.clamp(0, 1).numpy(); al = al.clamp(0, 1).numpy()
            for j in range(rgb.shape[0]):
                k = test[s + j]; r, m = rgb[j], al[j][..., None]
                inside = np.zeros((512, 512), bool); sx, sy, ox, oy = offs[s + j]; hh, ww = masks[s + j].shape
                inside[oy: oy + int(hh / sy), ox: ox + int(ww / sx)] = True
                bgpx = r[inside & (al[j] < 0.02)]
                if len(bgpx): bg_med.append(float(np.median(bgpx)))
                o = np.clip(r - (1 - m), 0, 1) * (m > 0) if a.composite == "unpremul_white" else r * m
                cv2.imwrite(str(OUT / f"{k}.png"), (o * 255 + 0.5).astype(np.uint8)[..., ::-1])
                if a.debug_dir:
                    D = Path(a.debug_dir) / ds / clip; D.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(D / f"{k}_rgb.png"), (r * 255 + 0.5).astype(np.uint8)[..., ::-1]); cv2.imwrite(str(D / f"{k}_alpha.png"), (al[j] * 255 + 0.5).astype(np.uint8))
                n_done += 1
        render_s = (time.time() - t_r) / max(n_done, 1)
    if a.debug_dir:
        D = Path(a.debug_dir) / ds / clip; D.mkdir(parents=True, exist_ok=True)
        for q, im in zip(refs, ref_imgs): cv2.imwrite(str(D / f"src_{q}.png"), im[..., ::-1])
    run = {"method": a.name, "model": "LHMPP-700M", "commit": COMMIT, "weights": MODEL, "guava_commit": GUAVA_SHA, "train_s": round(build_s, 2),
           "render_s_per_frame": round(render_s, 4), "gpu": GPU, "iterations": 0, "views": len(refs), "ref_keys": refs, "n_test": n_done,
           "shape_fit_mm": {"mean": round(sh_mean, 3), "max": round(sh_max, 3)}, "expr_map_mm": {"mean": round(ex_mean, 3), "max": round(ex_max, 3)},
           "bg_median_in_crop": round(float(np.median(bg_med)), 4) if bg_med else None, "composite": a.composite,
           "notes": "feed-forward, no per-person optimisation; GT SMPL-X driving; see phase3_dnt/baselines/lhmpp/NOTES.md",
           "loader_changes": "package adapter (run_lhmpp.py); LHM++ code unchanged"}
    json.dump(run, open(OUT / "run.json", "w"), indent=1)
    print(f"done {c['id']}: views {len(refs)}, {n_done} frames, build {build_s:.1f} s, {render_s * 1000:.0f} ms/frame, shape fit {sh_mean:.2f} mm, "
          f"expr map {ex_mean:.2f} mm, bg median {run['bg_median_in_crop']}", flush=True)
print("LHMPP_DONE", flush=True)
