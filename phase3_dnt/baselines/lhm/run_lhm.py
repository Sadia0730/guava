"""LHM (aigc3d/LHM, ICCV 2025; LHM-1B-HF weights) as a feed-forward single-image baseline under the AvatarTest-v1 contract
(phase3_dnt/baselines/CONTRACT.md): avatar from the clip's source frame only, driven by the GT SMPL-X 2020 params of each test
frame, rendered with the 512 crop camera (K[key], identity extrinsics) on black, test keys only. No per-person optimisation.

Conversions (identical to the LHM++ adapter, phase3_dnt/baselines/lhmpp/run_lhmpp.py; LHM uses the same SMPL-X layer):
- shape: 10 betas = least-squares fit of the package's 200-beta displacement (LHM's SMPLX_NEUTRAL.npz = SMPL-X 2020 template,
  regressor, skinning and first 10 shape dirs; asserted). Residual logged.
- expression: SMPL-X 2020 50 -> FLAME 100 on the SMPL-X face vertices by a fixed least-squares map (exact in practice). Logged.
- jaw / eyes / hands / body / root / transl copied (both flat_hand_mean = True).
- source image: LHM's own inference preprocessing (runners/infer/human_lrm.py infer_preprocess_image: alpha box x 1.1, pad to
  5:3, matte > 0.5 on white, resize to 896, centre crop by mask, resize to source_size 1024, multiple of 14), with the package's
  StyleMatte alpha instead of SAM2 / rembg. The function's `assert w < h` is dropped: 8 UBody source boxes are wider than tall
  and the function's own half-body branch (pad on top to 5:3) handles them unchanged.
- head image: LHM's own VGGHead face detector on the raw source crop, resized to 112 (zeros if no face, as in LHM).
- render: LHM's Gaussian rasteriser directly at 512 x 512 with K (render size = 2 x principal point) and background 0, i.e. the
  premultiplied render on black, which is the contract's output. Its face SR (Real-ESRGAN + GFPGAN v1.3) runs as configured.
Run with the lhmpp env + the overlay in /dev/shm/ubx858_baselines/lhm/pydeps (basicsr, gfpgan, facexlib, GPUtil, imagehash).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"   # lhmpp env numpy/OpenBLAS64 is wrong on the Xeon 8480+ otherwise (see lhmpp NOTES)
import numpy as np

LHM = Path("/dev/shm/ubx858_baselines/lhm/src")
MODEL = "/dev/shm/ubx858_baselines/weights/lhm_1b_hf"          # 3DAIGC/LHM-1B-HF @ 92372582, model.safetensors sha256 59dc2516...
MODEL_REV = "3DAIGC/LHM-1B-HF@92372582f660066b9f1b9513860744357265b3d5"
SMPLX2020 = "/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True, help="eval root with base/<ds>/<clip>/ (package) and renders/")
ap.add_argument("--split", required=True, help="split JSON (clip list)")
ap.add_argument("--name", required=True, help="method name -> E/renders/<name>/<ds>/<clip>/<key>.png")
ap.add_argument("--clips", default="", help="comma list of clip ids (ds/clip) to run; default all of the split")
ap.add_argument("--max-test", type=int, default=0, help="smoke only: render at most this many test keys per clip")
ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--clamp-face", action="store_true", help="clip the face box at the top / left image edge (amendment 1)")
ap.add_argument("--face-on-matte", action="store_true", help="head detection / crop on the matted source (amendment 1)")
ap.add_argument("--debug-dir", default="", help="smoke only: save the preprocessed source / head images here")
a = ap.parse_args()
E = Path(a.E); OUTROOT = E / "renders" / a.name

os.chdir(LHM); sys.path.insert(0, str(LHM))
import cv2, torch, smplx  # noqa: E402
torch._dynamo.config.disable = True
from engine.SegmentAPI.base import Bbox  # noqa: E402
from LHM.models import model_dict  # noqa: E402
from LHM.utils.hf_hub import wrap_model_hub  # noqa: E402
from LHM.utils.face_detector import FaceDetector  # noqa: E402
from LHM.runners.infer.utils import calc_new_tgt_size_by_aspect, center_crop_according_to_mask, resize_image_keepaspect_np  # noqa: E402

dev = "cuda"
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, "numpy BLAS wrong (OPENBLAS_CORETYPE)"
assert np.abs(np.linalg.lstsq(_A, _A @ _X, rcond=None)[0] - _X).max() < 1e-9, "numpy lstsq wrong (OPENBLAS_CORETYPE)"
COMMIT = subprocess.run(["git", "-C", str(LHM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
GUAVA_SHA = subprocess.run(["git", "-C", "/home/ubx858/guava", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
GPU = torch.cuda.get_device_name(0)

# ---------------- fixed linear conversions (shape 200 -> 10, expression SMPL-X 2020 50 -> FLAME 100) ----------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
lhm_sx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
assert np.abs(lhm_sx["v_template"] - Z["v_template"]).max() == 0 and np.abs(lhm_sx["shapedirs"][:, :, :10] - Z["shapedirs"][:, :, :10]).max() == 0
face_idx = np.load(LHM / "pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
flame = smplx.create(str(LHM / "pretrained_models/human_model_files"), "flame", gender="neutral", num_betas=10, num_expression_coeffs=100)
EF = flame.expr_dirs.detach().cpu().numpy().astype(np.float64).reshape(-1, 100)
ES = SD[face_idx, :, 300:350].reshape(-1, 50)
MEXP, *_ = np.linalg.lstsq(EF, ES, rcond=None)


def betas10(b200):
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    n = np.linalg.norm((S10 @ b - S200 @ b200).reshape(-1, 3), axis=1) * 1000
    return b.astype(np.float32), float(n.mean()), float(n.max())


def expr100(e50):
    e = (MEXP @ e50.T.astype(np.float64)).T
    n = np.linalg.norm((EF @ e.T - ES @ e50.T.astype(np.float64)).T.reshape(len(e50), -1, 3), axis=2) * 1000
    return e.astype(np.float32), float(n.mean()) if n.size else 0.0, float(n.max()) if n.size else 0.0


# ---------------- LHM's own source preprocessing (human_lrm.py get_bbox / infer_preprocess_image, minus `assert w < h`) ----------------
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


# ---------------- model ----------------
from accelerate import Accelerator  # noqa: E402
Accelerator()
t0 = time.time()
model = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"]).from_pretrained(MODEL); model.to(dev); model.eval(); model.to(torch.float32)
facedetect = FaceDetector("./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd", device=dev)
print(f"model loaded in {time.time() - t0:.1f} s (LHM {COMMIT[:7]}, {MODEL_REV}, {GPU})", flush=True)


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
    src = fr["source"]
    t_build = time.time()
    img = rd(B / "images" / f"{src}.png"); alpha = rd(B / "masks" / f"{src}.png", cv2.IMREAD_GRAYSCALE)
    image = infer_preprocess_image(img, alpha)
    face_box = None
    try:
        # --face-on-matte: detect and crop the head on the matted source (subject on white, as LHM's body input), so another
        # person in the photo cannot be picked (LHM takes the largest head; its own app assumes one person per photo).
        fimg = (img * (alpha[..., None] > 127) + 255 * (alpha[..., None] <= 127)).astype(np.uint8) if a.face_on_matte else img
        bbox = facedetect(torch.from_numpy(fimg).permute(2, 0, 1)); face_box = [int(v) for v in bbox]
        # LHM slices img[y0:y1, x0:x1] with the 1.65x expanded box; numpy clips y1 / x1 at the image edge but a negative y0 / x0
        # wraps around and gives an empty crop (-> zeros). --clamp-face clips y0 / x0 at 0 as well (PREREG addendum 2, amendment 1).
        y0, x0 = (max(0, face_box[1]), max(0, face_box[0])) if a.clamp_face else (face_box[1], face_box[0])
        head = fimg[y0: face_box[3], x0: face_box[2]]
        head = cv2.resize(head, dsize=(112, 112), interpolation=cv2.INTER_AREA); face_found = True
    except Exception:
        head = np.zeros((112, 112, 3), np.uint8); face_found = False
    head_t = torch.from_numpy(head / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    b10, sh_mean, sh_max = betas10(sp["betas"])
    ti = [sk[k] for k in test]; T = len(ti)
    e100, ex_mean, ex_max = expr100(sp["expression"][ti])
    P = {k: [] for k in ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose", "trans", "expr", "focal", "princpt", "img_size_wh")}
    intrs = []
    for j, (k, i) in enumerate(zip(test, ti)):
        K = cams["K"][ck[k]].astype(np.float64)
        assert abs(K[0, 2] - 256) < 1e-6 and abs(K[1, 2] - 256) < 1e-6
        I4 = torch.eye(4); I4[0, 0], I4[1, 1], I4[0, 2], I4[1, 2] = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]); intrs.append(I4)
        P["root_pose"].append(sp["global_orient"][i]); P["body_pose"].append(sp["body_pose"][i]); P["jaw_pose"].append(sp["jaw_pose"][i])
        P["leye_pose"].append(sp["leye_pose"][i]); P["reye_pose"].append(sp["reye_pose"][i]); P["lhand_pose"].append(sp["left_hand_pose"][i])
        P["rhand_pose"].append(sp["right_hand_pose"][i]); P["trans"].append(sp["transl"][i]); P["expr"].append(e100[j])
        P["focal"].append(np.array([K[0, 0], K[1, 1]], np.float32)); P["princpt"].append(np.array([K[0, 2], K[1, 2]], np.float32)); P["img_size_wh"].append(np.array([512, 512], np.float32))
    smp = {k: torch.from_numpy(np.stack(v).astype(np.float32))[None] for k, v in P.items()}
    smp["body_pose"] = smp["body_pose"].reshape(1, T, 21, 3); smp["lhand_pose"] = smp["lhand_pose"].reshape(1, T, 15, 3); smp["rhand_pose"] = smp["rhand_pose"].reshape(1, T, 15, 3)
    smp["betas"] = torch.from_numpy(b10)[None]
    c2ws = torch.eye(4)[None, None].repeat(1, T, 1, 1); intrs = torch.stack(intrs)[None]; bgs = torch.zeros(1, T, 3)
    with torch.no_grad():
        gs_list, qp, tmat = model.infer_single_view(image.unsqueeze(0).to(dev), head_t.unsqueeze(0).to(dev), None, None, render_c2ws=c2ws.to(dev),
                                                    render_intrs=intrs.to(dev), render_bg_colors=bgs.to(dev), smplx_params={k: v.to(dev) for k, v in smp.items()})
        torch.cuda.synchronize(); build_s = time.time() - t_build
        OUT.mkdir(parents=True, exist_ok=True); t_r = time.time(); n_done = 0
        for s in range(0, T, a.batch):
            bp = {"betas": smp["betas"].to(dev), "transform_mat_neutral_pose": tmat}
            bp.update({k: smp[k][:, s: s + a.batch].to(dev) for k in P})
            res = model.animation_infer(gs_list, qp, bp, render_c2ws=c2ws[:, s: s + a.batch].to(dev), render_intrs=intrs[:, s: s + a.batch].to(dev),
                                        render_bg_colors=bgs[:, s: s + a.batch].to(dev))
            rgb = res["comp_rgb"].clamp(0, 1).float().cpu().numpy()
            for j in range(rgb.shape[0]):
                cv2.imwrite(str(OUT / f"{test[s + j]}.png"), (rgb[j] * 255 + 0.5).astype(np.uint8)[..., ::-1]); n_done += 1
            del res
        render_s = (time.time() - t_r) / max(n_done, 1)
    if a.debug_dir:
        D = Path(a.debug_dir) / ds / clip; D.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(D / f"src_{src}.png"), (image[0].permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)[..., ::-1]); cv2.imwrite(str(D / f"head_{src}.png"), head[..., ::-1])
    run = {"method": a.name, "model": "LHM-1B-HF", "weights": MODEL_REV, "commit": COMMIT, "guava_commit": GUAVA_SHA, "train_s": round(build_s, 2),
           "render_s_per_frame": round(render_s, 4), "gpu": GPU, "iterations": 0, "views": 1, "ref_keys": [src], "n_test": n_done, "face_found": face_found, "face_box": face_box, "clamp_face": a.clamp_face, "face_on_matte": a.face_on_matte,
           "src_box_wider_than_tall": bool((lambda bb: (bb[2] - bb[0]) >= (bb[3] - bb[1]))(get_bbox(alpha).get_box())),
           "shape_fit_mm": {"mean": round(sh_mean, 3), "max": round(sh_max, 3)}, "expr_map_mm": {"mean": round(ex_mean, 3), "max": round(ex_max, 3)},
           "notes": "feed-forward, no per-person optimisation; GT SMPL-X driving; background 0; see phase3_dnt/baselines/lhm/NOTES.md",
           "loader_changes": "package adapter (run_lhm.py); LHM code unchanged (preprocessing copied without `assert w < h`)"}
    json.dump(run, open(OUT / "run.json", "w"), indent=1)
    print(f"done {c['id']}: {n_done} frames, build {build_s:.1f} s, {render_s * 1000:.0f} ms/frame, face {face_found}, shape fit {sh_mean:.2f} mm, expr map {ex_mean:.2f} mm", flush=True)
print("LHM_DONE", flush=True)
