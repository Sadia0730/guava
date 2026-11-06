"""IDOL (yiyuzhuang/IDOL, CVPR 2025) as a feed-forward single-image baseline under the AvatarTest-v1 contract
(phase3_dnt/baselines/CONTRACT.md): avatar (UV Gaussian code) from the clip's source frame only, driven by the GT SMPL-X 2020
params of each test frame, rendered with the 512 crop camera (K[key], identity extrinsics) on black, test keys only.

- Model: configs/idol_v0.yaml + model.ckpt (HF yiyuzhuang/IDOL @ 3c80ec95), loaded exactly as run_demo.py does.
- Source: IDOL's own load_image() on an RGBA PNG = 512 source crop + the package's StyleMatte alpha (IDOL's demo would run
  rembg; the matte is the one every row gets): alpha box, square pad (box / 0.85), 896, centre 640 x 896 crop, on white.
- Body model: IDOL's SMPL-X is v1.1 neutral (pkl), identical to SMPL-X 2020 in template, regressor, skinning, pose correctives and
  the first 10 shape dirs (asserted); only the expression basis differs. Conversions: 10 betas = least-squares fit of the 200-beta
  displacement (as for LHM / LHM++); 10 v1.1 expression coefficients = least-squares fit of the 2020 50-coefficient expression
  displacement over all vertices (IDOL drives 10; residual logged); hands = GT - v1.1 hands_mean (IDOL's layer is
  flat_hand_mean = False, use_pca = False with 189-d params; the GT is flat_hand_mean = True); body / root / transl / jaw / eyes
  copied; scale 1. The posed-mesh error of the converted params vs the GT mesh is logged per clip (body_fit_mm).
- Render: IDOL's decoder path of run_demo.py (novel_pose: _decode_feature, _sample_feature, deform_pcd(zeros_hands_off=True,
  value=0.02), forward_render) with cameras = [fx, fy, cx, cy] + identity w2c, image_size 512 x 512, renderer background set to
  0 (pure 3DGS, so this is the premultiplied render on black, the contract output).
Run with the lhmpp env + PYTHONPATH=/dev/shm/ubx858_baselines/idol/pydeps (antialiased diff-gaussian-rasterization @ 9c5c202 and
IDOL's fast-SNARF extensions, built there).
"""
import argparse, json, os, subprocess, sys, time, tempfile, pickle
from pathlib import Path

os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"
import numpy as np

IDOL = Path("/dev/shm/ubx858_baselines/idol/src")
CKPT_REV = "yiyuzhuang/IDOL@3c80ec956f9bede01d9fbb3917ca229f8d2673b1 (model.ckpt sha256 2294c771...)"
SMPLX2020 = "/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True); ap.add_argument("--split", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--clips", default=""); ap.add_argument("--max-test", type=int, default=0); ap.add_argument("--batch", type=int, default=5)
ap.add_argument("--debug-dir", default="")
a = ap.parse_args()
E = Path(a.E); OUTROOT = E / "renders" / a.name

os.chdir(IDOL); sys.path.insert(0, str(IDOL))
import cv2, torch, smplx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from lib.utils.infer_util import load_image  # noqa: E402
from lib.utils.train_util import instantiate_from_config  # noqa: E402

dev = torch.device("cuda")
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, "numpy BLAS wrong (OPENBLAS_CORETYPE)"
COMMIT = subprocess.run(["git", "-C", str(IDOL), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
GUAVA_SHA = subprocess.run(["git", "-C", "/home/ubx858/guava", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
GPU = torch.cuda.get_device_name(0)

# ---------------- fixed linear conversions ----------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)
V11 = pickle.load(open(IDOL / "lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.pkl", "rb"), encoding="latin1")
g11 = lambda k: np.asarray(V11[k].toarray() if hasattr(V11[k], "toarray") else V11[k])
for k in ("v_template", "J_regressor", "weights", "posedirs"): assert np.abs(g11(k) - Z[k]).max() == 0, k
SD11 = g11("shapedirs").astype(np.float64); assert np.abs(SD11[:, :, :10] - SD[:, :, :10]).max() == 0
HM = np.concatenate([g11("hands_meanl"), g11("hands_meanr")]).astype(np.float32)                 # (90,)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
E50 = SD[:, :, 300:350].reshape(-1, 50); E10 = SD11[:, :, 300:310].reshape(-1, 10)
MEXP, *_ = np.linalg.lstsq(E10, E50, rcond=None)                                                     # (10, 50)


def betas10(b200):
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    n = np.linalg.norm((S10 @ b - S200 @ b200).reshape(-1, 3), axis=1) * 1000
    return b.astype(np.float32), float(n.mean())


body2020 = smplx.SMPLX(SMPLX2020, num_betas=200, num_expression_coeffs=50, use_pca=False, flat_hand_mean=True).to(dev)
body11 = smplx.SMPLX(str(IDOL / "lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.npz"), num_betas=10, num_expression_coeffs=10,
                     use_pca=False, flat_hand_mean=False).to(dev)

# ---------------- model (as run_demo.py main) ----------------
t0 = time.time()
config = OmegaConf.load("configs/idol_v0.yaml")
model = instantiate_from_config(config.model)
model.encoder = model.encoder.to(torch.bfloat16)
model = model.__class__.load_from_checkpoint("work_dirs/ckpt/model.ckpt", **config.model.params)
model = model.to(dev).eval()
model.decoder.renderer.image_size = [512, 512]
model.decoder.renderer.bg = torch.zeros(3, device=dev)
print(f"model loaded in {time.time() - t0:.1f} s (IDOL {COMMIT[:7]}, {GPU})", flush=True)


def rd(p, flag=cv2.IMREAD_COLOR):
    x = cv2.imread(str(p), flag)
    if x is None: raise FileNotFoundError(p)
    return x[..., ::-1].copy() if flag == cv2.IMREAD_COLOR else x


split = json.load(open(a.split)); clips = split["clips"]
want = set(c for c in a.clips.split(",") if c)
todo = [c for c in clips if not want or c["id"] in want]
print(f"{len(todo)} clips -> {OUTROOT}", flush=True)
TMP = Path(tempfile.mkdtemp(prefix="idol_src_", dir="/dev/shm/ubx858_baselines/idol"))

for c in todo:
    ds, clip = c["dataset"], c["clip"]; B = E / "base" / ds / clip; OUT = OUTROOT / ds / clip
    fr = json.load(open(B / "frames.json")); test = list(fr["test"])[: a.max_test] if a.max_test else list(fr["test"])
    if OUT.exists() and all((OUT / f"{k}.png").exists() for k in test) and (OUT / "run.json").exists():
        print(f"skip {c['id']} (complete)", flush=True); continue
    cams = np.load(B / "cameras.npz"); sp = np.load(B / "smplx2020.npz")
    ck = {k: i for i, k in enumerate(cams["keys"].tolist())}; sk = {k: i for i, k in enumerate(sp["keys"].tolist())}
    src = fr["source"]; t_build = time.time()
    rgba = np.dstack([rd(B / "images" / f"{src}.png"), rd(B / "masks" / f"{src}.png", cv2.IMREAD_GRAYSCALE)])
    sp_png = TMP / f"{ds}_{clip}.png"; cv2.imwrite(str(sp_png), rgba[..., [2, 1, 0, 3]])
    image = load_image(str(sp_png), str(TMP))                                                        # [3, 896, 640] on white
    with torch.no_grad():
        code = model.forward_image_to_uv(image[None].to(dev), is_training=False)
    b10, sh_mean = betas10(sp["betas"])
    ti = [sk[k] for k in test]; T = len(ti)
    e10 = (MEXP @ sp["expression"][ti].T.astype(np.float64)).T.astype(np.float32)
    hands = np.concatenate([sp["left_hand_pose"][ti].reshape(T, -1), sp["right_hand_pose"][ti].reshape(T, -1)], 1) - HM[None]
    P = np.concatenate([np.ones((T, 1)), sp["transl"][ti].reshape(T, 3), sp["global_orient"][ti].reshape(T, 3), sp["body_pose"][ti].reshape(T, 63),
                        np.repeat(b10[None], T, 0), hands, sp["jaw_pose"][ti].reshape(T, 3), sp["leye_pose"][ti].reshape(T, 3),
                        sp["reye_pose"][ti].reshape(T, 3), e10], 1).astype(np.float32)
    assert P.shape[1] == 189
    with torch.no_grad():                                                                             # posed-mesh check of the conversion
        gt = lambda n: torch.from_numpy(sp[n][ti].reshape(T, -1)).float().to(dev)
        Vg = body2020(betas=torch.from_numpy(sp["betas"][None]).float().to(dev).expand(T, -1), global_orient=gt("global_orient"), body_pose=gt("body_pose"),
                      jaw_pose=gt("jaw_pose"), leye_pose=gt("leye_pose"), reye_pose=gt("reye_pose"), left_hand_pose=gt("left_hand_pose"),
                      right_hand_pose=gt("right_hand_pose"), expression=gt("expression"), transl=gt("transl")).vertices
        Pt = torch.from_numpy(P).to(dev)
        Vi = body11(betas=Pt[:, 70:80], global_orient=Pt[:, 4:7], body_pose=Pt[:, 7:70], left_hand_pose=Pt[:, 80:125], right_hand_pose=Pt[:, 125:170],
                    jaw_pose=Pt[:, 170:173], leye_pose=Pt[:, 173:176], reye_pose=Pt[:, 176:179], expression=Pt[:, 179:189], transl=Pt[:, 1:4]).vertices
        err = (Vi - Vg).norm(dim=-1) * 1000; fit_mean, fit_max = float(err.mean()), float(err.max())
    cam = np.zeros((T, 20), np.float32)
    for j, k in enumerate(test):
        K = cams["K"][ck[k]]; assert abs(K[0, 2] - 256) < 1e-6 and abs(K[1, 2] - 256) < 1e-6 and abs(K[0, 0] - K[1, 1]) < 1e-6
        cam[j, :4] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]; cam[j, 4:] = np.eye(4, dtype=np.float32).reshape(-1)
    torch.cuda.synchronize(); build_s = time.time() - t_build
    OUT.mkdir(parents=True, exist_ok=True); t_r = time.time(); n_done = 0
    with torch.no_grad():
        res_uv = model.decoder._decode_feature(code.expand(1, -1, -1, -1))                          # appearance is pose independent: decode once
        res_points = model.decoder._sample_feature(res_uv)
        for s in range(0, T, a.batch):
            n = min(a.batch, T - s)
            pts = tuple(x.repeat(n, *([1] * (x.dim() - 1))) if torch.is_tensor(x) else x for x in res_points)
            res_def = model.decoder.deform_pcd(pts, torch.from_numpy(P[s: s + n]).to(dev).to(code.dtype), zeros_hands_off=True, value=0.02)
            out = model.decoder.forward_render(res_def, torch.from_numpy(cam[s: s + n, None]).to(dev).to(code.dtype), num_imgs=1)
            img = out["image"][:, 0].float().clamp(0, 1).cpu().numpy()
            for j in range(n):
                cv2.imwrite(str(OUT / f"{test[s + j]}.png"), (img[j][..., :3] * 255 + 0.5).astype(np.uint8)[..., ::-1]); n_done += 1
    render_s = (time.time() - t_r) / max(n_done, 1)
    if a.debug_dir:
        D = Path(a.debug_dir) / ds / clip; D.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(D / f"src_{src}.png"), (image.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)[..., ::-1])
    run = {"method": a.name, "model": "IDOL", "weights": CKPT_REV, "commit": COMMIT, "guava_commit": GUAVA_SHA, "train_s": round(build_s, 2),
           "render_s_per_frame": round(render_s, 4), "gpu": GPU, "iterations": 0, "views": 1, "ref_keys": [src], "n_test": n_done,
           "shape_fit_mm": round(sh_mean, 3), "body_fit_mm": {"mean": round(fit_mean, 3), "max": round(fit_max, 3)},
           "notes": "feed-forward, no per-person optimisation; GT SMPL-X driving (v1.1, 10 betas, 10 expr); background 0; see phase3_dnt/baselines/idol/NOTES.md",
           "loader_changes": "package adapter (run_idol.py); IDOL code unchanged"}
    json.dump(run, open(OUT / "run.json", "w"), indent=1)
    print(f"done {c['id']}: {n_done} frames, build {build_s:.1f} s, {render_s * 1000:.0f} ms/frame, shape fit {sh_mean:.2f} mm, body fit {fit_mean:.2f} / {fit_max:.1f} mm", flush=True)
print("IDOL_DONE", flush=True)
