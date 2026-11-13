"""Timing of IDOL (yiyuzhuang/IDOL @ 9fd9296, CVPR 2025) for the paper's speed table: the evaluated pipeline of
phase3_dnt/baselines/idol/run_idol.py, cut into phases, on the timing subset (make_timing_subset.py). Writes one JSON
(phase_timer.py schema). Nothing under /dev/shm/ubx858_baselines is modified; the only runtime patches are timers and two
silenced print helpers of the official code.

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

Phases (fixed names, phase_timer.py). Reconstruction, once per person (default 1 warm-up + 3 timed builds per clip):
  preprocess   source 512 crop + package StyleMatte alpha -> RGBA PNG -> IDOL's lib.utils.infer_util.load_image (alpha box,
               square pad box / 0.85, 896 resize, centre 640 x 896 crop, on white) -> GPU; 200 GT betas -> 10 SMPL-X v1.1
               betas (least squares, as run_idol.py)
  segmentation null: the package's StyleMatte matte is used (IDOL's demo would run rembg; the evaluation did not)
  pose         null: GT SMPL-X given by the package
  network      SapiensGS_SA_v1.forward_image_to_uv (Sapiens-1B encoder, 40 layers + hidden states; neck_SA_v3_skip
               transformer; UV upsampler) + UVNDecoder_gender._decode_feature + _sample_feature (UV code -> per-Gaussian
               attributes; pose independent, decoded once per person in the evaluation)
Animation, per driving frame (10 warm-up + 50 timed per clip, cycling over the clip's test keys):
  preprocess   GT SMPL-X 2020 params of the frame -> IDOL's 189-d vector (50 -> 10 expression fit, hands minus v1.1
               hands_mean, fixed 10 betas, scale 1) + 20-d camera ([fx, fy, cx, cy] + identity w2c) -> GPU tensors
  deformation  copy of the per-person Gaussian attributes (deform_pcd clamps hand offsets in place) + UVNDecoder_gender.deform_pcd
               (SMPLXDeformer_gender.prepare_deformer: SMPL-X v1.1 body model; fast-SNARF forward skinning of ~100k Gaussians)
  rendering    UVNDecoder_gender.forward_render (rotations, simple_knn distCUDA2 scales, covariances, diff-gaussian-rasterization
               512 x 512 on black; in eval mode IDOL also runs a visibility pass: a second rasterisation + pytorch3d knn_points,
               kept because the evaluated call ran it; its share is reported in the clip notes) + float / clamp of the image
  total        end to end (image on the GPU; PNG writing of the evaluation excluded)
Batch 1 (the evaluation rendered in batches of 5; frames are independent there), fp32 (as evaluated: the Sapiens torchscript
is fp32 and the checkpoint's bf16 encoder weights are loaded into it), 512 px, K of each test frame.

usage: bash run_idol.sh SUBSET_ROOT OUT_DIR   (or: python time_idol.py --root SUBSET_ROOT --out OUT_DIR/idol.json)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
if "OPENBLAS_CORETYPE" not in os.environ:                 # the evaluation set SkylakeX (wrong BLAS on the server Xeon);
    try:                                                   # only valid on AVX-512 CPUs, so set it only there
        if " avx512f" in open("/proc/cpuinfo").read(): os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"
    except OSError:
        pass
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from phase_timer import PROTOCOL, Phases, sync, write  # noqa: E402

IDOL = Path(os.environ.get("IDOL_CODE", "/dev/shm/ubx858_baselines/idol/src"))
SMPLX2020 = str(HERE.parent.parent / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz")

ap = argparse.ArgumentParser(description="IDOL timing (see the module docstring)")
ap.add_argument("--root", default="/dev/shm/ubx858_timing/timing_subset_v1", help="timing subset root (manifest.json)")
ap.add_argument("--clips", default="", help="comma list of clip ids (default: all in the manifest)")
ap.add_argument("--out", required=True, help="output JSON path")
ap.add_argument("--warmup", type=int, default=PROTOCOL["warmup"]); ap.add_argument("--iters", type=int, default=PROTOCOL["iters"])
ap.add_argument("--recon-warmup", type=int, default=PROTOCOL["recon_warmup"])
ap.add_argument("--recon-reps", type=int, default=PROTOCOL["recon_reps"])
ap.add_argument("--split-frames", type=int, default=20, help="extra untimed-by-phase frames for the rendering split diagnostic (0 = off)")
ap.add_argument("--png", type=int, default=1, help="sanity PNGs per clip under <out dir>/idol_png (0 = none)")
ap.add_argument("--smplx2020", default=SMPLX2020)
ap.add_argument("--power-note", default="")
a = ap.parse_args()
ROOT = Path(a.root).resolve(); OUTJ = Path(a.out).resolve(); OUTD = OUTJ.parent; OUTD.mkdir(parents=True, exist_ok=True)
assert not str(OUTD).startswith(("/dev/shm/ubx858_baselines", "/raid", "/vault")), f"refusing to write under {OUTD}"
SMPLX2020 = str(Path(a.smplx2020).resolve())

os.chdir(IDOL); sys.path.insert(0, str(IDOL))             # IDOL's config uses paths relative to src/
import cv2, torch  # noqa: E401,E402
from omegaconf import OmegaConf  # noqa: E402
import lib.utils.infer_util as IU  # noqa: E402
from lib.utils.train_util import instantiate_from_config  # noqa: E402

dev = torch.device("cuda")
_r = np.random.default_rng(0); _A = _r.normal(size=(15069, 100)); _X = _r.normal(size=(100, 50))
assert np.abs(_A @ _X - (torch.from_numpy(_A) @ torch.from_numpy(_X)).numpy()).max() < 1e-9, \
    "numpy BLAS gives wrong products: set OPENBLAS_CORETYPE to your CPU's core type (e.g. Haswell / SkylakeX / Zen)"
COMMIT = subprocess.run(["git", "-C", str(IDOL), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
GPU = torch.cuda.get_device_name(0); CC = torch.cuda.get_device_capability(0)
print(f"IDOL {COMMIT} on {GPU} (sm_{CC[0]}{CC[1]}), torch {torch.__version__}", flush=True)

# ---------------- fixed linear conversions (as run_idol.py; per model, not per person) ----------------
Z = np.load(SMPLX2020, allow_pickle=True); SD = Z["shapedirs"].astype(np.float64)
V11 = pickle.load(open(IDOL / "lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.pkl", "rb"), encoding="latin1")
g11 = lambda k: np.asarray(V11[k].toarray() if hasattr(V11[k], "toarray") else V11[k])  # noqa: E731
for k in ("v_template", "J_regressor", "weights", "posedirs"): assert np.abs(g11(k) - Z[k]).max() == 0, k
SD11 = g11("shapedirs").astype(np.float64); assert np.abs(SD11[:, :, :10] - SD[:, :, :10]).max() == 0
HM = np.concatenate([g11("hands_meanl"), g11("hands_meanr")]).astype(np.float32)                 # (90,)
S200 = SD[:, :, :200].reshape(-1, 200); S10 = SD[:, :, :10].reshape(-1, 10)
E50 = SD[:, :, 300:350].reshape(-1, 50); E10 = SD11[:, :, 300:310].reshape(-1, 10)
MEXP, *_ = np.linalg.lstsq(E10, E50, rcond=None)                                                     # (10, 50)
EYE16 = np.eye(4, dtype=np.float32).reshape(-1)


def betas10(b200):
    b, *_ = np.linalg.lstsq(S10, S200 @ b200.astype(np.float64), rcond=None)
    return b.astype(np.float32)


# ---------------- model, loaded exactly as run_idol.py / run_demo.py main ----------------
t0 = time.time()
config = OmegaConf.load("configs/idol_v0.yaml")
model = instantiate_from_config(config.model)
model.encoder = model.encoder.to(torch.bfloat16)
model = model.__class__.load_from_checkpoint("work_dirs/ckpt/model.ckpt", **config.model.params)
model = model.to(dev).eval()
model.decoder.renderer.image_size = [512, 512]
model.decoder.renderer.bg = torch.zeros(3, device=dev)
DT = {str(p.dtype) for p in model.encoder.parameters()} | {str(p.dtype) for p in model.neck.parameters()}
print(f"model loaded in {time.time() - t0:.1f} s; encoder / neck param dtypes {sorted(DT)}", flush=True)
assert DT == {"torch.float32"}, f"expected the evaluated fp32 model, got {DT}"
LOAD_GPU = (torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)
HOST_LOAD_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

# official-code prints (one per call) silenced: they are log lines, not model work
import lib.models.decoders.uvmaps_decoder_gender as UD  # noqa: E402
UD.main_print = lambda *x, **k: None
IU.print = lambda *x, **k: None
PRINTS_NOTE = "IDOL's per-call log prints (main_print in the decoder, 'Processing:' in load_image) are silenced"

man = json.loads((ROOT / "manifest.json").read_text())
want = [c for c in a.clips.split(",") if c]
clips = [c for c in man["clips"] if not want or c["id"] in want]
assert clips and (not want or len(clips) == len(want)), f"clips {want} not all in the manifest"
TMP = Path(tempfile.mkdtemp(prefix="idol_src_", dir=str(OUTD)))
PNG = OUTD / "idol_png"
KIMG = "no kernel image"


def hint(e):
    if KIMG in str(e):
        print(f"\nCUDA extensions not built for this GPU (sm_{CC[0]}{CC[1]}): rebuild them as in the note at the top of "
              f"{Path(__file__).name}.\n", flush=True)
    raise e


res_clips = {}; peaks = {}
try:
    for c in clips:
        cid = c["id"]; ds, clip = c["dataset"], c["clip"]; B = ROOT / c["E"] / "base" / ds / clip
        fr = json.loads((B / "frames.json").read_text()); test = list(fr["test"]); src = fr["source"]
        # one-time I/O of the package parameters (the evaluation read them once per clip, outside its timers)
        cams = dict(np.load(B / "cameras.npz")); sp = dict(np.load(B / "smplx2020.npz"))
        ck = {k: i for i, k in enumerate(cams["keys"].tolist())}; sk = {k: i for i, k in enumerate(sp["keys"].tolist())}
        for k in test:
            K = cams["K"][ck[k]]; assert abs(K[0, 2] - 256) < 1e-6 and abs(K[1, 2] - 256) < 1e-6 and abs(K[0, 0] - K[1, 1]) < 1e-6
        print(f"== {cid}: source {src}, {len(test)} test keys", flush=True)

        # ---------------- reconstruction ----------------
        R = Phases(); torch.cuda.reset_peak_memory_stats(); sub = {"encoder": [], "neck_upsample": [], "uv_decode": []}
        enc_fwd = model.encoder.forward
        for i in range(a.recon_warmup + a.recon_reps):
            rec = i >= a.recon_warmup
            try:
                R.start_run(record=rec)
                with R("preprocess"):
                    rgba = np.dstack([cv2.imread(str(B / "images" / f"{src}.png"), cv2.IMREAD_COLOR)[..., ::-1],
                                      cv2.imread(str(B / "masks" / f"{src}.png"), cv2.IMREAD_GRAYSCALE)])
                    sp_png = TMP / f"{ds}_{clip}.png"; cv2.imwrite(str(sp_png), rgba[..., [2, 1, 0, 3]])
                    image = IU.load_image(str(sp_png), str(TMP))                                       # [3, 896, 640] on white
                    img_dev = image[None].to(dev)
                    b10 = betas10(sp["betas"])
                with R("network"), torch.no_grad():
                    sync(); t1 = time.perf_counter()
                    feats = {}
                    def enc(*x, **k):                                                                   # noqa: E306
                        out = enc_fwd(*x, **k); sync(); feats["t"] = time.perf_counter(); return out
                    model.encoder.forward = enc
                    code = model.forward_image_to_uv(img_dev, is_training=False)
                    sync(); t2 = time.perf_counter()
                    res_uv = model.decoder._decode_feature(code.expand(1, -1, -1, -1))
                    res_points = model.decoder._sample_feature(res_uv)
                    sync(); t3 = time.perf_counter()
                    if rec:
                        sub["encoder"].append((feats["t"] - t1) * 1e3); sub["neck_upsample"].append((t2 - feats["t"]) * 1e3)
                        sub["uv_decode"].append((t3 - t2) * 1e3)
                R.end_run()
            except RuntimeError as e:
                hint(e)
            finally:
                model.encoder.__dict__.pop("forward", None)            # back to the class method
        recon = R.summary(); recon.update(segmentation=None, pose=None)
        pk_r = (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30)

        # ---------------- animation (GT pose given), batch 1 ----------------
        def prep(k):
            i = sk[k]
            e10 = (MEXP @ sp["expression"][i].astype(np.float64)).astype(np.float32)
            hands = np.concatenate([sp["left_hand_pose"][i].reshape(-1), sp["right_hand_pose"][i].reshape(-1)]) - HM
            p = np.concatenate([[1.0], sp["transl"][i].reshape(3), sp["global_orient"][i].reshape(3), sp["body_pose"][i].reshape(63), b10,
                                hands, sp["jaw_pose"][i].reshape(3), sp["leye_pose"][i].reshape(3), sp["reye_pose"][i].reshape(3), e10]).astype(np.float32)
            K = cams["K"][ck[k]]
            cam = np.concatenate([np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32), EYE16])
            return (torch.from_numpy(p[None]).to(dev).to(code.dtype), torch.from_numpy(cam[None, None]).to(dev).to(code.dtype))

        def deform(Pt):
            pts = tuple(x.repeat(1, *([1] * (x.dim() - 1))) if torch.is_tensor(x) else x for x in res_points)
            return model.decoder.deform_pcd(pts, Pt, zeros_hands_off=True, value=0.02)

        def render(res_def, camt):
            out = model.decoder.forward_render(res_def, camt, num_imgs=1)
            return out["image"][:, 0].float().clamp(0, 1)

        A = Phases(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(a.warmup + a.iters):
                k = test[i % len(test)]
                try:
                    A.start_run(record=i >= a.warmup)
                    with A("preprocess"): Pt, camt = prep(k)
                    with A("deformation"): res_def = deform(Pt)
                    with A("rendering"): img = render(res_def, camt)
                    A.end_run()
                except RuntimeError as e:
                    hint(e)
        anim = A.summary(); pk_a = (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30)

        # ---------------- diagnostic: share of IDOL's eval-mode visibility pass inside "rendering" ----------------
        split_note = ""
        if a.split_frames > 0:
            rg0 = model.decoder.renderer.render_gaussian; knn0 = UD.ops.knn_points; st = {"n": 0}
            dd = {"rendering": [], "raster_image": [], "visibility_pass": []}
            def rg(*x, **kw):                                                                           # noqa: E306
                sync(); t = time.perf_counter(); out = rg0(*x, **kw); sync(); dt = (time.perf_counter() - t) * 1e3
                if st["n"] % 2 == 0: dd["raster_image"].append(dt)
                else: st["vis"] = dt
                st["n"] += 1; return out
            def knn(*x, **kw):                                                                          # noqa: E306
                sync(); t = time.perf_counter(); out = knn0(*x, **kw); sync()
                dd["visibility_pass"].append(st.pop("vis") + (time.perf_counter() - t) * 1e3); return out
            model.decoder.renderer.render_gaussian = rg; UD.ops.knn_points = knn
            try:
                with torch.no_grad():
                    for i in range(a.split_frames):
                        Pt, camt = prep(test[i % len(test)]); res_def = deform(Pt)
                        sync(); t = time.perf_counter(); render(res_def, camt); sync(); dd["rendering"].append((time.perf_counter() - t) * 1e3)
            finally:
                del model.decoder.renderer.render_gaussian; UD.ops.knn_points = knn0
            m = {k: float(np.median(v)) for k, v in dd.items()}
            split_note = (f"rendering split (diagnostic, {a.split_frames} extra frames with inner syncs, not in the phases; median ms): "
                          f"rendering {m['rendering']:.1f} = image rasterisation {m['raster_image']:.1f} + eval-mode visibility pass "
                          f"(2nd rasterisation + knn_points) {m['visibility_pass']:.1f} + rest (rotations, distCUDA2 scales, covariances) "
                          f"{m['rendering'] - m['raster_image'] - m['visibility_pass']:.1f}. ")

        # ---------------- sanity PNGs (outside all timers) ----------------
        if a.png > 0:
            PNG.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for k in test[: a.png]:
                    Pt, camt = prep(k); im = render(deform(Pt), camt)[0].cpu().numpy()
                    cv2.imwrite(str(PNG / f"{ds}__{clip}__{k}.png"), (im[..., :3] * 255 + 0.5).astype(np.uint8)[..., ::-1])

        md = lambda v: float(np.median(v)) if v else float("nan")                                       # noqa: E731
        peaks[cid] = {"recon_alloc_gb": round(pk_r[0], 2), "recon_reserved_gb": round(pk_r[1], 2),
                      "anim_alloc_gb": round(pk_a[0], 2), "anim_reserved_gb": round(pk_a[1], 2)}
        note = (f"source {src}; {len(test)} test keys cycled; network split (median ms): Sapiens-1B encoder {md(sub['encoder']):.1f}, "
                f"neck + UV upsampler {md(sub['neck_upsample']):.1f}, UV attribute decode + sampling {md(sub['uv_decode']):.1f}; "
                + split_note + f"peak GPU memory (GB): reconstruction {pk_r[0]:.2f} allocated / {pk_r[1]:.2f} reserved, "
                f"animation {pk_a[0]:.2f} / {pk_a[1]:.2f}")
        res_clips[cid] = {"reconstruction": recon, "animation": anim, "notes": note}
        g = lambda d, k: d[k]["median_ms"] if d.get(k) else None                                       # noqa: E731
        print(f"   reconstruction median ms: " + ", ".join(f"{k} {g(recon, k)}" for k in ("preprocess", "network", "total")), flush=True)
        print(f"   animation median ms: " + ", ".join(f"{k} {g(anim, k)}" for k in ("preprocess", "deformation", "rendering", "total")), flush=True)
        print(f"   {note}", flush=True)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

host_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
mem_note = (f"GPU memory: model resident {LOAD_GPU[0]:.2f} GB allocated after load (load peak {LOAD_GPU[1]:.2f}); peak during timing "
            f"{max(max(v['recon_alloc_gb'], v['anim_alloc_gb']) for v in peaks.values()):.2f} GB allocated / "
            f"{max(max(v['recon_reserved_gb'], v['anim_reserved_gb']) for v in peaks.values()):.2f} GB reserved (+ CUDA context); "
            f"host RAM peak {host_gb:.1f} GB (after model load {HOST_LOAD_GB:.1f} GB).")
notes = ("IDOL (official code @ %s, configs/idol_v0.yaml + model.ckpt HF yiyuzhuang/IDOL@3c80ec95), the evaluated pipeline of "
         "phase3_dnt/baselines/idol/run_idol.py. Reconstruction: preprocess = source crop + package StyleMatte alpha -> RGBA -> "
         "IDOL load_image (crop / pad / 896 resize / 640 x 896 centre crop on white) -> GPU, and 200 -> 10 betas (least squares); "
         "segmentation = null (package StyleMatte matte, no rembg); pose = null (GT SMPL-X); network = forward_image_to_uv "
         "(Sapiens-1B + neck transformer + UV upsampler) + decoder._decode_feature + _sample_feature (UV -> Gaussians, once per person). "
         "Animation: preprocess = GT SMPL-X 2020 -> IDOL 189-d v1.1 params + 20-d camera -> GPU; deformation = Gaussian attribute copy "
         "+ decoder.deform_pcd (SMPL-X v1.1 body model + fast-SNARF skinning); rendering = decoder.forward_render (distCUDA2 scales, "
         "covariances, antialiased diff-gaussian-rasterization 512 x 512, plus IDOL's eval-mode visibility pass: a 2nd rasterisation "
         "+ pytorch3d knn_points, kept as evaluated; see clip notes for its share) + image clamp; total ends with the image on the GPU. "
         "Batch 1 (the evaluation rendered 5 frames per call), fp32, 512 px, per-frame K. %s. %s" % (COMMIT, PRINTS_NOTE, mem_note))
protocol = dict(PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=a.recon_warmup, recon_reps=a.recon_reps,
                precision="fp32", render_px=512)
write(OUTJ, method="idol", row="IDOL", clips=res_clips, notes=notes, protocol=protocol, power_note=a.power_note)
print(mem_note, flush=True)
