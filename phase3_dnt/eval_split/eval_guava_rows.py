"""AvatarTest-v1 GUAVA-family rows (PREREG "AvatarTest-v1 addendum 1"): every test frame rendered at 512 (8-bit PNG, black
background) into E/renders/<row>/<dataset>/<clip>/<key>.png, plus run.json per clip (avatar-creation time, render time).
  fb_gt      GUAVA-FB (guava_fb_B30k best.pt, 28k) from the source frame's GT record + crop, driven by the GT test records.
  guava      released GUAVA (best_160000.pt; EHM model with the FLAME head), same source / targets; FLAME fields from the
             converter (flame_shape fit, expression = exp50, jaw / eyes); camera patch (per-record invtanfov). The EHM head
             vs the GT SMPL-X 2020 head is measured on 3 frames per clip (mm) and logged.
  fb_sprint  GUAVA-FB from SPRINT's fit of the source frame's ORIGINAL image (sprint_avatar_fit, 150 refinement steps), driven
             per test frame by SPRINT raw with the shape lock (Avatar A's SL); rendered at 512 in SPRINT's crop camera (GUAVA's
             output size) and warped into the metric crop with the exact homography (SPRINT crop -> original -> metric crop).
Seed 0 before every inference / render call (StyleUNet noise), as fb_common / compare_fb.
Usage: CUDA_VISIBLE_DEVICES=1 python eval_guava_rows.py --row fb_gt [--clips id,id]"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--row", required=True, choices=("fb_gt", "guava", "fb_sprint"))
ap.add_argument("--clips", default=""); ap.add_argument("--sprint-iters", type=int, default=150)
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
ap.add_argument("--guava-ckpt", default=str(ROOT / "assets/GUAVA/checkpoints/best_160000.pt"))
ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
a = ap.parse_args()
SPLIT = json.loads((HERE / "avatar_test_v1.json").read_text())
E = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1")
OUT = E / "renders" / a.row
clips = SPLIT["clips"]
if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]

fitter = None
if a.row == "fb_sprint":                                    # SPRINT first: it loads PEAR's packages, then purges them
    spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt/sprint_avatar_fit.py")
    SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
    fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", a.sprint_iters)

import cv2, torch, torchvision                                                                       # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer              # noqa: E402
dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)


def load_ckpt(inf, ren, path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    r1 = inf.load_state_dict(st["model"], strict=False); r2 = ren.load_state_dict(st["render_model"], strict=False)
    return {"ckpt": path, "step": st.get("step", st.get("global_iter")), "missing": len(r1.missing_keys) + len(r2.missing_keys),
            "unexpected": len(r1.unexpected_keys) + len(r2.unexpected_keys)}


if a.row == "guava":
    cfg = F.get_cfg(); inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev); ren = GaussianRenderer(cfg.MODEL).to(dev)
    CK = load_ckpt(inf, ren, a.guava_ckpt); F.patch_inferer(inf)
else:
    cfg, inf, ren, _ = F.setup_a20(dev); CK = load_ckpt(inf, ren, a.fb_ckpt)
inf.eval(); ren.eval()
print("model", CK, GPU, flush=True)


class FBDataRel(F.FBData):
    """FBData + the record's FLAME fields (released GUAVA's EHM reads them)."""
    def info(self, ci, k):
        r, img, msk = super().info(ci, k); c = self.clips[ci]
        fc = {kk: torch.as_tensor(np.asarray(v, np.float32).reshape(-1)) for kk, v in c["flame"][k].items()
              if kk in ("expression_params", "jaw_params", "pose_params", "neck_pose_params", "eye_pose_params", "eyelid_params")}
        fc["shape_params"] = torch.as_tensor(c["flame_shape"]); r["flame_coeffs"] = fc
        return r, img, msk


@torch.no_grad()
def build(src):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    torch.manual_seed(0); vg, ug, _ = inf(src)
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    torch.cuda.synchronize(); return av, (time.perf_counter() - t0) * 1000


@torch.no_grad()
def render(av, t):
    torch.manual_seed(0); return ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)[0]


def u8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def original_frames(c, frames):
    """{frame: RGB uint8} of the ORIGINAL video on the GT's timeline."""
    import convert_bedlam2 as cb
    if c["dataset"] == "bedlam2": return cb.decode_frames(c["job"], c["seq"], frames)
    import convert_test as CT
    CT.TMP.mkdir(parents=True, exist_ok=True); out = {}
    with tempfile.TemporaryDirectory(dir=CT.TMP) as td:
        subprocess.run([CT.FFMPEG, "-loglevel", "error", "-i", c["video"], "-f", "image2", "-r", "30", f"{td}/%06d.png"], check=True)
        for f in frames: out[f] = cv2.cvtColor(cv2.imread(f"{td}/{f + 1:06d}.png"), cv2.COLOR_BGR2RGB)
    return out


_ANN, _MAPS = {}, {}


def metric_H(c):
    """original px -> 1024 metric crop px (the converter's geometry, recomputed), per GT frame."""
    import convert_bedlam2 as cb
    import convert_test as CT
    if c["dataset"] == "bedlam2":
        d = cb.job_npz(c["job"]); bm = np.load(cb.BMAP)
        betas = bm["M"].astype(np.float64) @ d["shape16"][c["rows"][0]].astype(np.float64) + bm["c"].astype(np.float64)
        g = cb.clip_geometry({"rows": c["rows"]}, d, betas)
    else:
        if c["scene"] not in _ANN:
            _ANN.clear(); _ANN[c["scene"]] = json.load(zipfile.ZipFile("/raid/ubx858/datasets/raw/UBody/annotations.zip").open(f"annotations/{c['scene']}/smplx_annotation.json"))
        sx = _ANN[c["scene"]]; _MAPS.setdefault("m", CT.maps()); g, _, _ = CT.ub_geometry(c, [sx[str(i)] for i in c["ann_ids"]], _MAPS["m"])
    return {f: gi["H"] for f, gi in zip(c["frames"], g)}


S512 = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])                                    # pixel-index 1024 -> 512


def main():
    OUT.mkdir(parents=True, exist_ok=True); warm = False
    for c in clips:
        ds, name = c["dataset"], c["clip"]; od = OUT / ds / name
        if (od / "run.json").exists(): continue
        cdir = E / "data" / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
        if not (cdir / "meta.json").exists(): print("missing data", c["id"], flush=True); continue
        od.mkdir(parents=True, exist_ok=True); info = {"method": a.row, **CK, "gpu": GPU, "n_test": len(fj["test"])}
        if a.row in ("fb_gt", "guava"):
            cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1)
            if a.row == "guava":
                raw = F.load_pkl(cdir / "optim_tracking_ehm.pkl"); ids = F.load_pkl(cdir / "id_share_params.pkl")
                cl[0]["flame"] = {k: raw[k]["flame_coeffs"] for k in raw}; cl[0]["flame_shape"] = np.asarray(ids["flame_shape"], np.float32).reshape(300)
                fbd = FBDataRel(cfg, cl)
            else:
                fbd = F.FBData(cfg, cl)
            src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev)
            if not warm: build(src); warm = True                                                   # CUDA / cuDNN warm-up (not timed)
            av, ms = build(src); info["recon_ms"] = ms; info["recon_input"] = "first frame (GT SMPL-X record)"
            if a.row == "guava":                                                                   # EHM (FLAME head) vs GT SMPL-X 2020 head
                from smplx_lean import SMPLXLean
                lean = SMPLXLean(str(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), 200, dtype=torch.float32).to(dev)
                fid = torch.as_tensor(np.load(ROOT / "assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy"), device=dev); errs = []
                for k in fj["test"][:: max(1, len(fj["test"]) // 3)][:3]:
                    t = to_dev(F.collate1(fbd.target_info(0, k)), dev); sc = t["smplx_coeffs"]
                    ve = inf.ehm(sc, t["flame_coeffs"])["vertices"][0, :F.NV]
                    pose = torch.cat([sc["global_pose"].reshape(1, 3), sc["body_pose"].reshape(1, 63), sc["jaw_pose"].reshape(1, 3),
                                      sc["eye_pose"].reshape(1, 6), sc["left_hand_pose"].reshape(1, 45), sc["right_hand_pose"].reshape(1, 45)], 1)
                    vl, _ = lean(pose, sc["shape"].reshape(1, 200), sc["exp"].reshape(1, 50))
                    d = (ve - vl[0]).norm(dim=-1) * 1000; errs.append([float(d[fid].mean()), float(d[fid].max()), float(d.mean())])
                info["ehm_vs_smplx2020_mm"] = {"head_mean": float(np.mean([e[0] for e in errs])), "head_max": float(np.max([e[1] for e in errs])),
                                               "all_mean": float(np.mean([e[2] for e in errs]))}
            tr = []
            for k in fj["test"]:
                t = to_dev(F.collate1(fbd.target_info(0, k)), dev)
                torch.cuda.synchronize(); t0 = time.perf_counter(); r = render(av, t); torch.cuda.synchronize(); tr.append(time.perf_counter() - t0)
                torchvision.io.write_png(u8(r), str(od / f"{k}.png"))
        else:                                                                                      # fb_sprint
            sf = c["source_frame"]; tf = [int(k.split("_")[1]) for k in fj["test"]]
            ori = original_frames(c, [sf] + tf); Hm = metric_H(c)
            if not warm: fitter.fit(ori[sf])                                                     # warm-up (not timed)
            rs = fitter.fit(ori[sf]); shp = rs["shape"]
            with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                key = fj["source"]; SAF.write_tracked(Path(td), name, key, rs)
                cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl)
                src = to_dev(F.collate1(fbd.source_info(0, key)), dev)
            if not warm: build(src); warm = True
            av, ms = build(src)
            info["recon_ms"] = ms + rs["times"]["total"] * 1000; info["recon_ms_model"] = ms; info["sprint_fit_ms"] = rs["times"]["total"] * 1000
            info["sprint_fit_times_s"] = rs["times"]; info["recon_input"] = "first frame (SPRINT fit, own box / crop / matte)"
            tr, tfit = [], []
            for k, f in zip(fj["test"], tf):
                r_ = fitter.fit(ori[f], iters=0, shape_lock=shp); tfit.append(r_["times"]["total"])
                rr = dict(r_["record"]); rr["smplx_coeffs"] = r_["raw_smplx"]
                comp = F.compact_record(rr, shp); sm = {kk: torch.as_tensor(v.copy()) for kk, v in comp["smplx"].items()}
                sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
                RT = sm["camera_RT_params"]; M4 = torch.eye(4); M4[:3, :4] = RT; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
                t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
                     "invtanfov": comp["invtanfov"]}
                t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
                t = to_dev(F.collate1(t), dev)
                torch.cuda.synchronize(); t0 = time.perf_counter(); r = render(av, t); torch.cuda.synchronize(); tr.append(time.perf_counter() - t0)
                W = S512 @ Hm[f] @ np.linalg.inv(r_["M"].astype(np.float64)) @ np.linalg.inv(S512)
                img = u8(r).permute(1, 2, 0).numpy()
                wp = cv2.warpPerspective(img, W, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                torchvision.io.write_png(torch.from_numpy(wp).permute(2, 0, 1).contiguous(), str(od / f"{k}.png"))
            info["sprint_drive_s_per_frame"] = float(np.mean(tfit))
        info["render_ms_per_frame"] = float(np.mean(tr) * 1000); info["gpu"] = GPU
        (od / "run.json").write_text(json.dumps(info, indent=1)); print(json.dumps({"id": c["id"], **{k: v for k, v in info.items() if k != "sprint_fit_times_s"}}), flush=True)


main()
