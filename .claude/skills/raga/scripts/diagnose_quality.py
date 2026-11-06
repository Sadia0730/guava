"""Why does the RAGA avatar look the way it does? Splits the render-vs-camera error into its likely causes.

Uses the live script's own code (imported, not copied), so the avatar, crop and render are exactly the live ones:
  1. Best case: the reference frame rendered from its own refined fit (the avatar was built from this image).
  2. Pose: sampled live frames rendered from SPRINT's raw pose (what live shows) and from a 150-step refined pose of the
     same frame (fitter.fit, shape locked). The difference = the cost of the live pose error.
  3. View: frames grouped by how far the body has turned from the reference (unseen sides are guessed).
  4. Where: PSNR inside the face, hand and body regions (DWPose keypoints), sharpness (Laplacian energy) and colour
     (brightness, saturation) of the render against the camera, inside the person.
Ground truth as in score_replays.py: the camera frame in the 512 render crop x StyleMatte alpha.

usage: diagnose_quality.py --session REC [--every 9] --out OUT       (guava env python; ~2-4 min; heat-guarded)
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path(os.environ.get("RAGA_PKG", "/data/GUAVA/guava_live"))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True); ap.add_argument("--every", type=int, default=9, help="every N-th recorded frame")
ap.add_argument("--max-cpu", type=float, default=92.0); ap.add_argument("--max-gpu", type=float, default=88.0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
REC = Path(args.session).resolve(); OUT = Path(args.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heat_guard import HeatGuard  # noqa: E402

hg = HeatGuard(args.max_cpu, args.max_gpu)

# the live script as a module (its argparse sees these arguments; main() does not run)
sys.argv = ["live_sprint_guavafb.py", "--input", str(REC), "--worker", "off", "--no-display"]
spec = importlib.util.spec_from_file_location("raga_live", PKG / "phase3_dnt" / "live" / "live_sprint_guavafb.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
import torch  # noqa: E402
sys.path.insert(0, str(PKG))
from main.metrics import psnr, ssim  # noqa: E402
from submodules.lpipsPyTorch import LPIPS  # noqa: E402

dev = L.dev; SAF = L.SAF; fitter = L.fitter; lp = LPIPS("vgg", "0.1").to(dev).eval()
S = np.diag([0.5, 0.5, 1.0])
tens = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(255.0).contiguous().to(dev)  # noqa: E731


def to_u8(rgb):
    return (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def regions(kp, sc, M, alpha512):
    """Masks in 512 render px: person (alpha > 0.5), face, hands (DWPose keypoint boxes), body = person - face - hands."""
    A = S @ M; pts = (np.c_[kp, np.ones(len(kp))] @ A.T)[:, :2]
    person = alpha512 > 0.5; m = {"person": person}
    for name, idx, grow in (("face", np.arange(24, 92), 0.15), ("hands", np.r_[92:113], 0.30), ("hands2", np.r_[113:134], 0.30)):
        g = idx[sc[idx] >= SAF.KP_THR]; box = np.zeros_like(person)
        if len(g) >= 4:
            (x0, y0), (x1, y1) = pts[g].min(0), pts[g].max(0); w, h = max(x1 - x0, 12), max(y1 - y0, 12); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = w * (1 + grow), h * (1 + grow)
            box[max(0, int(cy - h / 2)):max(0, int(cy + h / 2)), max(0, int(cx - w / 2)):max(0, int(cx + w / 2))] = True
        m[name] = box & person
    m["hands"] = m["hands"] | m.pop("hands2")
    m["body"] = person & ~m["face"] & ~m["hands"]
    return m


def masked_psnr(r, g, mask):
    if mask.sum() < 20: return float("nan")
    d = ((r.astype(np.float64) - g.astype(np.float64)) / 255.0) ** 2
    return float(10 * np.log10(1.0 / max(d[mask].mean(), 1e-10)))


def appearance(r, g, mask):
    """Sharpness ratio (render / camera Laplacian energy, eroded person) and brightness / saturation differences."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 50: return {}
    lap = lambda x: np.abs(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float64), cv2.CV_64F, ksize=3))[inner].mean()  # noqa: E731
    hsv_r, hsv_g = cv2.cvtColor(r, cv2.COLOR_RGB2HSV)[inner].astype(np.float64), cv2.cvtColor(g, cv2.COLOR_RGB2HSV)[inner].astype(np.float64)
    return {"sharpness_ratio": float(lap(r) / max(lap(g), 1e-6)),
            "brightness_diff_pct": float((hsv_r[:, 2].mean() - hsv_g[:, 2].mean()) / max(hsv_g[:, 2].mean(), 1e-6) * 100),
            "saturation_diff_pct": float((hsv_r[:, 1].mean() - hsv_g[:, 1].mean()) / max(hsv_g[:, 1].mean(), 1e-6) * 100)}


@torch.no_grad()
def score(av, rs, shp, cam):
    """Render the frame from SPRINT's raw pose and from the refined pose; score both against the camera."""
    M = np.asarray(rs["M"], np.float64); box = np.asarray(rs["box"], np.float64)
    al = torch.nn.functional.interpolate(torch.from_numpy(rs["alpha"]).float()[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()
    g = (cv2.warpAffine(cam, (S @ M)[:2], (512, 512), flags=cv2.INTER_LINEAR).astype(np.float64) * al[..., None]).round().astype(np.uint8)
    kp, sc = rs["record"]["dwpose_raw"]["keypoints"], rs["record"]["dwpose_raw"]["scores"]
    m = regions(kp, sc, M, al); out, imgs = {}, {"gt": g}
    for name, pose in (("raw", rs["raw_smplx"]), ("refined", rs["record"]["smplx_coeffs"])):
        r = to_u8(L.render(av, pose, box, M, shp)[0]); imgs[name] = r
        rt, gt = tens(r), tens(g)
        out[name] = {"psnr": float(psnr(rt, gt).mean()), "ssim": float(ssim(rt, gt)), "lpips": float(lp(rt, gt).mean()),
                     **{f"psnr_{k}": masked_psnr(r, g, v) for k, v in m.items()}, **appearance(r, g, m["person"])}
    ys = np.nonzero(m["person"].any(1))[0]; fy = np.nonzero(m["face"].any(1))[0]
    out["px"] = {"person_h_render": int(ys.max() - ys.min() + 1) if len(ys) else 0, "face_h_render": int(fy.max() - fy.min() + 1) if len(fy) else 0,
                 "person_h_camera": float(box[3] - box[1])}
    out["kp_err_px1024"] = {"raw": rs["diag"]["kp_err_px1024_before"], "refined": rs["diag"]["kp_err_px1024_after"]}
    return out, imgs


def tile(imgs, label):
    err = cv2.applyColorMap(np.clip(np.abs(imgs["raw"].astype(np.int16) - imgs["gt"].astype(np.int16)).mean(2) * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    row = np.concatenate([imgs["gt"], imgs["raw"], imgs["refined"], err], 1).copy()
    for k, t in enumerate(("camera (ground truth)", "avatar, live pose", "avatar, refined pose", "error, live pose")):
        cv2.putText(row, t, (8 + 512 * k, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(row, label, (8, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return row


ref = cv2.cvtColor(cv2.imread(str(REC / "reference.png")), cv2.COLOR_BGR2RGB)
av, shp, box_ref, rs_ref, ref_ms, _ = L.build_avatar(ref)
R0 = L.WC.root_cam({k: np.asarray(v) for k, v in rs_ref["raw_smplx"].items()})
rows, tiles = [], []
res_ref, im_ref = score(av, rs_ref, shp, ref)
rows.append({"frame": "reference", "view_deg": 0.0, **{f"{k}_{m}": v for k in ("raw", "refined") for m, v in res_ref[k].items()}, **res_ref["px"]})
tiles.append(tile(im_ref, "reference frame (the avatar was built from it)"))
meta = [json.loads(line) for line in open(REC / "frames.jsonl")][::args.every]
show = set(int(v) for v in np.linspace(0, len(meta) - 1, 5)[1:-1])
for n, e in enumerate(meta):
    hg.guard()
    cam = cv2.cvtColor(cv2.imread(str(REC / "frames" / f"{e['i']:06d}.jpg")), cv2.COLOR_BGR2RGB)
    rs = fitter.fit(cam, box=np.asarray(e["box"], np.float64), shape_lock=shp)
    view = float(np.degrees(np.linalg.norm(cv2.Rodrigues(L.WC.root_cam({k: np.asarray(v) for k, v in rs["raw_smplx"].items()}) @ R0.T)[0])))
    r, im = score(av, rs, shp, cam)
    rows.append({"frame": e["i"], "view_deg": view, **{f"{k}_{m}": v for k in ("raw", "refined") for m, v in r[k].items()}, **r["px"],
                 "kp_raw": r["kp_err_px1024"]["raw"], "kp_refined": r["kp_err_px1024"]["refined"]})
    if n in show: tiles.append(tile(im, f"frame {e['i']}, turned {view:.0f} deg from the reference"))
    print(f"{n + 1}/{len(meta)} frames", flush=True)

cv2.imwrite(str(OUT / "quality_grid.jpg"), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump(rows, open(OUT / "per_frame.json", "w"), indent=0)
live = [r for r in rows if r["frame"] != "reference"]
groups = {"reference frame (best case)": [rows[0]], "live frames, all": live,
          "  facing the camera (< 30 deg)": [r for r in live if r["view_deg"] < 30],
          "  turned 30-90 deg": [r for r in live if 30 <= r["view_deg"] < 90], "  turned away (>= 90 deg)": [r for r in live if r["view_deg"] >= 90]}
mean = lambda sel, k: float(np.nanmean([r.get(k, np.nan) for r in sel])) if sel else float("nan")  # noqa: E731
lines = ["| frames | n | pose | PSNR | SSIM | LPIPS | PSNR face | PSNR hands | PSNR body | sharpness vs camera | brightness | saturation |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, sel in groups.items():
    for p in ("raw", "refined"):
        lines.append(f"| {name} | {len(sel)} | {'live (SPRINT)' if p == 'raw' else 'refined'} | {mean(sel, p + '_psnr'):.2f} | {mean(sel, p + '_ssim'):.4f} | "
                     f"{mean(sel, p + '_lpips'):.4f} | {mean(sel, p + '_psnr_face'):.2f} | {mean(sel, p + '_psnr_hands'):.2f} | {mean(sel, p + '_psnr_body'):.2f} | "
                     f"{mean(sel, p + '_sharpness_ratio') * 100:.0f}% | {mean(sel, p + '_brightness_diff_pct'):+.1f}% | {mean(sel, p + '_saturation_diff_pct'):+.1f}% |")
lines += ["", f"sizes in the 512 render: person {mean(live, 'person_h_render'):.0f} px tall, face {mean(live, 'face_h_render'):.0f} px "
          f"(person {mean(live, 'person_h_camera'):.0f} px tall in the 1280x720 camera frame)",
          f"keypoint error vs DWPose (1024 crop px): live pose {mean(live, 'kp_raw'):.1f}, refined {mean(live, 'kp_refined'):.1f}",
          hg.summary(), f"grid: {OUT / 'quality_grid.jpg'}"]
(OUT / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
