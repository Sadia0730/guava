"""Diagnostic A (PREREG "GUAVA-FB improvement phase 1"): where do the errors of each method come from, on ValBench (validation only).
Per target pixel (512 metric crop): GT SMPL-X 2020 mesh rasterised (pytorch3d) -> body region of the hit face (head, hands, torso,
arms, legs, feet), "off_body" (GT mask > 0.5, no body surface), "spill_out" (render > 0.02 outside mask and body); source visibility
of the hit surface (vertices of the faces visible in the SOURCE frame's depth-tested rasterisation at 1024; a face is "seen" if >= 2
of its vertices are). Error per pixel = mean over RGB of the squared / absolute error of the 8-bit render vs E/gt512, SSIM map as
GUAVA's metric (11 px Gaussian). Per frame: PSNR / SSIM / LPIPS-VGG (GUAVA's functions), source->target view angle (body forward
axes), motion (root-aligned body-joint distance, m). Coverage: miss = mask > 0.5 & render <= 0.02 (all channels), spill = mask < 0.5
& render > 0.02. Writes <out>/<method>.json (per dataset: class table with pixel share, PSNR over the class, L1, SSIM, share of the
image's squared error; view / motion bins; coverage) and per-frame rows for paired comparisons.
Usage: python diag_regions.py --E <root> --split <json> --methods fb_gt,fb_gt_raw,gart,fb_sprint --out <dir>"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torchvision

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--methods", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); a.split = str(Path(a.split).resolve())
E, OUT = Path(a.E), Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent / "guava_fb")); os.chdir(ROOT)
from main.metrics import psnr, ssim, create_window                                                  # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
from smplx_lean import SMPLXLean                                                                     # noqa: E402
from pytorch3d.structures import Meshes                                                              # noqa: E402
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes                                # noqa: E402
import torch.nn.functional as Fn                                                                     # noqa: E402

dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval()
X = SMPLXLean(str(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), 200, dtype=torch.float32).to(dev)
FACES = torch.as_tensor(X.faces, device=dev)
W8 = np.load(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["weights"].argmax(1)
JREG = {"head": [15, 22, 23, 24], "torso": [0, 3, 6, 9, 12, 13, 14], "arms": [16, 17, 18, 19], "hands": [20, 21] + list(range(25, 55)),
        "legs": [1, 2, 4, 5], "feet": [7, 8, 10, 11]}
REG = list(JREG); vreg = np.zeros(len(W8), np.int64)
for i, r in enumerate(REG): vreg[np.isin(W8, JREG[r])] = i
fv = torch.as_tensor(vreg, device=dev)[FACES]                                                      # (F, 3)
FREG = torch.mode(fv, 1).values                                                                    # region per face
CLASSES = REG + ["off_body", "spill_out"]
WIN = create_window(11, 3).to(dev)
split = json.loads(Path(a.split).read_text())["clips"]
methods = a.methods.split(",")


def ssim_map(r, g):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = Fn.conv2d(r, WIN, padding=5, groups=3), Fn.conv2d(g, WIN, padding=5, groups=3)
    s1 = Fn.conv2d(r * r, WIN, padding=5, groups=3) - mu1 ** 2; s2 = Fn.conv2d(g * g, WIN, padding=5, groups=3) - mu2 ** 2
    s12 = Fn.conv2d(r * g, WIN, padding=5, groups=3) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean(1)[0]


def raster(V, K, size):
    """V (Nv, 3) camera-frame (OpenCV) -> pix_to_face (size, size) at `size` px (K given for 512)."""
    s = size / 512.0; u = V[:, 0] / V[:, 2] * K[0, 0] * s + K[0, 2] * s; v = V[:, 1] / V[:, 2] * K[1, 1] * s + K[1, 2] * s
    nd = torch.stack([1 - 2 * u / size, 1 - 2 * v / size, V[:, 2]], 1)
    p2f, _, _, _ = rasterize_meshes(Meshes([nd], [FACES]), image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    return p2f[0, :, :, 0]


def mesh(sx, i):
    pose = torch.cat([sx["global_orient"][i:i + 1], sx["body_pose"][i:i + 1].reshape(1, 63), sx["jaw_pose"][i:i + 1], sx["leye_pose"][i:i + 1],
                      sx["reye_pose"][i:i + 1], sx["left_hand_pose"][i:i + 1].reshape(1, 45), sx["right_hand_pose"][i:i + 1].reshape(1, 45)], 1)
    V, J = X(pose, sx["betas"][None], sx["expression"][i:i + 1]); return V[0] + sx["transl"][i], J[0] + sx["transl"][i]


def Rmat(go):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(np.asarray(go, np.float64)).as_matrix()


load = lambda p: (torchvision.io.read_image(str(p))[:3].float() / 255.0)[None].contiguous().to(dev)     # noqa: E731
acc = {m: {} for m in methods}; rows = {m: [] for m in methods}
for c in split:
    ds, name = c["dataset"], c["clip"]; B = E / "base" / ds / name
    if not all((E / "renders" / m / ds / name / "run.json").exists() for m in methods): print("skip (renders missing)", c["id"]); continue
    fj = json.loads((B / "frames.json").read_text()); z = np.load(B / "smplx2020.npz"); K = torch.as_tensor(np.load(B / "cameras.npz")["K"], device=dev)
    keys = list(z["keys"]); sx = {k: torch.as_tensor(z[k], device=dev) for k in z.files if k != "keys"}
    si = keys.index(fj["source"]); Vs, Js = mesh(sx, si)
    p2f_s = raster(Vs, K[si], 1024); visf = torch.unique(p2f_s[p2f_s >= 0])
    vis_v = torch.zeros(Vs.shape[0], device=dev); vis_v[FACES[visf].reshape(-1)] = 1
    face_seen = (vis_v[FACES].sum(1) >= 2)
    Rsm = Rmat(z["global_orient"][si]); fwd_s = Rsm @ np.array([0, 0, 1.0])
    Jsl = (Js[:22] - Js[0]) @ torch.as_tensor(Rsm, device=dev, dtype=torch.float32)                 # joints in the body frame
    for k in fj["test"]:
        i = keys.index(k); Vt, Jt = mesh(sx, i); p2f = raster(Vt, K[i], 512)
        Rt = Rmat(z["global_orient"][i]); Jtl = (Jt[:22] - Jt[0]) @ torch.as_tensor(Rt, device=dev, dtype=torch.float32)
        motion = float((Jtl[1:] - Jsl[1:]).norm(dim=1).mean()); ang = float(np.degrees(np.arccos(np.clip(fwd_s @ (Rt @ np.array([0, 0, 1.0])), -1, 1))))
        g = load(E / "gt512" / ds / name / f"{k}.png")
        msk = torchvision.io.read_image(str(B / "masks" / f"{k}.png"))[0].float().to(dev) / 255.0 > 0.5
        hit = p2f >= 0; freg = torch.where(hit, FREG[p2f.clamp(min=0)], torch.full_like(p2f, -1)); seen = torch.where(hit, face_seen[p2f.clamp(min=0)], torch.zeros_like(hit))
        for m in methods:
            r = load(E / "renders" / m / ds / name / f"{k}.png")
            e2 = ((r - g) ** 2).mean(1)[0]; e1 = (r - g).abs().mean(1)[0]; sm = ssim_map(r, g); cov = (r > 0.02).any(1)[0]
            cls = freg.clone(); cls[(~hit) & msk] = len(REG); cls[(~hit) & (~msk) & cov] = len(REG) + 1
            sse = float(e2.sum()); A = acc[m].setdefault(ds, {"cls": {}, "frames": 0, "sse": 0.0, "miss": 0, "spill": 0, "mask": 0})
            A["frames"] += 1; A["sse"] += sse; A["miss"] += int((msk & ~cov).sum()); A["spill"] += int((~msk & cov).sum()); A["mask"] += int(msk.sum())
            for ci, cn in enumerate(CLASSES):
                sel = cls == ci
                parts = [(cn, sel)] + ([(cn + "|seen", sel & seen), (cn + "|unseen", sel & ~seen)] if ci < len(REG) else [])
                for nm, vsel in parts:
                    n = int(vsel.sum())
                    if n == 0: continue
                    q = A["cls"].setdefault(nm, [0, 0.0, 0.0, 0.0])
                    q[0] += n; q[1] += float(e2[vsel].sum()); q[2] += float(e1[vsel].sum()); q[3] += float(sm[vsel].sum())
            with torch.no_grad():
                rows[m].append({"id": c["id"], "key": k, "ds": ds, "psnr": float(psnr(r, g).mean()), "ssim": float(ssim(r, g)), "lpips": float(lp(r, g).mean()),
                                "angle": ang, "motion": motion, "miss": int((msk & ~cov).sum()), "spill": int((~msk & cov).sum()), "mask": int(msk.sum()),
                                "unseen_px": int((hit & ~seen & msk).sum()), "body_px": int((hit & msk).sum())})
    print("done", c["id"], flush=True)
for m in methods:
    out = {"method": m, "classes": CLASSES, "datasets": {}}
    for ds, A in acc[m].items():
        npx = 512 * 512 * A["frames"]
        tab = {cn: {"px_share": q[0] / npx, "psnr": float(10 * np.log10(1.0 / max(q[1] / q[0], 1e-12))), "l1": q[2] / q[0], "ssim": q[3] / q[0],
                    "sse_share": q[1] / A["sse"]} for cn, q in sorted(A["cls"].items())}
        R = [r for r in rows[m] if r["ds"] == ds]; ang = np.array([r["angle"] for r in R]); mo = np.array([r["motion"] for r in R])
        bins = {"angle": {}, "motion": {}}
        for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 181)):
            s = [r for r, x in zip(R, ang) if lo <= x < hi]
            if s: bins["angle"][f"{lo}-{hi}"] = {"n": len(s), **{k: float(np.mean([r[k] for r in s])) for k in ("psnr", "ssim", "lpips")}}
        t1, t2 = np.quantile(mo, [1 / 3, 2 / 3])
        for nm, (lo, hi) in (("low", (-1, t1)), ("mid", (t1, t2)), ("high", (t2, 9))):
            s = [r for r, x in zip(R, mo) if lo < x <= hi]
            if s: bins["motion"][nm] = {"n": len(s), "max_m": float(hi), **{k: float(np.mean([r[k] for r in s])) for k in ("psnr", "ssim", "lpips")}}
        out["datasets"][ds] = {"frames": A["frames"], "classes": tab, "bins": bins, "miss_frac_of_mask": A["miss"] / A["mask"],
                               "spill_frac_of_mask": A["spill"] / A["mask"], "psnr": float(np.mean([r["psnr"] for r in R])),
                               "ssim": float(np.mean([r["ssim"] for r in R])), "lpips": float(np.mean([r["lpips"] for r in R]))}
    out["frames"] = rows[m]
    (OUT / f"{m}.json").write_text(json.dumps(out, indent=1))
    print(m, json.dumps({ds: {k: v for k, v in d.items() if k in ("psnr", "ssim", "lpips", "miss_frac_of_mask", "spill_frac_of_mask")} for ds, d in out["datasets"].items()}), flush=True)
