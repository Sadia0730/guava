"""GUAVA-FB / Motion-X, stage 2 (GPU): registered alignment filter on the dataset labels, SPRINT joint-video refit of the
clips that fail it, filter again, write the kept clips in the GUAVA-FB tracked format (FORMAT.md).

Input: motionx_prep.py work dir (<work>/<subset>/<clip>/{frames/*.jpg, prep.npz} + <clip>.prep.json).
Per clip:
 (a) dataset labels (SMPL-X 2020 conversion of prep.npz; one shape per clip) in the OSX crop camera K (per frame,
     fx = fy): perspective crop (FORMAT): virtual camera rotated (minimal rotation R_v) onto the ray through the centre of
     the DWPose person box (detection matched to the label mesh, prep), homography K' R_v K^-1, side S = 1.1 x long box side,
     1024 px, principal point at the crop centre (pixel-index 511.5, GUAVA's convention), f' = 1024 f / S,
     invtanfov = 2 f / S; global_pose <- R_v global_pose, translation about the pelvis T' = R_v (J0 + T) - J0.
     StyleMatte on the crop; registered filter = fit_ubc.frame_filter on fit_ubc.silhouettes (pytorch3d) exactly as for UBC
     (limb reading: max over 5 limbs of the limb's mean keypoint error < 3 %; the single-keypoint-max reading is stored too);
     clip kept iff >= 80 % of ALL selected frames pass (frames without a matched person count as failures).
 (b) EVERY clip (so (a) and (b) pass rates are known for all clips; the refit is USED only when (a) fails):
     video_fit.VideoFit (= fit_ubc.VideoFit, the shared joint-video refiner: SPRINT from scratch, camera input focal = W,
     shape locked to the median, one shared shape, temporal smoothing, 300 Adam iterations, batched on the GPU) on the frames
     with a matched person; its own frame_filter results; same 80 % rule.
Front / back flip check (ubc_flip_frames.py rule, video_fit.flip_check): flipped frames are not written.
Kept clips: ONLY the passing, non-flipped frames are written (O/data/motionx/<subset>__<clip>/), label_source "dataset" or
"sprint_refit" in every record and in the diagnostics. Records as fit_ubc.write_clip (body_crop "original" = the 1024 crop,
head / hand crops from the projected SMPL-X head / hand vertices, persp_crop info, invtanfov, view "0"); dataset-label clips
also get gt_proj.npz (<= 10 rows: SOURCE v1.1 mesh, per-frame label shape and translation, OSX camera, into the crop).
Diagnostics: O/runs/motionx/clips/<subset>__<clip>/{fit.json, overlay.jpg, params_dataset.npz, params_refit.npz}."""
from __future__ import annotations
import argparse, json, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np, cv2, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import sprint_avatar_fit as SAF                                         # noqa: E402
import video_fit as VF                                                  # noqa: E402
from fit_ubc import shaded                                              # noqa: E402

O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
CROP, PART = 1024, 512
DIAG = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def persp_crop_K(box, K):
    l, t, r, b = [float(v) for v in box]; S = 1.1 * max(r - l, b - t)
    d = np.linalg.inv(K) @ np.array([(l + r) / 2, (t + b) / 2, 1.0]); d /= np.linalg.norm(d)
    Rv = VF.rot_to_z(d); f = float(K[0, 0]); fp = f * CROP / S
    Kp = np.array([[fp, 0, (CROP - 1) / 2], [0, fp, (CROP - 1) / 2], [0, 0, 1.0]])
    return Kp @ Rv @ np.linalg.inv(K), Rv, 2 * f / S, S


def norm_px(px):                       # pixel-index coords (centres at integers) -> normalised [0, 1] (VF.proj convention)
    return (np.asarray(px) + 0.5) / CROP


class Ctx:
    def __init__(self, dev, iters):
        self.fitter = SAF.SprintFitter(VF.CKPT, dev, iters=0, dwpose_threads=2, cuda_graph=False)
        assert "sprint_scratch" in self.fitter.ckpt
        self.vf = VF.VideoFit(self.fitter, iters); self.dev = self.fitter.dev; self.L = self.fitter.lean; self.faces = self.vf.faces
        self.faces_t = torch.from_numpy(self.faces).to(self.dev)


def load_clip(work, subset, clip):
    d = dict(np.load(work / subset / clip / "prep.npz"))
    imgs = [cv2.imread(str(work / subset / clip / "frames" / f"{v:06d}.jpg"))[..., ::-1].copy() for v in d["video_frame"]]
    kp134, sc134 = [], []
    for j in range(len(imgs)):
        k, s = SAF.to_openpose134(d["kp"][j], d["sc"][j]); kp134.append(k); sc134.append(s)
    return d, imgs, np.stack(kp134), np.stack(sc134)


@torch.no_grad()
def label_path(C, d, imgs, kp134, sc134):
    dev = C.dev; B = len(imgs); R_ = lambda x: SAF.rodrigues(torch.as_tensor(np.asarray(x, np.float32), device=dev))
    Hm, Rv, inv, crops, alphas, tgt = [], [], [], [], [], []
    for j in range(B):
        K = d["K"][j].astype(np.float64)
        box = d["det_box"][j] if d["det_ok"][j] else d["mesh_box"][j]
        h, r, iv, S = persp_crop_K(box, K); Hm.append(h); Rv.append(r); inv.append(iv)
        crop = cv2.warpPerspective(imgs[j], h, (CROP, CROP), flags=cv2.INTER_LINEAR); crops.append(crop)
        ct = torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0
        alphas.append(C.fitter.matting(ct).clamp(0, 1).cpu().numpy())
        tgt.append(norm_px(VF.hom(h, kp134[j])))
    Rv_t = torch.as_tensor(np.stack(Rv), dtype=torch.float32, device=dev)
    rots = torch.cat([Rv_t[:, None] @ R_(d["root"])[:, None], R_(d["body"].reshape(B, 21, 3)), R_(d["jaw"])[:, None],
                      torch.eye(3, device=dev).expand(B, 2, 3, 3), R_(d["lhand"].reshape(B, 15, 3)), R_(d["rhand"].reshape(B, 15, 3))], 1)
    beta = torch.zeros(200, device=dev); beta[:10] = torch.as_tensor(d["beta10"], device=dev)
    exp = torch.as_tensor(d["exp50"], device=dev)
    # translation: T from prep (smplx package, same model); check the lean layer reproduces the source mesh (unrotated)
    rots0 = rots.clone(); rots0[:, 0] = R_(d["root"])
    V0, _, J0 = VF.lean_batch(C.L, rots0, beta, exp)
    T = torch.as_tensor(d["T"], device=dev)
    nv = min(10, B); vs = torch.as_tensor(d["vsrc10"][:nv], device=dev)
    conv_err = float((V0[:nv] + T[:nv, None] - vs).norm(dim=-1).mean() * 1000)
    Tt = (Rv_t @ (J0 + T)[..., None])[..., 0] - J0
    Vv, P, _ = VF.lean_batch(C.L, rots, beta, exp)
    inv_t = torch.as_tensor(inv, dtype=torch.float32, device=dev)
    vimg = VF.proj(Vv + Tt[:, None], inv_t).cpu().numpy(); uv = VF.proj(P + Tt[:, None], inv_t).cpu().numpy()
    sils = VF.silhouettes(torch.as_tensor(vimg, device=dev), (Vv + Tt[:, None])[..., 2], C.faces_t)
    filt = []
    for j in range(B):
        fac = VF.facing_of(rots[j, 0].cpu().numpy())
        if not d["det_ok"][j]:
            filt.append({"pass": False, "pass_a": False, "pass_b": False, "no_person": True, "facing": fac}); continue
        filt.append({**VF.frame_filter(sils[j], alphas[j], uv[j], tgt[j], sc134[j], d["det_box"][j], Hm[j]), "facing": fac})
    # gt_proj: source mesh (v1.1, per-frame label shape + transl, OSX camera) into the crop; and our record's projection error
    gt = []
    for j in range(nv):
        g = VF.hom(Hm[j], (lambda p: p[:, :2] / p[:, 2:])(d["vsrc10"][j].astype(np.float64) @ d["K"][j].astype(np.float64).T))
        gt.append(g.astype(np.float32))
    rec_err = [float(np.linalg.norm((vimg[j] * CROP - 0.5) - gt[j], axis=1).mean() / 2) for j in range(nv)]     # px at 512
    vcam = (Vv + Tt[:, None]).cpu().numpy()
    return {"rots": rots, "beta": beta, "exp": exp, "Tt": Tt, "inv": inv, "Hm": Hm, "Rv": Rv, "crops": crops, "alphas": alphas, "vimg": vimg,
            "uv": uv, "tgt": tgt, "filt": filt, "gt_proj": gt, "focal": [float(k[0, 0]) for k in d["K"]], "vcam": vcam,
            "check": {"lean_vs_source_mesh_mm_first10": conv_err, "record_vs_gt_proj_px512_mean": float(np.mean(rec_err)),
                      "record_vs_gt_proj_px512_max": float(np.max(rec_err)),
                      "fx_fy_rel_diff_max": float(np.max(np.abs(d["K"][:, 0, 0] - d["K"][:, 1, 1]) / d["K"][:, 0, 0]))}}


def annotate(f, kp1024, sc):
    """single-keypoint-max reading + UBC flip check, in place."""
    if "mesh_out_frac" in f:
        f["pass_single"] = bool(f["pass_a"] and f["n_kp05"] >= VF.KP_MIN and f["max_single_kp_err_rel"] < VF.KP_ERR_MAX)
        fl = VF.flip_check(np.asarray(kp1024), np.asarray(sc), f["facing"]) if np.isfinite(f["facing"]) else None
        f["flip_checked"] = fl is not None; f["flipped"] = bool(fl)
    else:
        f.update(pass_single=False, flip_checked=False, flipped=False)
    return f


def summ(filt, B):
    s = lambda k: float(np.sum([bool(f.get(k)) for f in filt]) / B)
    return {"pass_frac": s("pass"), "pass_a_frac": s("pass_a"), "pass_b_frac": s("pass_b"), "pass_frac_single_kp_max": s("pass_single"),
            "kept_single_kp_max": s("pass_single") >= VF.CLIP_PASS, "flip_checked": int(sum(f.get("flip_checked", False) for f in filt)),
            "flipped": int(sum(f.get("flipped", False) for f in filt))}


def save_params(path, R, keys):
    import roma
    aa = roma.rotmat_to_rotvec(R["rots"].reshape(-1, 3, 3)).reshape(R["rots"].shape[0], 55, 3).cpu().numpy()
    np.savez_compressed(path, keys=np.array(keys), pose_aa=aa.astype(np.float32), beta=R["beta"].cpu().numpy(), exp=R["exp"].cpu().numpy(),
                        Tt=R["Tt"].cpu().numpy(), invtanfov=np.asarray(R["inv"], np.float32), H_o2c=np.stack(R["Hm"]).astype(np.float32),
                        R_virtual=np.stack(R["Rv"]).astype(np.float32))


def write_clip(C, out_dir, name, R, keys, frames_meta, keep, label_source, W, H):
    import lmdb, roma, torchvision
    out_dir.mkdir(parents=True, exist_ok=True)
    aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3)).cpu().numpy().astype(np.float32)
    rots = R["rots"]; recs = {}; z50 = np.zeros(50, np.float32)
    Mlr = np.diag([SAF.BODY_LR / CROP, SAF.BODY_LR / CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
    hv, lv, rv = C.vf.head_v.cpu().numpy(), C.vf.lh_v.cpu().numpy(), C.vf.rh_v.cpu().numpy()
    wk = [b for b in range(len(keys)) if keep[b]]
    for b in wk:
        key = keys[b]; fm = frames_meta[b]
        t = (DIAG @ R["Tt"][b].cpu().numpy()).astype(np.float32); RT = np.concatenate([DIAG, t[:, None]], 1).astype(np.float32)
        smplx = {"global_pose": aa(rots[b, 0])[0], "body_pose": aa(rots[b, 1:22]), "jaw_pose": aa(rots[b, 22]).reshape(1, 3),
                 "left_hand_pose": aa(rots[b, 25:40]), "right_hand_pose": aa(rots[b, 40:55]), "eye_pose": np.zeros((2, 3), np.float32),
                 "exp": R["exp"][b].cpu().numpy().astype(np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}
        vi = R["vimg"][b] * CROP
        def part(vidx, sl, sc):
            M = SAF.box_crop_matrix(SAF.landmark_box(vi[vidx], sl), PART, sc)
            return {"M_o2c": M, "M_c2o": np.linalg.inv(M).astype(np.float32)}
        crops = {"head_crop": part(hv, 1.1, 1.3), "left_hand_crop": part(lv, 1.5, 1.3), "right_hand_crop": part(rv, 1.5, 1.3)}
        recs[key] = {"smplx_coeffs": smplx,
                     "flame_coeffs": {"expression_params": z50, "jaw_params": np.zeros(3, np.float32), "pose_params": np.zeros(3, np.float32),
                                      "neck_pose_params": np.zeros(3, np.float32), "eye_pose_params": np.zeros(6, np.float32),
                                      "eyelid_params": np.zeros(2, np.float32), "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
                     "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
                     "persp_crop": {"H_o2c": np.asarray(R["Hm"][b], np.float32), "R_virtual": np.asarray(R["Rv"][b], np.float32),
                                    "focal_px_orig": float(R["focal"][b]), "orig_wh": [int(W), int(H)], "frame_index": int(fm["video_frame"]),
                                    "label_frame": int(fm["label_frame"])},
                     "dwpose_raw": {"keypoints": fm["kp"].astype(np.float32), "scores": fm["sc"].astype(np.float32), "bbox": np.asarray(fm["box"], np.float32),
                                    "keypoints_crop1024": (np.asarray(R["tgt"][b]) * CROP - 0.5).astype(np.float32)},
                     "invtanfov": float(R["inv"][b]), "view": "0", "label_source": label_source, **crops}
    with open(out_dir / "optim_tracking_ehm.pkl", "wb") as fh: pickle.dump(recs, fh)
    with open(out_dir / "id_share_params.pkl", "wb") as fh: pickle.dump(SAF.id_share(R["beta"].cpu().numpy()), fh)
    ks = [keys[b] for b in wk]
    (out_dir / "videos_info.json").write_text(json.dumps({name: {"frames_num": len(ks), "frames_keys": ks}}, indent=1))
    env = lmdb.open(str(out_dir / "img_lmdb"), map_size=1 << 32); txn = env.begin(write=True)
    for b in wk:
        enc = torchvision.io.encode_jpeg(torch.from_numpy(np.ascontiguousarray(R["crops"][b])).permute(2, 0, 1).contiguous(), quality=95)
        txn.put(f"{keys[b]}/body_image".encode(), bytes(enc.numpy().tobytes()))
        a = torch.from_numpy((np.clip(R["alphas"][b], 0, 1) * 255).round().astype(np.uint8))[None].expand(3, -1, -1).contiguous()
        txn.put(f"{keys[b]}/body_mask".encode(), bytes(torchvision.io.encode_jpeg(a, quality=95).numpy().tobytes()))
    txn.commit(); env.close()
    if label_source == "dataset":
        rows = [b for b in wk if b < len(R["gt_proj"])][:10]
        np.savez(out_dir / "gt_proj.npz", frame_key=np.array([keys[b] for b in rows]), v2d_1024=np.stack([R["gt_proj"][b] for b in rows]) if rows else np.zeros((0, 10475, 2), np.float32),
                 note=np.array("SOURCE Motion-X SMPL-X v1.1 mesh (per-frame label betas / expr / transl, OSX crop camera) projected into the "
                               "1024 perspective crop, pixel-index coordinates (centres at integers, principal point 511.5)"))
    return ks


def _mesh_tile(C, R, b, f, size=320):
    if not (np.isfinite(R["vimg"][b]).all() and np.isfinite(R["vcam"][b]).all()):
        t = np.zeros((size, size, 3), np.uint8); cv2.putText(t, "non-finite fit", (4, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return t
    mesh = shaded(R["crops"][b], R["vimg"][b], C.faces, R["vcam"][b], size=size)
    mesh = cv2.addWeighted(cv2.resize(np.ascontiguousarray(R["crops"][b]), (size, size)), 0.35, mesh, 0.65, 0)
    sil = np.zeros((size, size), np.uint8); cv2.fillPoly(sil, list((R["vimg"][b][C.faces] * size).round().astype(np.int32)), 255)
    cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(mesh, cs, -1, (255, 255, 0), 1)
    txt = "no person" if f.get("no_person") else f"out={f['mesh_out_frac']:.3f} kp={f['max_kp_err_rel']:.3f} {'PASS' if f['pass'] else 'FAIL'}"
    cv2.putText(mesh, txt, (4, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return mesh


def overlay(C, path, title, Ra, Rb, n=4, size=320, rows=("img", "a", "b")):
    """columns = up to n frames; rows: image crop (dataset-label perspective crop), dataset-label mesh, SPRINT refit mesh."""
    B = len(Ra["crops"]); sel = sorted(set(int(round(q)) for q in np.linspace(0, B - 1, min(n, B))))
    bmap = {j: i for i, j in enumerate(Rb["frame_idx"])} if Rb is not None else {}
    cols = []
    for j in sel:
        col = []
        if "img" in rows: col.append(cv2.resize(np.ascontiguousarray(Ra["crops"][j]), (size, size), interpolation=cv2.INTER_AREA))
        if "a" in rows: col.append(_mesh_tile(C, Ra, j, Ra["filt"][j], size))
        if "b" in rows:
            col.append(_mesh_tile(C, Rb, bmap[j], Rb["filt"][bmap[j]], size) if j in bmap else np.zeros((size, size, 3), np.uint8))
        cols.append(np.concatenate(col, 0))
    g = np.concatenate(cols, 1); cv2.putText(g, title[:100], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def process(C, work, subset, clip, out_root, diag_root):
    name = f"{subset}__{clip}"; dd = diag_root / name; dd.mkdir(parents=True, exist_ok=True)
    T = {}; t0 = time.perf_counter()
    d, imgs, kp134, sc134 = load_clip(work, subset, clip); B = len(imgs); W, H = int(d["W"]), int(d["H"])
    T["load"] = time.perf_counter() - t0; t0 = time.perf_counter()
    keys = [f"frame_{int(v):06d}" for v in d["video_frame"]]
    meta = [{"video_frame": int(d["video_frame"][j]), "label_frame": int(d["frame"][j]), "kp": kp134[j], "sc": sc134[j],
             "box": d["det_box"][j]} for j in range(B)]
    info = {"name": name, "subset": subset, "clip": clip, "n_frames": B, "n_person": int(d["det_ok"].sum()), "W": W, "H": H, "fps": float(d["fps"])}
    Ra = label_path(C, d, imgs, kp134, sc134); torch.cuda.synchronize(); T["label_filter"] = time.perf_counter() - t0
    for j in range(B): annotate(Ra["filt"][j], Ra["tgt"][j] * CROP - 0.5, sc134[j])
    pa = [f["pass"] for f in Ra["filt"]]
    info["a"] = {**summ(Ra["filt"], B), "kept": bool(np.mean(pa) >= VF.CLIP_PASS), "check": Ra["check"], "frames": Ra["filt"]}
    save_params(dd / "params_dataset.npz", Ra, keys)
    R, keep, src = None, None, None
    if info["a"]["kept"]:
        R, keep, src = Ra, [f["pass"] and not f.get("flipped") for f in Ra["filt"]], "dataset"
    Rb = None; t0 = time.perf_counter()
    idx = [j for j in range(B) if d["det_ok"][j]]
    if len(idx) >= 3 and len(idx) >= VF.CLIP_PASS * B:
        V = {"name": name, "W": W, "H": H, "frames": [{"idx": int(d["video_frame"][j]), "rgb": imgs[j], "box": d["det_box"][j].astype(np.float64),
                                                      "kp": kp134[j], "sc": sc134[j]} for j in idx]}
        Rb = C.vf.fit(V); torch.cuda.synchronize(); T.update({f"refit_{k}": v for k, v in Rb["times"].items()})
        inv_t = torch.as_tensor(Rb["inv"], dtype=torch.float32, device=C.dev)
        with torch.no_grad():
            Vv, _, _ = VF.lean_batch(C.L, Rb["rots"], Rb["beta"], Rb["exp"]); Rb["vcam"] = (Vv + Rb["Tt"][:, None]).cpu().numpy()
        Rb["frame_idx"] = idx; Rb["focal"] = [float(W)] * len(idx); Rb["tgt"] = [np.asarray(x) for x in Rb["tgt"]]
        fin = [bool(np.isfinite(Rb["vimg"][i]).all() and np.isfinite(Rb["vcam"][i]).all()) for i in range(len(idx))]
        for i, ok_ in enumerate(fin):
            if not ok_: Rb["filt"][i].update(nonfinite=True, **{k: False for k in ("pass", "pass_a", "pass_b")})
        for i, j in enumerate(idx): annotate(Rb["filt"][i], Rb["tgt"][i] * CROP, sc134[j])
        pb = [f["pass"] for f in Rb["filt"]]; frac = float(np.sum(pb) / B)
        info["b"] = {**summ(Rb["filt"], B), "kept": frac >= VF.CLIP_PASS, "frame_idx": idx, "diag": Rb["diag"],
                     "sprint_shape_std10": Rb["sprint_shape_std"], "nonfinite_frames": int(len(fin) - sum(fin)), "frames": Rb["filt"]}
        save_params(dd / "params_refit.npz", Rb, [keys[j] for j in idx])
        if R is None and info["b"]["kept"]:
            R, keep, src = Rb, [f["pass"] and not f.get("flipped") for f in Rb["filt"]], "sprint_refit"; keys = [keys[j] for j in idx]; meta = [meta[j] for j in idx]
    else:
        info["b"] = {"pass_frac": 0.0, "kept": False, "skipped": f"only {len(idx)} / {B} frames with a matched person"}
    T["refit"] = time.perf_counter() - t0
    info["kept"] = R is not None; info["label_source"] = src
    t0 = time.perf_counter()
    if R is not None:
        ks = write_clip(C, out_root / name, name, R, keys, meta, keep, src, W, H); info["written_frames"] = len(ks)
        info["flipped_passing_frames_dropped"] = int(sum(f["pass"] and bool(f.get("flipped")) for f in R["filt"]))
    T["write"] = time.perf_counter() - t0
    overlay(C, dd / "overlay.jpg", f"{name} kept={src} a={info['a']['pass_frac']:.2f} b={info['b']['pass_frac']:.2f}", Ra, Rb)
    info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    info["times_s"] = T
    (dd / "fit.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--follow", action="store_true", help="wait for prep results until PREP_DONE")
    ap.add_argument("--out", type=Path, default=O / "data" / "motionx"); ap.add_argument("--diag", type=Path, default=O / "runs" / "motionx" / "clips")
    a = ap.parse_args()
    picks = json.loads((a.work / "candidates.json").read_text())["picks"]
    if a.only: picks = [p for p in picks if p in set(a.only)]
    picks = picks[a.shard::a.nshard]
    C = Ctx(a.device, a.iters); todo = list(picks); t_start = time.time(); n = 0
    while todo:
        progressed = False
        for p in list(todo):
            s, c = p.split("/", 1); pj = a.work / s / f"{c}.prep.json"
            if (a.diag / f"{s}__{c}" / "fit.json").exists(): todo.remove(p); continue
            if not pj.exists(): continue
            todo.remove(p); progressed = True
            pinfo = json.loads(pj.read_text())
            if not pinfo.get("ok"):
                dd = a.diag / f"{s}__{c}"; dd.mkdir(parents=True, exist_ok=True)
                (dd / "fit.json").write_text(json.dumps({"name": f"{s}__{c}", "subset": s, "clip": c, "kept": False, "prep_error": pinfo.get("error")}))
                continue
            try:
                info = process(C, a.work, s, c, a.out, a.diag); n += 1
                print(n, p, "a", f"{info['a']['pass_frac']:.2f}", "b", f"{info.get('b', {}).get('pass_frac', -1):.2f}", info["label_source"],
                      {k: round(v, 2) for k, v in info["times_s"].items()}, f"{time.time() - t_start:.0f}s", flush=True)
            except Exception as e:                                  # noqa: BLE001
                dd = a.diag / f"{s}__{c}"; dd.mkdir(parents=True, exist_ok=True)
                (dd / "fit.json").write_text(json.dumps({"name": f"{s}__{c}", "subset": s, "clip": c, "kept": False, "error": repr(e), "trace": traceback.format_exc()}))
                print("ERROR", p, repr(e), flush=True)
            torch.cuda.empty_cache()
        if not progressed:
            if not a.follow or (a.work / "PREP_DONE").exists(): break
            time.sleep(20)
    print("FIT_DONE", n, f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
