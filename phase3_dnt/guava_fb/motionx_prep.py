"""GUAVA-FB / Motion-X, stage 1 (CPU): candidate pool + per-clip frame decoding, DWPose and the dataset labels converted to
SMPL-X 2020. No GPU.

Candidate pool (PREREG "GUAVA-FB", AVATAR_B_PLAN 2b):
- the 7 Motion-X++ subsets with labels and video on disk (idea400, haa500, kungfu, music, humman, animation, perform; fitness has
  no labels); a clip is full-body as in the audit (motionx_overlays.py): >= 3 stride-12 rows of the processed motionx_gt build
  and >= 80 % of them with all 55 projected joints inside the (image-clipped) mesh box;
- evaluation-set name check (People-Snapshot / NeuMan / X-Humans / UBC-Fashion names) -> report;
- performer proxy = source video (clip name minus its trailing `_clipN` parts; clips of one source video show the same
  performer); cap 1 clip per source video, so a subject (= source video) has exactly one clip and one shape (shape lock);
- stratified sample, seed 0: equal quota per subset (water-filled), total --n (1,500).
Per clip (worker): label frame n -> decoded frame n-1 (build_motionx_gt_crops.frame_map); frames every ~0.5 s
(step = round(0.5 fps) video frames over the labelled frames), cap 40 (evenly spread); DWPose (EHM-Tracker yolox_l +
dw-ll_ucoco_384 onnx, CPU; detection with the highest IoU to the label mesh box, IoU >= 0.3, else the frame has no
keypoints). Labels: SMPL-X v1.1 (OSX, 10 betas, 10 expr, absolute hands) -> SMPL-X 2020: body exact (v1.1 and 2020 share
v_template, J_regressor, posedirs, weights and shapedirs[:, :, :10]); ONE shape per clip = mean of the clip's labelled
betas10, zero-padded to 200; exp50 = W expr10 (expr_maps_to_smplx2020.npz v11_expr10_to_2020_exp50, the linear map of
build_face_labels_smplx2020.py, motionx_gt method 'linear'); jaw copied; eyes 0; camera = OSX crop camera K per frame
(crop_camera(bbox)); translation T = least-squares offset of the 2020 mesh (clip shape, no transl) onto the source mesh
(per-frame shape, label transl) in that camera.
Writes <work>/<subset>/<clip>/frames/<video_frame:06d>.jpg (q95, full resolution) and prep.npz; <work>/candidates.json."""
from __future__ import annotations
import argparse, json, os, re, sys, time, tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np, cv2, torch

sys.path.insert(0, "/home/ubx858/guava/third_party/PEAR/tools")
import build_motionx_gt_crops as B                                   # noqa: E402

EHMT = Path("/home/ubx858/guava/EHM-Tracker")
X2020 = Path("/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz")
EXPR_MAP = Path("/raid/ubx858/outputs/dnt/sprint/data/expr_maps_to_smplx2020.npz")
PROC = Path("/raid/ubx858/datasets/processed/motionx_gt")
SUBSETS = ["idea400", "haa500", "kungfu", "music", "humman", "animation", "perform"]
EVAL_PAT = re.compile(r"snapshot|people_?snap|neuman|x_?humans|xhuman|ubc|fashion|citron|seattle|jogging|parkinglot|"
                      r"female-\d|male-\d|\b000(16|17|18|19|20|25|27|28|34|35|41|58|87)\b|3dpw|downtown_|outdoors_", re.I)
STEP_S, CAP = 0.5, 40


def source_video(clip):
    return re.sub(r"(_clip\d+)+$", "", clip)


def candidates(n_total, seed=0):
    rep = {"subsets": {}}; pools = {}
    for s in SUBSETS:
        d = np.load(PROC / f"motionx_{s}.npz", allow_pickle=True)
        j2, box, clip = d["joints2d_img"], d["box_xyxy"], d["clip"]
        full = ((j2[:, :, 0] >= box[:, None, 0] - 1) & (j2[:, :, 0] <= box[:, None, 2] + 1) & (j2[:, :, 1] >= box[:, None, 1] - 1)
                & (j2[:, :, 1] <= box[:, None, 3] + 1)).all(1)
        cl, inv = np.unique(clip, return_inverse=True); n = np.bincount(inv); nf = np.bincount(inv, weights=full)
        hmed = np.array([np.median((box[inv == i, 3] - box[inv == i, 1])) for i in range(len(cl))])
        ok = (n >= 3) & (nf / n >= 0.8)
        names = [str(c) for c in cl[ok]]; hm = dict(zip(names, hmed[ok].tolist()))
        hits = [c for c in names if EVAL_PAT.search(c)]
        groups = {}
        for c in names: groups.setdefault(source_video(c), []).append(c)
        rng = np.random.default_rng(seed)
        one = sorted(str(rng.choice(sorted(v))) for _, v in sorted(groups.items()))      # 1 clip per source video
        pools[s] = one
        rep["subsets"][s] = {"clips_labelled": int(len(cl)), "full_body_clips": int(ok.sum()), "source_videos": len(groups),
                             "after_cap_1_per_video": len(one), "eval_name_pattern_hits": hits,
                             "median_body_h_px_full_body": float(np.median(hmed[ok]))}
        rep.setdefault("_hmed", {}).update({f"{s}/{c}": hm[c] for c in one})
    # equal quota, water-filled
    left, quota, alloc = n_total, None, {s: 0 for s in SUBSETS}
    rem = [s for s in SUBSETS]
    while rem and left > 0:
        q = left // len(rem); small = [s for s in rem if len(pools[s]) - alloc[s] <= q]
        if not small:
            for i, s in enumerate(sorted(rem)): alloc[s] += q + (1 if i < left - q * len(rem) else 0)
            left = 0; break
        for s in small: left -= len(pools[s]) - alloc[s]; alloc[s] = len(pools[s]); rem.remove(s)
    picks = []
    for s in SUBSETS:
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(pools[s], alloc[s], replace=False).tolist()) if alloc[s] < len(pools[s]) else list(pools[s])
        rep["subsets"][s]["sampled"] = len(sel); picks += [(s, c) for c in sel]
    rep["total_sampled"] = len(picks)
    return picks, rep


# ------------------------------------------------------------------------------------------------ worker
_W = {}


def _init(threads):
    torch.set_num_threads(1); cv2.setNumThreads(1); os.nice(0)
    import importlib.util, onnxruntime as ort, smplx

    def lp(name, p):
        sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    _W["det"] = lp("onnxdet", EHMT / "src/modules/dwpose/onnxdet.py"); _W["pose"] = lp("onnxpose", EHMT / "src/modules/dwpose/onnxpose.py")
    _W["sdet"] = ort.InferenceSession(str(EHMT / "pretrained/dwpose/yolox_l.onnx"), so, providers=["CPUExecutionProvider"])
    _W["spose"] = ort.InferenceSession(str(EHMT / "pretrained/dwpose/dw-ll_ucoco_384.onnx"), so, providers=["CPUExecutionProvider"])
    _W["src"] = B.smplx_model(B.V11.parent, 10, True).eval()
    _W["x20"] = smplx.SMPLX(model_path=str(X2020), gender="neutral", num_betas=10, use_pca=False, flat_hand_mean=True,
                            num_expression_coeffs=50).eval()
    _W["Wexp"] = np.load(EXPR_MAP)["v11_expr10_to_2020_exp50"].astype(np.float64)        # (50, 10)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


@torch.no_grad()
def work(job):
    subset, clip, raw_label, raw_video, cfg = job
    out = Path(cfg["work"]) / subset / clip; t0 = time.time()
    info = {"subset": subset, "clip": clip}
    try:
        frames, anns = B.parse_label(raw_label)
        vid = B.Video(raw_video, cfg["tmp"])
        try:
            vmap, status = B.frame_map(frames, anns, vid.count, 0)
            info.update(W=vid.W, H=vid.H, fps=vid.fps, video_frames=vid.count, labels=len(frames), status=status)
            if vmap is None: raise RuntimeError(status)
            step = max(1, int(round(STEP_S * (vid.fps if vid.fps > 0 else 25))))
            sel, last = [], -10 ** 9
            for i in range(len(frames)):
                if vmap[i] is not None and vmap[i] - last >= step: sel.append(i); last = vmap[i]
            if len(sel) > CAP: sel = [sel[int(round(q))] for q in np.linspace(0, len(sel) - 1, CAP)]
            ims = {}
            for i in sel:
                im = vid.frame(vmap[i])
                if im is None: break
                ims[i] = im
        finally:
            vid.close()
        sel = [i for i in sel if i in ims]
        if len(sel) < 4: raise RuntimeError(f"few_frames_{len(sel)}")
        # ---- labels -> SMPL-X 2020
        T = lambda k, idx: torch.as_tensor(np.array([anns[i]["smplx_params"][k] for i in idx], np.float32))
        z3 = torch.zeros(len(sel), 3)
        o = _W["src"](betas=T("shape", sel), global_orient=T("root_pose", sel), body_pose=T("body_pose", sel), left_hand_pose=T("lhand_pose", sel),
                      right_hand_pose=T("rhand_pose", sel), jaw_pose=T("jaw_pose", sel), leye_pose=z3, reye_pose=z3, expression=T("expr", sel),
                      transl=T("trans", sel))
        vsrc = o.vertices.numpy()
        beta = np.array([a["smplx_params"]["shape"] for a in anns], np.float64).mean(0).astype(np.float32)        # one shape / clip
        expr10 = T("expr", sel).numpy(); exp50 = (expr10.astype(np.float64) @ _W["Wexp"].T).astype(np.float32)
        o2 = _W["x20"](betas=torch.as_tensor(beta)[None].expand(len(sel), -1), global_orient=T("root_pose", sel), body_pose=T("body_pose", sel),
                       left_hand_pose=T("lhand_pose", sel), right_hand_pose=T("rhand_pose", sel), jaw_pose=T("jaw_pose", sel), leye_pose=z3, reye_pose=z3,
                       expression=torch.as_tensor(exp50))
        v20 = o2.vertices.numpy(); Tt = (vsrc - v20).mean(1)
        conv_err = np.linalg.norm(v20 + Tt[:, None] - vsrc, axis=-1)
        K = np.stack([B.crop_camera(anns[i]["bbox"]) for i in sel])
        uv = np.stack([B.project(K[j], vsrc[j]) for j in range(len(sel))])
        mbox = np.concatenate([uv.min(1), uv.max(1)], 1)
        # ---- DWPose
        out.mkdir(parents=True, exist_ok=True); (out / "frames").mkdir(exist_ok=True)
        kp = np.zeros((len(sel), 133, 2), np.float32); sc = np.zeros((len(sel), 133), np.float32); dbox = np.zeros((len(sel), 4), np.float32)
        dok = np.zeros(len(sel), bool); diou = np.zeros(len(sel), np.float32); ndet = np.zeros(len(sel), np.int32); td = 0.0
        for j, i in enumerate(sel):
            im = ims[i]; rgb = np.ascontiguousarray(im[..., ::-1]); t1 = time.time()
            boxes = _W["det"].inference_detector(_W["sdet"], rgb); ndet[j] = len(boxes)
            mb = np.clip(mbox[j], 0, [info["W"], info["H"], info["W"], info["H"]])
            if len(boxes):
                ious = [iou(b, mb) for b in boxes]; b = int(np.argmax(ious)); diou[j] = ious[b]
                if ious[b] >= 0.3:
                    k_, s_ = _W["pose"].inference_pose(_W["spose"], boxes[b][None], rgb)
                    kp[j], sc[j], dbox[j], dok[j] = k_[0], s_[0], boxes[b], True
            td += time.time() - t1
            cv2.imwrite(str(out / "frames" / f"{vmap[i]:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        P = lambda k: np.array([anns[i]["smplx_params"][k] for i in sel], np.float32)
        np.savez(out / "prep.npz", frame=np.array([frames[i] for i in sel], np.int32), video_frame=np.array([vmap[i] for i in sel], np.int32),
                 root=P("root_pose"), body=P("body_pose"), jaw=P("jaw_pose"), lhand=P("lhand_pose"), rhand=P("rhand_pose"), expr10=expr10,
                 exp50=exp50, beta10=beta, betas_src=P("shape"), transl_src=P("trans"), T=Tt.astype(np.float32), K=K.astype(np.float32),
                 bbox_label=np.array([anns[i]["bbox"] for i in sel], np.float32), mesh_box=mbox.astype(np.float32),
                 vsrc10=vsrc[:10].astype(np.float32), conv_err_mean_mm=(conv_err.mean(1) * 1000).astype(np.float32),
                 conv_err_max_mm=(conv_err.max(1) * 1000).astype(np.float32),
                 kp=kp, sc=sc, det_box=dbox, det_ok=dok, det_iou=diou, n_det=ndet, W=info["W"], H=info["H"], fps=info["fps"])
        info.update(ok=True, n_frames=len(sel), det_ok=int(dok.sum()), dwpose_s=round(td, 2), seconds=round(time.time() - t0, 2),
                    conv_err_mean_mm=float(conv_err.mean() * 1000))
    except Exception as e:                                                  # noqa: BLE001
        info.update(ok=False, error=repr(e)[:300], seconds=round(time.time() - t0, 2))
    (out.parent).mkdir(parents=True, exist_ok=True)
    (Path(cfg["work"]) / subset / f"{clip}.prep.json").write_text(json.dumps(info))
    return info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=40); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True); t0 = time.time()
    picks, rep = candidates(a.n, a.seed)
    rep_out = {k: v for k, v in rep.items() if k != "_hmed"}; rep_out["picks"] = [f"{s}/{c}" for s, c in picks]
    rep_out["median_body_h_px"] = {f"{s}/{c}": rep["_hmed"][f"{s}/{c}"] for s, c in picks}
    (a.work / "candidates.json").write_text(json.dumps(rep_out, indent=1))
    print(json.dumps({k: v for k, v in rep_out["subsets"].items()}, indent=1), flush=True)
    if a.only: picks = [p for p in picks if f"{p[0]}/{p[1]}" in set(a.only)]
    if a.limit: picks = picks[:a.limit]
    picks = [p for p in picks if not (a.work / p[0] / f"{p[1]}.prep.json").exists()]
    tmp = tempfile.mkdtemp(dir=str(a.work)); cfg = {"work": str(a.work), "tmp": tmp}
    by = {}
    for s, c in picks: by.setdefault(s, []).append(c)

    def gen():
        for s in SUBSETS:
            if s not in by: continue
            vz = B.VideoZip(s); lz, labels = B.label_zip(s)
            for c in vz.ordered([c for c in by[s] if c in vz.members and c in labels]):
                yield s, c, lz.read(labels[c]), vz.read(c), cfg
    n = 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.threads,)) as pool:
        for info in pool.imap_unordered(work, gen()):
            n += 1
            if n % 25 == 0 or n <= 5 or not info.get("ok"):
                print(n, len(picks), info.get("subset"), info.get("clip"), info.get("ok"), info.get("n_frames"), info.get("seconds"), info.get("error", ""),
                      f"{time.time() - t0:.0f}s", flush=True)
    os.rmdir(tmp) if not os.listdir(tmp) else None
    if not a.only and not a.limit: (a.work / "PREP_DONE").write_text(str(n))
    print("PREP_DONE", n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
