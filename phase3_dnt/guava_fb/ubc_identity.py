"""GUAVA-FB, UBC-Fashion identity-overlap check (PREREG "GUAVA-FB": ArcFace antelopev2 glintr100, a train video whose face
crops match an evaluation identity with cosine > 0.5 is removed from training).

Evaluation identities = the 100 official TEST videos + GART's 6 UBC videos (which are in the official TRAIN list; they are
excluded from training anyway and are treated as evaluation identities here).
Per video: 16 frames evenly spaced over the whole video; SCRFD-10G (antelopev2, CPU onnxruntime) face box + 5 landmarks on
the full frame; faces kept if det score >= 0.6, box >= 40 px and frontal (nose x between the two eyes, eye-distance ratio
check); the up to 6 highest-scoring kept faces -> ArcFace crops (norm_crop, 112 px) -> glintr100 (GPU, onnx_arcface_torch).
Video-pair similarity (registered): MAX cosine over all face-crop pairs. Also reported: cosine of the mean embeddings.
Output: <out>/ubc_identity.npz (per-video embeddings) and <out>/identity.json."""
import argparse, importlib.util, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

D = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/ubc_fashion")
GART6 = ["91+bCFG1jOS", "91Iegdp9HFS", "91fxYsir49S", "91QFEra7jDS", "91Ile3zLhMS", "91WvLcNpdzS"]
ANT = Path("/home/ubx858/VACE/models/VACE-Annotators/face/models/antelopev2")
SCRFD_PY = "/home/ubx858/miniconda3/envs/vace/lib/python3.10/site-packages/insightface/model_zoo/scrfd.py"
THR = 0.5
_det = None
CFG = {"n": 16, "score": 0.6, "minw": 40, "frontal": True}


def _init(threads, cfg=None):
    global _det
    if cfg:
        CFG.update(cfg)
    import onnxruntime as ort
    spec = importlib.util.spec_from_file_location("scrfd_mod", SCRFD_PY); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    p = str(ANT / "scrfd_10g_bnkps.onnx")
    _det = M.SCRFD(p, session=ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]))
    _det.prepare(-1, input_size=(640, 640), det_thresh=0.5)


def faces_of(args):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from onnx_arcface_torch import umeyama2d, ARCFACE_DST
    split, name = args
    cap = cv2.VideoCapture(str(D / split / f"{name}.mp4")); n = int(cap.get(7))
    want = set(np.linspace(0, n - 1, CFG["n"]).round().astype(int).tolist())
    found = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            bb, kp = _det.detect(fr, input_size=(640, 640))
            if len(bb):
                j = int(np.argmax(bb[:, 4] * (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
                s, b, k = float(bb[j, 4]), bb[j, :4], kp[j]
                w = float(b[2] - b[0]); le, re, nose = k[0], k[1], k[2]
                ed = float(np.linalg.norm(re - le))
                frontal = (min(le[0], re[0]) < nose[0] < max(le[0], re[0])) and ed > 0.25 * w
                if s >= CFG["score"] and w >= CFG["minw"] and (frontal or not CFG["frontal"]):
                    M = umeyama2d(k.astype(np.float64), ARCFACE_DST.astype(np.float64))
                    crop = cv2.warpAffine(fr, M, (112, 112), borderValue=0.0)[:, :, ::-1]   # RGB
                    found.append((s, i, w, crop))
        i += 1
    found.sort(key=lambda r: -r[0]); found = found[:6]
    return split, name, n, [(r[0], r[1], r[2]) for r in found], (np.stack([r[3] for r in found]) if found else np.zeros((0, 112, 112, 3), np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/ubc")
    ap.add_argument("--procs", type=int, default=24); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [("train", p.stem) for p in sorted((D / "train").glob("*.mp4"))] + [("test", p.stem) for p in sorted((D / "test").glob("*.mp4"))]
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.threads,)) as pool:
        res = pool.map(faces_of, jobs, chunksize=2)
    # retry for videos without a usable face: 96 frames, score >= 0.5, box >= 24 px, frontal check off
    retry_cfg = {"n": 96, "score": 0.5, "minw": 24, "frontal": False}
    redo = [(r[0], r[1]) for r in res if len(r[4]) == 0]
    if redo:
        with Pool(min(a.procs, len(redo)), initializer=_init, initargs=(a.threads, retry_cfg)) as pool:
            res2 = {(r[0], r[1]): r for r in pool.map(faces_of, redo, chunksize=1)}
        res = [res2.get((r[0], r[1]), r) for r in res]
    t_det = time.time() - t0
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from onnx_arcface_torch import OnnxArcFace
    net = OnnxArcFace().to(a.device).eval()
    emb = {}
    for split, name, n, meta, crops in res:
        if len(crops):
            x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(a.device) / 255.0
            emb[(split, name)] = net(net.preprocess(x)).cpu().numpy()
        else:
            emb[(split, name)] = np.zeros((0, 512), np.float32)
    info = {(s, nm): {"frames": n, "faces": meta} for s, nm, n, meta, _ in res}
    evals = [k for k in emb if k[0] == "test"] + [("train", g) for g in GART6]
    trains = [k for k in emb if k[0] == "train" and k[1] not in GART6]
    def sims(k1, k2):
        e1, e2 = emb[k1], emb[k2]
        if not len(e1) or not len(e2):
            return None, None
        mx = float((e1 @ e2.T).max())
        m1, m2 = e1.mean(0), e2.mean(0); mc = float(m1 @ m2 / np.linalg.norm(m1) / np.linalg.norm(m2))
        return mx, mc
    per_train, removed, pairs = {}, [], []
    for k in trains:
        best = (-1.0, None, None)
        for e in evals:
            mx, mc = sims(k, e)
            if mx is None:
                continue
            if mx > best[0]:
                best = (mx, e, mc)
            if mx > THR:
                pairs.append({"train": k[1], "eval": f"{e[0]}/{e[1]}", "max_cos": round(mx, 4), "mean_emb_cos": round(mc, 4)})
        per_train[k[1]] = {"n_faces": len(emb[k]), "max_cos": None if best[1] is None else round(best[0], 4),
                           "nearest_eval": None if best[1] is None else f"{best[1][0]}/{best[1][1]}", "mean_emb_cos_nearest": None if best[2] is None else round(best[2], 4)}
        if best[1] is not None and best[0] > THR:
            removed.append(k[1])
    # within-train duplicates (information only), all train incl. GART6
    tr_all = [k for k in emb if k[0] == "train"]
    dup = []
    for i in range(len(tr_all)):
        for j in range(i + 1, len(tr_all)):
            mx, mc = sims(tr_all[i], tr_all[j])
            if mx is not None and mx > THR:
                dup.append([tr_all[i][1], tr_all[j][1], round(mx, 4), round(mc, 4)])
    # within-train identity groups (connected components), information
    par = {k[1]: k[1] for k in tr_all}
    def fnd(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for u, v, *_ in dup:
        par[fnd(u)] = fnd(v)
    groups = {}
    for k in tr_all:
        groups.setdefault(fnd(k[1]), []).append(k[1])
    no_face = [k[1] for k in trains if len(emb[k]) == 0]
    rep = {"rule": "registered: train video removed iff max cosine over face-crop pairs with any evaluation identity > 0.5",
           "model": "antelopev2 glintr100 (onnx_arcface_torch), SCRFD-10G 5-point alignment (norm_crop)",
           "evaluation_identities": "100 official test videos + GART's 6 (official-train) videos",
           "faces_per_video": "16 evenly spaced frames, SCRFD score >= 0.6, box >= 40 px, frontal; best <= 6; videos with none: retry 96 frames, score >= 0.5, box >= 24 px, no frontal check",
           "n_retried": len(redo), "retried": [f"{x[0]}/{x[1]}" for x in redo],
           "n_train_checked": len(trains), "n_train_no_face": len(no_face), "train_no_face": no_face,
           "n_eval_no_face": sum(1 for e in evals if len(emb[e]) == 0), "eval_no_face": [f"{e[0]}/{e[1]}" for e in evals if len(emb[e]) == 0],
           "n_removed": len(removed), "removed": sorted(removed), "pairs_over_thr": sorted(pairs, key=lambda r: -r["max_cos"]),
           "max_cos_hist": np.histogram([v["max_cos"] for v in per_train.values() if v["max_cos"] is not None], bins=[-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])[0].tolist(),
           "max_cos_hist_bins": [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
           "removed_if_mean_emb_rule": sum(1 for v in per_train.values() if v["mean_emb_cos_nearest"] is not None and v["mean_emb_cos_nearest"] > THR),
           "within_train_pairs_over_thr": len(dup), "within_train_identity_groups_size_gt1": sorted([sorted(g) for g in groups.values() if len(g) > 1], key=len, reverse=True),
           "n_within_train_identities": len(groups),
           "per_train": per_train, "within_train_pairs": dup, "time_s": {"detect_cpu": round(t_det, 1), "total": round(time.time() - t0, 1)}}
    (out / "identity.json").write_text(json.dumps(rep, indent=1))
    np.savez(out / "ubc_identity.npz", **{f"{s}/{n}": e for (s, n), e in emb.items()})
    print(json.dumps({k: rep[k] for k in ("n_train_checked", "n_train_no_face", "n_eval_no_face", "n_removed", "within_train_pairs_over_thr", "n_within_train_identities", "max_cos_hist", "removed_if_mean_emb_rule", "time_s")}, indent=1))


if __name__ == "__main__":
    main()
