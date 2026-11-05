"""PREREG phase 6 extra metrics for the step-2 rule and the phase-6 report (PSNR / SSIM / LPIPS come from score.py).
Per scored frame of E/renders/<method>/<dataset>/<clip>/ against E/gt512 (image x mask on black, 512 metric crop):
- unseen-region LPIPS: LPIPS-VGG of (render x U, GT x U), U = unseen/<key>.png written by replay.py (pixels whose SMPL-X surface had
  zero visibility in the source frame, under the frame's live pose, warped like the render); frames with U on >= 0.5 % of the image;
- CSIM: ArcFace (glintr100, onnx_arcface_torch) cosine of render vs GT, both aligned with the GT's SCRFD-10G 5 landmarks (score
  >= 0.5; frames without a detected face are skipped);
- tLPIPS: mean over consecutive scored frames of |LPIPS(R_t, R_t-1) - LPIPS(G_t, G_t-1)|;
- silhouette IoU: render silhouette (sil/<key>.png, depth > 0) vs the GT mask (base masks; ValBench-v1 masks for BEDLAM2).
Per clip = mean over its frames; pooled = mean over clips (weight 1, the phase-6 rule's pooling), per dataset and all.
The unseen masks depend only on the live pose and the source fit, identical for every arm of a clip: --mask-from takes them from
one method's run (e.g. the uv arm, for the k0 arm that predates the masks).
Usage: python p6_metrics.py --methods p6_k0,p6_uv_p2_t0.3_notavg --split ../worker6/p6_val.json --subset ../worker6/p6_tuning.json
       --E <E6> --tag tuning --mask-from p6_uv_p2_t0.3_notavg"""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--methods", required=True); ap.add_argument("--subset", default="")
ap.add_argument("--split", required=True); ap.add_argument("--E", required=True); ap.add_argument("--tag", default="full")
ap.add_argument("--mask-from", default="", help="method whose unseen/<key>.png masks are used (default: each method's own)")
ap.add_argument("--vb1", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/valbench_v1")
ap.add_argument("--antelope", default="/home/ubx858/VACE/models/VACE-Annotators/face/models/antelopev2")
ap.add_argument("--scrfd-py", default="/home/ubx858/miniconda3/envs/vace/lib/python3.10/site-packages/insightface/model_zoo/scrfd.py")
ap.add_argument("--min-unseen", type=float, default=0.005); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); E = Path(a.E)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "phase3_dnt")); os.chdir(ROOT)
import onnx_arcface_torch as AF                                                                       # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval(); arc = AF.OnnxArcFace().to(dev).eval()


def scrfd():
    import onnxruntime as ort
    spec = importlib.util.spec_from_file_location("scrfd_mod", a.scrfd_py); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; f = str(Path(a.antelope) / "scrfd_10g_bnkps.onnx")
    d = M.SCRFD(f, session=ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])); d.prepare(-1, input_size=(640, 640), det_thresh=0.5)
    return d


DET = scrfd()
img = lambda p: torch.from_numpy(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).to(dev).permute(2, 0, 1)[None].float() / 255.0   # noqa: E731
gray = lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)                                                                          # noqa: E731


def gt_mask(ds, clip, k):
    for p in (E / "base" / ds / clip / "masks" / f"{k}.png", Path(a.vb1) / "base" / ds / clip / "masks" / f"{k}.png"):
        if p.exists(): return gray(p) > 127
    return cv2.imread(str(E / "gt512" / ds / clip / f"{k}.png")).max(-1) > 0                         # fallback: non-black GT pixels


def faces(ds, clip, keys):
    """SCRFD on the GT crops, cached per clip (identical for every method)."""
    cp = E / "reports" / "p6_face_lmk" / f"{ds}__{clip}.npz"
    if cp.exists():
        z = np.load(cp); return {str(k): (kp if s > 0 else None) for k, s, kp in zip(z["keys"], z["score"], z["kps"])}
    S, K = [], []
    for k in keys:
        bb, kp = DET.detect(cv2.imread(str(E / "gt512" / ds / clip / f"{k}.png")), input_size=(640, 640))
        if len(bb):
            i = int(np.argmax(bb[:, 4] * (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]))); S.append(float(bb[i, 4])); K.append(kp[i])
        else:
            S.append(0.0); K.append(np.full((5, 2), np.nan, np.float32))
    cp.parent.mkdir(parents=True, exist_ok=True); np.savez(cp, keys=np.array(keys), score=np.array(S, np.float32), kps=np.array(K, np.float32))
    return {k: (kp if s > 0 else None) for k, s, kp in zip(keys, S, K)}


@torch.no_grad()
def emb(x, kps5):
    th = AF.align_theta(kps5, x.shape[-1])[None]; return arc(arc.preprocess(AF.crops(x, th)))[0]


@torch.no_grad()
def clip_metrics(m, c, fk, gcache):
    ds, name = c["dataset"], c["clip"]; keys = json.loads((E / "base" / ds / name / "frames.json").read_text())["test"]
    rd = E / "renders" / m / ds / name; ud = E / "renders" / (a.mask_from or m) / ds / name / "unseen"
    if not all((rd / f"{k}.png").exists() for k in keys): return None
    un, cs, iou, tl = [], [], [], []; prev = None
    for k in keys:
        R, G = img(rd / f"{k}.png"), img(E / "gt512" / ds / name / f"{k}.png")
        if (ud / f"{k}.png").exists():
            U = torch.from_numpy(gray(ud / f"{k}.png") > 127).to(dev)[None, None].float()
            if float(U.mean()) >= a.min_unseen: un.append(float(lp(R * U, G * U).mean()))
        if fk.get(k) is not None:
            if ("E", k) not in gcache: gcache[("E", k)] = emb(G, fk[k])
            cs.append(float((emb(R, fk[k]) * gcache[("E", k)]).sum()))
        sp = rd / "sil" / f"{k}.png"
        if sp.exists():
            s, g = gray(sp) > 127, gt_mask(ds, name, k); iou.append(float((s & g).sum()) / max(float((s | g).sum()), 1.0))
        if prev is not None:
            gk = ("T", prev[0], k)
            if gk not in gcache: gcache[gk] = float(lp(G, prev[2]).mean())
            tl.append(abs(float(lp(R, prev[1]).mean()) - gcache[gk]))
        prev = (k, R, G)
    mean = lambda v: float(np.mean(v)) if v else None                                              # noqa: E731
    return {"dataset": ds, "n": len(keys), "unseen_lpips": mean(un), "n_unseen": len(un), "csim": mean(cs), "n_face": len(cs),
            "tlpips": mean(tl), "sil_iou": mean(iou)}


split = json.loads(Path(a.split).read_text())["clips"]
if a.subset: keep = set(json.loads(Path(a.subset).read_text())["clips"]); split = [c for c in split if c["id"] in keep]
res = {m: {} for m in a.methods.split(",")}
for c in split:
    keys = json.loads((E / "base" / c["dataset"] / c["clip"] / "frames.json").read_text())["test"]; fk = faces(c["dataset"], c["clip"], keys); gc = {}
    for m in res:
        r = clip_metrics(m, c, fk, gc)
        if r is not None: res[m][c["id"]] = r
for m, per in res.items():
    agg = {}
    for part in ("dress4d", "bedlam2", "ubody", "all"):
        cs = [p for p in per.values() if part == "all" or p["dataset"] == part]
        if not cs: continue
        agg[part] = {"clips": len(cs)}
        for k in ("unseen_lpips", "csim", "tlpips", "sil_iou"):
            v = [p[k] for p in cs if p[k] is not None]; agg[part][k] = float(np.mean(v)) if v else None; agg[part][f"clips_{k}"] = len(v)
    out = {"method": m, "tag": a.tag, "subset": a.subset or None, "mask_from": a.mask_from or m, "aggregate_clip_mean": agg, "per_clip": per}
    (E / "scores").mkdir(parents=True, exist_ok=True); (E / "scores" / f"{m}_{a.tag}_extra.json").write_text(json.dumps(out, indent=1))
    print(m, a.tag, json.dumps(agg), flush=True)
