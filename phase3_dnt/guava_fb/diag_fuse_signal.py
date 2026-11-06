"""Phase 7 audit (user question 2026-11-06 "check bug or data range issues"): how much of the TRAINING SIGNAL can fusion-aware training
get? For the fused items logged in an arm's train_log.jsonl (source + support frames + 3 targets), compute per-texel visibility with the
SMPL-X mesh (as fb_fusion.frame_feats does) and report, per dataset: the share of each target's visible texels that the source does not
see (only there can fused content change the render), and how much of that share the support frames cover. Also the image value range
of the loaded frames. Descriptive; GPU for the rasterizer only."""
import argparse, json, collections, sys
from pathlib import Path
import numpy as np, torch
import fb_common as F

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data")
ap.add_argument("--max-items", type=int, default=120); a = ap.parse_args()
dev = torch.device("cuda:0"); cfg, inf, ren, _ = F.setup_a20(dev); inf.eval()
sm = inf.smplx; faces = sm.faces_tensor; fidx = sm.uvmap_f_idx.long(); umask = sm.uvmap_mask.bool(); nF = faces.shape[0]
rows = [json.loads(l) for l in open(Path(a.run) / "train_log.jsonl") if l.startswith('{"split": "train"')]
items = []
for r in rows:
    for n, p in zip(r["fuse_n_support"], r["pairs"]):
        if n > 0: items.append(p)
items = items[: a.max_items]; print("fused items", len(items), flush=True)
CL = {}
def clip(ds, name):
    k = (ds, name)
    if k not in CL:
        cl = F.load_clips_parallel([(str(Path(a.data_root) / ds / name), None, True)], workers=1); CL[k] = F.FBData(cfg, cl)
    return CL[k]
@torch.no_grad()
def vis_of(ds, key):
    cname, fk = key.split("/", 1); fbd = clip(ds, cname); b = F.to_dev(F.collate1(fbd.source_info(0, fk)), dev)
    itf = F.batch_invtanfov(b, 1, dev); V = inf.ehm(b["smplx_coeffs"], b["flame_coeffs"])["vertices"]
    p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, b["w2c_cam"], itf)
    fv = torch.zeros(nF, dtype=torch.bool, device=dev); fv[p2f[p2f >= 0].long() % nF] = True
    im = b["image"]; return umask & fv[fidx], (float(im.min()), float(im.max()), float(im[im > 0].mean()) if (im > 0).any() else 0.0)
acc = collections.defaultdict(list); rng = []
for i, p in enumerate(items):
    ds = p["source"]
    try:
        vs, r0 = vis_of(ds, p["src"]); rng.append(r0)
        vsup = torch.zeros_like(vs)
        for s in p["support"]: v, _ = vis_of(ds, s); vsup |= v
        for t, is_self in zip(p["tgt"], p["self"]):
            vt, _ = vis_of(ds, t); unseen = vt & ~vs; n_t = int(vt.sum()); n_u = int(unseen.sum()); n_c = int((unseen & vsup).sum())
            acc[ds].append((n_u / max(n_t, 1), n_c / max(n_u, 1), n_c / max(n_t, 1), bool(is_self)))
    except Exception as e:
        print("skip", p["src"], repr(e)[:120], flush=True)
    if i % 20 == 0: print("item", i, flush=True)
print(); print("image range over sources: min %.3f max %.3f mean-of-nonzero %.3f" % (min(r[0] for r in rng), max(r[1] for r in rng), float(np.mean([r[2] for r in rng]))))
print(); print("| dataset | targets | self-targets | target texels unseen by source | of those, covered by supports | fused-and-visible share of target |"); print("|---|---|---|---|---|---|")
allv = []
for ds, v in sorted(acc.items()):
    arr = np.array([x[:3] for x in v]); allv += v
    print(f"| {ds} | {len(v)} | {sum(x[3] for x in v)} | {100 * arr[:, 0].mean():.1f} % | {100 * arr[:, 1].mean():.1f} % | {100 * arr[:, 2].mean():.1f} % |")
arr = np.array([x[:3] for x in allv]); ns = np.array([x[:3] for x in allv if not x[3]])
print(f"| all | {len(allv)} | {sum(x[3] for x in allv)} | {100 * arr[:, 0].mean():.1f} % | {100 * arr[:, 1].mean():.1f} % | {100 * arr[:, 2].mean():.1f} % |")
print(f"| all, non-self targets | {len(ns)} | 0 | {100 * ns[:, 0].mean():.1f} % | {100 * ns[:, 1].mean():.1f} % | {100 * ns[:, 2].mean():.1f} % |")
