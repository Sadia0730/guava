"""Check of scan_geom.py / the dress4d_scan cache before pilot D uses it (train + val rows; no model training).
(1) alignment: scan silhouette rendered in the row's camera (512) vs the target matte (IoU), also for a 2x zoomed target;
(2) label semantics: per scan label, the SMPL-X joint group of the nearest refit-mesh vertex (head / feet / torso+arms / legs);
(3) garment offset statistics: distance of scan vertices to the refit SMPL-X surface per label (cm, and in units of GUAVA's
    local face scale of the nearest face, the unit of the UV Gaussians' local_pos and of threshold_local_xyz);
(4) rendering time for a 6-target batch; overlays (image | scan normal | labels | scan - SMPL-X depth)."""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data"); ap.add_argument("--scan-root", default="/dev/shm/ubx858_guava_fb/data/dress4d_scan")
ap.add_argument("--n", type=int, default=24); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(HERE))
import numpy as np, torch, torchvision
import fb_common as F
import scan_geom as SG
from fb_common import to_dev
os.chdir(F.ROOT); dev = torch.device("cuda:0")
from utils.graphics_utils import compute_face_orientation
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

cfg = F.get_cfg(); _, inf, _, _ = F.setup_a20(dev); NV = F.NV
faces = inf.ehm.smplx.faces_tensor; faces = faces[(faces < NV).all(1)]
jdom = inf.ehm.smplx.lbs_weights[:NV].argmax(1)
GROUP = {"head": [12, 15, 22, 23, 24], "feet": [7, 8, 10, 11], "legs": [1, 2, 4, 5], "torso_arms": [0, 3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21]}
gid = torch.full((NV,), -1, device=dev)
for gi, js in enumerate(GROUP.values()): gid[torch.isin(jdom, torch.tensor(js, device=dev))] = gi
store = SG.ScanStore(a.scan_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
idx = json.loads((Path(a.data_root) / "dress4d" / "index.json").read_text())["clips"]
have = [c for c in idx if (Path(a.scan_root) / c["name"] / "meta.json").exists()]
rng = np.random.default_rng(0); pick = [have[i] for i in rng.choice(len(have), size=min(a.n, len(have)), replace=False)]
clips = [c for c in F.load_clips_parallel([(str(Path(a.data_root) / "dress4d" / c["name"]), None, True) for c in pick], workers=8) if "keys" in c]
fbd = F.FBData(cfg, clips)
rows, vis, dist = [], [], {l: [] for l in SG.LABELS}
grp_count = {l: np.zeros(len(GROUP)) for l in SG.LABELS}
for ci, c in enumerate(clips):
    k = c["keys"][int(rng.integers(len(c["keys"])))]
    for zoom in (False, True):
        t = fbd.target_info_zoom(ci, k, np.random.default_rng(ci), 2.0, 0.0) if zoom else fbd.target_info(ci, k)
        if t is None: continue
        t = to_dev(F.collate1(t), dev); itf = F.batch_invtanfov(t, 1, dev)
        V, Fc, L = store.get(c["name"], k); sc = (torch.from_numpy(V).to(dev), torch.from_numpy(Fc).to(dev), torch.from_numpy(L).to(dev))
        Ns, Ds, Ls = SG.render_scans([sc], t["w2c_cam"], itf)
        sil = Ls >= 0; m = t["mask"][:, :1] > 0.5
        row = {"clip": c["name"], "key": k, "zoom": zoom, "iou_scan_matte": float((sil & m).sum() / (sil | m).sum().clamp(min=1)),
               "scan_in_matte": float((sil & m).sum() / sil.sum().clamp(min=1)), "matte_in_scan": float((sil & m).sum() / m.sum().clamp(min=1))}
        if not zoom:
            with torch.no_grad():
                Vb = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][:, :NV]
            Vs = sc[0][None]
            # (2) label -> joint group of the nearest SMPL-X vertex; (3) distance to the SMPL-X surface
            d1 = knn_points(Vs, Vb, K=1); nn = d1.idx[0, :, 0]
            pts = sample_points_from_meshes(Meshes(verts=[Vb[0]], faces=[faces]), 200000)
            d2 = knn_points(Vs, pts, K=1).dists[0, :, 0].sqrt()
            _, fs = compute_face_orientation(Vb, faces, return_scale=True); fsv = torch.zeros(NV, device=dev).index_reduce_(0, faces.flatten(), fs[0].repeat_interleave(3), "amax", include_self=False)
            for l in SG.LABELS:
                sel = sc[2] == l
                if sel.any():
                    g = gid[nn[sel]]; grp_count[l] += np.bincount(g[g >= 0].cpu().numpy(), minlength=len(GROUP))
                    dist[l].append(torch.stack([d2[sel], d2[sel] / fsv[nn[sel]].clamp(min=1e-4)], 1).cpu().numpy())
        rows.append(row)
        if len(vis) < 8 * 4:
            lab_rgb = torch.tensor([[0.9, 0.75, 0.6], [0.3, 0.2, 0.1], [0.2, 0.2, 0.2], [0.9, 0.2, 0.2], [0.2, 0.3, 0.9], [0.2, 0.8, 0.3]], device=dev)
            lc = torch.where(Ls >= 0, Ls, torch.zeros_like(Ls))[0, 0]; lv = lab_rgb[lc].permute(2, 0, 1) * (Ls[0] >= 0)
            edge = (sil.float() - torch.nn.functional.max_pool2d(-sil.float(), 3, 1, 1).neg()).abs()[0]
            ov = t["image"][0] * t["mask"][0] * 0.7 + torch.cat([edge, edge * 0, edge * 0]) * 0.8
            vis += [ov.clamp(0, 1), (Ns[0] * 0.5 + 0.5) * (Ls[0] >= 0), lv, (Ds[0] - Ds[0][Ds[0] > 0].min() if (Ds[0] > 0).any() else Ds[0]).clamp(0, 1).repeat(3, 1, 1)]
# timing: 6 targets
sc6, w6, i6 = [], [], []
for ci, c in enumerate(clips[:6]):
    k = c["keys"][0]; t = to_dev(F.collate1(fbd.target_info(ci, k)), dev); V, Fc, L = store.get(c["name"], k)
    sc6.append((torch.from_numpy(V).to(dev), torch.from_numpy(Fc).to(dev), torch.from_numpy(L).to(dev))); w6.append(t["w2c_cam"]); i6.append(F.batch_invtanfov(t, 1, dev))
w6, i6 = torch.cat(w6), torch.cat(i6)
for _ in range(2): SG.render_scans(sc6, w6, i6)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5): SG.render_scans(sc6, w6, i6)
torch.cuda.synchronize(); ms = (time.time() - t0) / 5 * 1e3
torchvision.utils.save_image(torch.stack(vis), out / "scan_check.jpg", nrow=4, padding=2, pad_value=1.0)
summ = {"n_rows": len(rows), "render_ms_6_targets": ms}
for z in (False, True):
    rr = [r for r in rows if r["zoom"] == z]
    summ["zoom" if z else "full"] = {k: {"mean": float(np.mean([r[k] for r in rr])), "min": float(np.min([r[k] for r in rr]))} for k in ("iou_scan_matte", "scan_in_matte", "matte_in_scan")}
summ["labels"] = {}
for l, nm in SG.LABELS.items():
    if not dist[l]: continue
    d = np.concatenate(dist[l]); g = grp_count[l] / max(1, grp_count[l].sum())
    summ["labels"][f"{l}"] = {"assumed": nm, "n_vertices": int(len(d)), "nearest_joint_group": {gn: round(float(x), 3) for gn, x in zip(GROUP, g)},
                             "dist_cm_p50_p90_p99": [round(float(np.percentile(d[:, 0], q)) * 100, 2) for q in (50, 90, 99)],
                             "dist_face_units_p50_p90_p99": [round(float(np.percentile(d[:, 1], q)), 2) for q in (50, 90, 99)]}
(out / "scan_check.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1)); print(json.dumps(summ, indent=1))
