"""Check of FBData.target_info_zoom (before any training uses it): (1) GUAVA's projection with the zoomed camera equals the
homography image of the original 1024-crop projection (< 0.01 px); (2) the projected SMPL-X body sits inside the warped
matte as well as in the unzoomed target (mask precision); overlays for a visual check."""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data"); ap.add_argument("--sources", default="dress4d,ubc")
ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(HERE))
import cv2, numpy as np, torch, torchvision
import fb_common as F
os.chdir(F.ROOT); dev = torch.device("cuda:0")
cfg = F.get_cfg(); _, inf, _, _ = F.setup_a20(dev); NV = F.NV; faces = inf.ehm.smplx.faces_tensor.cpu().numpy(); faces = faces[(faces < NV).all(1)]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = []
for s in a.sources.split(","):
    ents = [e for e in json.loads((Path(a.data_root) / s / "index.json").read_text())["clips"] if e.get("split") in ("train", "val")]
    rng = np.random.default_rng(0); pick = [ents[i] for i in rng.choice(len(ents), size=min(a.n, len(ents)), replace=False)]
    clips = [c for c in F.load_clips_parallel([(str(Path(a.data_root) / s / e["name"]), None, True) for e in pick], workers=8) if "keys" in c]
    fbd = F.FBData(cfg, clips)
    for ci, c in enumerate(clips):
        k = c["keys"][int(rng.integers(len(c["keys"])))]
        for mode, hp in (("cloth", 0.0), ("head", 1.0)):
            tz = fbd.target_info_zoom(ci, k, np.random.default_rng([ci, int(hp)]), 2.0, hp, with_M=True)
            if tz is None: res.append({"source": s, "clip": c["name"], "key": k, "mode": mode, "skipped": True}); continue
            t0 = fbd.target_info(ci, k)
            b0 = {kk: (v[None] if torch.is_tensor(v) else v) for kk, v in t0.items()}
            with torch.no_grad():
                V = inf.ehm(F.to_dev(F.collate1(t0["smplx_coeffs"]), dev), F.to_dev(F.collate1(t0["flame_coeffs"]), dev))["vertices"][:, :NV]
            itf0 = torch.tensor([float(t0["invtanfov"])], device=dev); itfz = torch.tensor([float(tz["invtanfov"])], device=dev)
            p_full, _ = F.project_px(V, t0["w2c_cam"][None].to(dev).float(), itf0, size=1024)
            p_zoom, _ = F.project_px(V, tz["w2c_cam"][None].to(dev).float(), itfz, size=512)
            Minv = np.linalg.inv(tz["zoom_M"].numpy()); ph = np.c_[p_full[0].cpu().numpy(), np.ones(NV)] @ Minv.T; ph = ph[:, :2] / ph[:, 2:]
            err = np.abs(ph - p_zoom[0].cpu().numpy()).max()
            def prec(px, msk):
                body = np.zeros(msk.shape, np.uint8); cv2.fillPoly(body, [np.round(px[f]).astype(np.int32) for f in faces], 1)
                b = body > 0; return float((b & msk).sum() / max(b.sum(), 1)), b
            m0 = t0["mask"][0].numpy() > 0.5; mz = tz["mask"][0].numpy() > 0.5
            p0, _ = prec(F.project_px(V, t0["w2c_cam"][None].to(dev).float(), itf0, size=512)[0][0].cpu().numpy(), m0)
            pz, bz = prec(p_zoom[0].cpu().numpy(), mz)
            res.append({"source": s, "clip": c["name"], "key": k, "mode": mode, "max_proj_err_px": float(err),
                        "mask_prec_unzoomed": p0, "mask_prec_zoomed": pz, "itf": float(t0["invtanfov"]), "itf_zoom": float(tz["invtanfov"]),
                        "head_box_zoom": tz["head_box"].tolist()})
            if len(res) <= 16:
                img = (tz["image"] * 255).byte().permute(1, 2, 0).numpy().copy()
                cnt, _ = cv2.findContours(bz.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnt, -1, (255, 255, 0), 1); hb = tz["head_box"].tolist()
                if hb[1] > hb[0]: cv2.rectangle(img, (hb[0], hb[2]), (hb[1], hb[3]), (255, 0, 0), 1)
                full = (t0["image"] * 255).byte().permute(1, 2, 0).numpy().copy()
                cv2.imwrite(str(out / f"{s}_{c['name']}_{k}_{mode}.jpg"), cv2.cvtColor(np.concatenate([full, img], 1), cv2.COLOR_RGB2BGR))
ok = [r for r in res if not r.get("skipped")]
summ = {"n": len(ok), "skipped": len(res) - len(ok), "max_proj_err_px": max(r["max_proj_err_px"] for r in ok),
        "mask_prec_unzoomed_mean": float(np.mean([r["mask_prec_unzoomed"] for r in ok])), "mask_prec_zoomed_mean": float(np.mean([r["mask_prec_zoomed"] for r in ok])),
        "by_mode": {m: {"n": sum(r["mode"] == m for r in ok), "prec_zoomed": float(np.mean([r["mask_prec_zoomed"] for r in ok if r["mode"] == m]))} for m in ("cloth", "head")}}
(out / "zoom_check.json").write_text(json.dumps({"summary": summ, "rows": res}, indent=1)); print(json.dumps(summ, indent=1))
