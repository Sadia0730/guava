"""Pose qualitative comparison, released PEAR vs SPRINT (PREREG "Pose table: qualitative comparison with released PEAR").

  errors   both models through the author's-protocol crops of COCO-val and LSP-Extended (tools/eval_author_protocol.py: same
           crops, joints, projection); per-person normalised 2D error and PCK; the dataset PCK must match the stored table rows.
  select   registered rule: "ours better" (main figure), "PEAR better" and random (supplement); 3DPW from the stored per-frame
           PA-MPJPE of both table rows.
  figure   mesh overlays (input, PEAR, ours) + zoom of the joint with the largest error change; checks the shown 3DPW frames
           against the stored per-frame values.
Run with the evaluator's env (guava python, PYTHONPATH=/tmp/pytorch3d-torch22:/raid/ubx858/datasets/eval_assets/pylib).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_author_protocol as eap

EP = Path("/raid/ubx858/outputs/dnt/eval_programme")
SPRINT_CKPT = EP / "sprint_scratch/best_ema.pt"
STORED = {("pear", "COCO-VAL"): Path("/raid/ubx858/outputs/phase2/author/pear/COCO-VAL/summary.json"),
          ("pear", "LSP-EXTENDED"): Path("/raid/ubx858/outputs/phase2/author/pear/LSP-EXTENDED/summary.json"),
          ("pear", "3DPW-TEST"): Path("/raid/ubx858/outputs/phase2/author/pear/3DPW-TEST/per_frame_mpjpe_pa.npy"),
          ("sprint", "COCO-VAL"): EP / "sprint_scratch_camwidth/author/COCO-VAL/summary.json",
          ("sprint", "LSP-EXTENDED"): EP / "sprint_scratch_camwidth/author/LSP-EXTENDED/summary.json",
          ("sprint", "3DPW-TEST"): EP / "sprint_scratch_camwidth/author/3DPW-TEST/per_frame_mpjpe_pa.npy"}
KIND = {"pear": ("pear", "ehm", False), "sprint": ("sprint_raw", "smplx", True)}     # kind, head, camera input (focal = width)
NAMES = {"COCO-VAL": "COCO", "LSP-EXTENDED": "LSP-Extended", "3DPW-TEST": "3DPW"}
KL = list(range(25, 38)) + [43]                       # 3DPW evaluated joints (author's protocol), pelvis = 39
OP_OF_KL = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0]   # their OpenPose-25 ground-truth 2D index (window centre only)


def setup(device):
    ev = eap.load_module("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    return ev, eap.reference_tables(), torch.device(device)


def dataset(ev, name):
    npz, img_root, _ = eap.DATASETS[name]
    return eap.HMR2Crops(ev, eap.EVAL_ROOT / "hmr2_evaluation_data" / npz, img_root)


class Runner:
    """One model as in eval_author_protocol.main: forward, SMPL-X vertices, author joints, projection."""
    def __init__(self, ev, tables, dev, which, batch):
        self.ev, self.dev, self.which = ev, dev, which
        kind, head, self.camera = KIND[which]
        self.kind, self.head = kind, ev.effective_head(kind, head)
        cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
        try:
            ck = SPRINT_CKPT if which == "sprint" else ev.DEFAULT_TEACHER
            self.model, self.report = ev.load_model(kind, ck, "configs/student_l70_v2.yaml", batch, dev)
            from models.modules.ehm import EHM_v2
            self.ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval()
        finally:
            os.chdir(cwd)
        self.mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(dev)
        self.joints = eap.AuthorJoints(ev, tables, dev)

    @torch.inference_mode()
    def __call__(self, batch, ds_name):
        ck = self.ev.camera_kwargs(self.kind, self.camera, batch.get("cam_feat"), self.dev)
        out = self.model(batch["image"].to(self.dev).float() / 255.0, **ck)
        out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
               for k, v in out.items()}
        vx = self.ev.predicted_smplx_vertices(out, self.ehm, self.head)                         # (B, 10475, 3)
        v = torch.stack([torch.sparse.mm(self.mapping, x.float()) for x in vx])
        j = self.joints(v, ds_name)
        aff = batch["affine"].to(self.dev).float()
        px = self.ev.project_to_image(j, out["pd_cam"], aff)
        return {"j2d": px.cpu().numpy(), "j3d": j.cpu().numpy(), "verts": vx.cpu().numpy(),
                "vpx": self.ev.project_to_image(vx, out["pd_cam"], aff).cpu().numpy(),
                "vcam": (torch.einsum("bij,bkj->bki", out["pd_cam"][:, :3, :3], vx) + out["pd_cam"][:, None, :3, 3]).cpu().numpy()}


def norm_err(pred2d, ds):
    """per-person mean normalised distance and PCK@0.05 / 0.1 over ground-truth keypoints with conf > 0.5."""
    vis = ds.kp2d[:, :, 2] > 0.5
    d = np.linalg.norm(pred2d - ds.kp2d[:, :, :2], axis=-1) / ds.box_wh.max(1)[:, None]
    n = vis.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d, vis, (d * vis).sum(1) / n, ((d < 0.05) & vis).sum(1) / n, ((d < 0.1) & vis).sum(1) / n


def cmd_errors(a):
    ev, tables, dev = setup(a.device); pck_mod = eap.load_module("guava_eval_pck_hmr2", ROOT / "tools" / "eval_pck_hmr2.py")
    a.out.mkdir(parents=True, exist_ok=True); report = {}
    for which in ("pear", "sprint"):
        run = Runner(ev, tables, dev, which, a.batch)
        for name in ("COCO-VAL", "LSP-EXTENDED"):
            t0 = time.time(); ds = dataset(ev, name); ds.camera = run.camera; ds.focals = None
            idx = np.arange(0, len(ds), a.stride); pred2d = np.zeros((len(ds), 44, 2))
            for b in DataLoader(Subset(ds, idx), batch_size=a.batch, num_workers=a.workers, pin_memory=True):
                pred2d[b["index"].numpy()] = run(b, name)["j2d"]
            r = pck_mod.pck(pred2d[idx], ds.kp2d[idx, :, :2], ds.kp2d[idx, :, 2], ds.box_wh[idx].max(1))
            stored = json.loads(STORED[(which, name)].read_text())["pck"]
            got = {str(t): v["pck"] for t, v in r.items()}
            ok = a.stride == 1 and all(abs(got[k] - stored[k]) < 1e-3 for k in stored)
            _, vis, e, p05, p10 = norm_err(pred2d, ds)
            np.savez_compressed(a.out / f"{which}_{name}.npz", pred2d=pred2d, err=e, pck05=p05, pck10=p10, nvis=vis.sum(1), scored=idx)
            report[f"{which}/{name}"] = {"pck": got, "stored": stored, "reproduced": bool(ok), "seconds": round(time.time() - t0, 1),
                                         "load_report": run.report}
            print(which, name, got, "stored", stored, "OK" if ok else "MISMATCH", flush=True)
        del run; torch.cuda.empty_cache()
    (a.out / "errors_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")


def cmd_select(a):
    ev, _, _ = setup("cpu"); sel = {"rule": "PREREG 'Pose table: qualitative comparison with released PEAR'", "datasets": {}}
    for name in ("LSP-EXTENDED", "COCO-VAL", "3DPW-TEST"):
        ds = dataset(ev, name)
        if name == "3DPW-TEST":
            ep, es = (np.load(STORED[(w, name)])[:, 1] for w in ("pear", "sprint"))
            elig = np.arange(len(ds)); group = np.array([n.split("/")[1] for n in ds.imgname])
        else:
            P, S = (np.load(a.out / f"{w}_{name}.npz") for w in ("pear", "sprint"))
            ep, es = P["err"], S["err"]
            elig = np.where(P["nvis"] >= 12)[0]; group = np.array(ds.imgname)
        diff = ep - es                                                                             # > 0: ours lower error
        def top(order):
            seen, out = set(), []
            for i in order:
                if group[i] in seen: continue
                seen.add(group[i]); out.append(int(i))
                if len(out) == a.candidates: break
            return out
        o = elig[np.argsort(-diff[elig], kind="stable")]
        sel["datasets"][name] = {
            "eligible": int(len(elig)), "share_ours_lower": float((diff[elig] > 0).mean()),
            "share_pear_lower": float((diff[elig] < 0).mean()), "mean_err_pear": float(ep[elig].mean()), "mean_err_ours": float(es[elig].mean()),
            "ours_better": top(o), "pear_better": top(o[::-1]),
            "random": [int(i) for i in np.random.default_rng(0).choice(elig, 6, replace=False)],
            "metric": "PA-MPJPE mm (stored per frame)" if name == "3DPW-TEST" else "mean normalised 2D error (/ max box side)"}
        d = sel["datasets"][name]
        print(f"{name}: eligible {d['eligible']}, ours lower {d['share_ours_lower']:.3f}, PEAR lower {d['share_pear_lower']:.3f}, "
              f"mean PEAR {d['mean_err_pear']:.4f} ours {d['mean_err_ours']:.4f}")
    (a.out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")


# ---------------------------------------------------------------- figure
def render_mesh(img, vpx, vcam, faces, color=(196, 204, 222)):
    """painter's algorithm in display pixels (vpx), depth and shading from camera coordinates (as lsp_qual.py)."""
    tri = vcam[faces]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = np.abs(n[:, 2]) * 0.72 + 0.28
    layer = img.copy(); mask = np.zeros(img.shape[:2], np.uint8)
    for f in np.argsort(-tri[:, :, 2].mean(1)):
        poly = np.round(vpx[faces[f]] * 4).astype(np.int32)
        cv2.fillConvexPoly(layer, poly, tuple(int(c * shade[f]) for c in color), lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, poly, 1, shift=2)
    m = mask[..., None].astype(np.float32) * 0.88
    return (img * (1 - m) + layer * m).astype(np.uint8)


def to_disp(p_img, aff, S):
    A = np.vstack([aff, [0, 0, 1]]); q = np.c_[p_img, np.ones(len(p_img))] @ A.T
    return q[:, :2] * (S / eap.PATCH)


def cmd_figure(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    ev, tables, dev = setup(a.device); sel = json.loads((a.out / "selection.json").read_text())
    skip = {tuple(s.split(":")) for s in a.skip}
    S = a.size; faces = np.load(ev.PEAR_ROOT / "assets/SMPLX/smplx_faces.npy").astype(np.int64)
    runs = {w: Runner(ev, tables, dev, w, 8) for w in ("pear", "sprint")}
    cache = {}

    def example(name, i):
        if (name, i) in cache: return cache[(name, i)]
        ds = dataset(ev, name); img = cv2.imread(str(ds.img_root / ds.imgname[i]))
        aff = ds.affine(i); A = np.vstack([aff, [0, 0, 1]]); A[:2] *= S / eap.PATCH
        big = cv2.cvtColor(cv2.warpAffine(img, A[:2], (S, S), flags=cv2.INTER_CUBIC, borderValue=0), cv2.COLOR_BGR2RGB)
        ex = {"input": big, "imgname": ds.imgname[i], "index": int(i)}
        gt2 = to_disp(ds.kp2d[i, :, :2], aff, S); vis = ds.kp2d[i, :, 2] > 0.5
        res = {}
        for w, run in runs.items():
            ds.camera = run.camera; ds.focals = None
            b = {k: (v[None] if torch.is_tensor(v) else torch.tensor([v])) for k, v in ds[i].items()}
            r = run(b, name); res[w] = r
            ex[w] = render_mesh(big, to_disp(r["vpx"][0], aff, S), r["vcam"][0], faces)
            ex[w + "_j"] = to_disp(r["j2d"][0], aff, S)
        if name == "3DPW-TEST":
            gt3 = ds.kp3d[i, :, :3] - ds.kp3d[i, 39, :3]; err = {}
            for w in runs:
                pr = res[w]["j3d"][0] - res[w]["j3d"][0, 39]
                al = eap.compute_similarity_transform(torch.from_numpy(pr[KL][None]).double(), torch.from_numpy(gt3[KL][None]).double())[0].numpy()
                pj = np.linalg.norm(al - gt3[KL], axis=1) * 1000
                stored = float(np.load(STORED[(w, name)])[i, 1])
                assert abs(pj.mean() - stored) < 0.5, (w, i, pj.mean(), stored)
                err[w] = pj; ex[w + "_lab"] = f"PA-MPJPE {pj.mean():.0f} mm"
            k = int(np.argmax(np.abs(err["pear"] - err["sprint"]))); jk = KL[k]; gk = OP_OF_KL[k]
            gt_pt = gt2[gk] if vis[gk] else None
            ex["pts"] = {w: ex[w + "_j"][jk] for w in runs}
        else:
            for w in runs:
                d = np.load(a.out / f"{w}_{name}.npz")
                ex[w + "_lab"] = f"PCK@0.05 {d['pck05'][i]:.2f}, error {100 * d['err'][i]:.0f}%"
            dist = {w: np.linalg.norm(ex[w + "_j"] - gt2, axis=1) for w in runs}
            k = int(np.argmax(np.where(vis, np.abs(dist["pear"] - dist["sprint"]), -1))); gt_pt = gt2[k]
            ex["pts"] = {w: ex[w + "_j"][k] for w in runs}
        c = gt_pt if gt_pt is not None else (ex["pts"]["pear"] + ex["pts"]["sprint"]) / 2
        h = 0.30 * S / 2; x0, y0 = np.clip(c - h, 0, S - 2 * h)
        ex.update(win=(int(round(x0)), int(round(y0)), int(round(2 * h))), gt=gt_pt)
        cache[(name, i)] = ex; return ex

    def draw(ax, img, ex, pt=None, zoom=False, lw=0.9):
        x0, y0, w = ex["win"]
        if zoom:
            ax.imshow(img[y0:y0 + w, x0:x0 + w], extent=(x0, x0 + w, y0 + w, y0), interpolation="lanczos")
            ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + w, y0)
            for s in ax.spines.values(): s.set_edgecolor("#f2b705"); s.set_linewidth(1.1)
            mk = 4.2
        else:
            ax.imshow(img); ax.add_patch(Rectangle((x0, y0), w, w, fill=False, ec="#f2b705", lw=lw))
            for s in ax.spines.values(): s.set_visible(False)
            mk = 2.2
        if pt is not None and ex["gt"] is not None:
            ax.plot([ex["gt"][0], pt[0]], [ex["gt"][1], pt[1]], "-", color="#ff3b30", lw=0.8 if zoom else 0.5)
        if pt is not None: ax.plot(*pt, "o", ms=mk, mfc="#ff3b30", mec="white", mew=0.4)
        if ex["gt"] is not None: ax.plot(*ex["gt"], "o", ms=mk, mfc="#34c759", mec="white", mew=0.4)
        if zoom: ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + w, y0)
        else: ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(S - 0.5, -0.5)          # points outside the crop must not rescale the panel
        ax.set_xticks([]); ax.set_yticks([])

    def fig(groups, fname, title_rows):
        """groups: list of rows; each row = (row label, [(dataset, index), ...]) with 2 examples."""
        W = 6.88; cols = 6; gap = 0.04; sep = 0.10; cw = (W - (cols - 1) * gap - sep) / cols
        head, lab, rgap = 0.20, 0.13, 0.10
        rowh = 2 * cw + gap + lab
        H = head + len(groups) * rowh + (len(groups) - 1) * rgap
        f = plt.figure(figsize=(W, H), dpi=450)
        def xpos(c): return c * (cw + gap) + (sep if c >= 3 else 0)
        for c, t in enumerate(["Input", "PEAR", "Ours"] * 2):
            f.text((xpos(c) + cw / 2) / W, 1 - head / 2 / H, t, ha="center", va="center", fontsize=8)
        meta = []
        for r, (rl, items) in enumerate(groups):
            top = head + r * (rowh + rgap)
            if r: f.add_artist(plt.Line2D([0, 1], [1 - (top - rgap / 2) / H] * 2, transform=f.transFigure, color="0.72", lw=0.5))
            for e, (name, i) in enumerate(items):
                ex = example(name, i); meta.append({"row": rl, "dataset": name, "index": i, "image": ex["imgname"],
                                                     "pear": ex["pear_lab"], "ours": ex["sprint_lab"], "window": ex["win"]})
                for p, key in enumerate(["input", "pear", "sprint"]):
                    c = 3 * e + p; x = xpos(c) / W
                    ax = f.add_axes([x, 1 - (top + cw) / H, cw / W, cw / H]); draw(ax, ex[key], ex, ex["pts"].get(key))
                    az = f.add_axes([x, 1 - (top + 2 * cw + gap) / H, cw / W, cw / H]); draw(az, ex[key], ex, ex["pts"].get(key), zoom=True)
                    t = NAMES[name] if p == 0 else ex[key + "_lab"]
                    f.text((xpos(c) + cw / 2) / W, 1 - (top + 2 * cw + gap + 0.02) / H, t, ha="center", va="top", fontsize=6.3)
            if title_rows: f.text(0.002, 1 - (top + cw) / H, rl, rotation=90, ha="left", va="center", fontsize=6.5)
        f.savefig(a.out / f"{fname}.pdf", dpi=450); f.savefig(a.out / f"{fname}.png", dpi=300); plt.close(f)
        (a.out / f"{fname}.json").write_text(json.dumps(meta, indent=1) + "\n"); print("wrote", fname, flush=True)

    D = sel["datasets"]; order = ("LSP-EXTENDED", "COCO-VAL", "3DPW-TEST")
    def pick(name, key, n=2): return [(name, i) for i in D[name][key] if (name, str(i)) not in skip][:n]
    fig([(NAMES[n], pick(n, "ours_better")) for n in order], "fig_pose_ours_better", False)
    fig([(NAMES[n], pick(n, "pear_better")) for n in order], "fig_pose_pear_better", False)
    for n in order:
        rnd = [(n, i) for i in D[n]["random"]]
        fig([(NAMES[n], rnd[k:k + 2]) for k in range(0, 6, 2)], f"fig_pose_random_{n.split('-')[0].lower()}", False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("errors", "select", "figure"))
    p.add_argument("--out", type=Path, default=Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/pose_qual"))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch", type=int, default=64); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="errors: smoke test only (no selection from a strided run)")
    p.add_argument("--candidates", type=int, default=6, help="select: candidates kept per list (figure uses the first 2 not skipped)")
    p.add_argument("--skip", nargs="*", default=[], help="figure: DATASET:INDEX candidates skipped by the registered rule (log in RUNLOG)")
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    {"errors": cmd_errors, "select": cmd_select, "figure": cmd_figure}[a.stage](a)


if __name__ == "__main__":
    main()
