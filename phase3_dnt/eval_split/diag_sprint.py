"""Diagnostic B (PREREG "GUAVA-FB improvement phase 1"): which part of the SPRINT pipeline costs quality, on ValBench (validation only).
All variants are DIAGNOSTIC (GT-assisted) except dB_sp_actual. SPRINT (from scratch EMA, deployed setting) runs on the METRIC crop
image (1024, GT-framed; the deployed row runs on the original frame: reference row fb_sprint), so SPRINT's outputs map into the
metric camera exactly up to its intrinsics assumption: SPRINT's camera = the metric camera rotated onto SPRINT's crop-centre ray
(R_c) with focal f_s = 24 x 512 / s_M (its invtanfov 24 on its 1024 crop); a point keeps SPRINT's ray and image size: (x, y, z) ->
R_c^T (x, y, z f / f_s). Records are assembled from components: root orientation, pelvis position, body pose, hands, face (jaw +
expression); the avatar's own shape sets the pelvis offset. Avatars: A_gt (GT source record / crop / matte = fb_gt), A_shp (A_gt
with SPRINT's source shape), A_pose (GT crop / matte / shape, SPRINT's refined source pose + camera mapped), A_sp (SPRINT's own source
fit: its crop, matte, record, shape). GUAVA-FB = B30k best.pt; seed 0 before every call. Renders -> E/renders/<variant>/...
Usage: CUDA_VISIBLE_DEVICES=5 python diag_sprint.py --E <root> --split <json>"""
import argparse, importlib.util, json, os, pickle, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt"); a = ap.parse_args(); a.split = str(Path(a.split).resolve())
E = Path(a.E); CACHE = Path("/dev/shm/ubx858_eval/diag_sprint_cache"); CACHE.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt/sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF); fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", 150)
import cv2, torch, torchvision                                                                       # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                        # noqa: E402
from smplx_lean import SMPLXLean                                                                     # noqa: E402
import convert_bedlam2 as cb                                                                         # noqa: E402
dev = torch.device("cuda:0")
cfg, inf, ren, _ = F.setup_a20(dev); st = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"]); inf.eval(); ren.eval(); del st
LEAN = SMPLXLean(str(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), 200)
Dm = np.diag([-1.0, -1.0, 1.0]); S512 = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])
R2a = lambda R: cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)                              # noqa: E731
a2R = lambda v: cv2.Rodrigues(np.asarray(v, np.float64).reshape(3))[0]                              # noqa: E731


def J0(beta):
    _, J = LEAN.shaped(torch.as_tensor(np.asarray(beta, np.float64).reshape(1, -1)[:, :200])); return J[0, 0].numpy()


def comp_gt(sm):
    RT = np.asarray(sm["camera_RT_params"], np.float64); R0 = Dm @ RT[:, :3] @ a2R(sm["global_pose"])
    P = Dm @ (RT[:, :3] @ J0(sm["shape"]) + RT[:, 3])
    return {"R": R0, "P": P, "body": sm["body_pose"], "lh": sm["left_hand_pose"], "rh": sm["right_hand_pose"], "jaw": sm["jaw_pose"],
            "exp": sm["exp"], "eye": sm["eye_pose"]}


def comp_sp(raw, M, shape_s, itf):
    """SPRINT smplx (its crop camera) -> components in the metric camera (1024 px, focal 512 * itf, principal point 512)."""
    RT = np.asarray(raw["camera_RT_params"], np.float64); Rr = Dm @ RT[:, :3] @ a2R(raw["global_pose"])
    Ps = Dm @ (RT[:, :3] @ J0(shape_s) + RT[:, 3])
    f = 512.0 * itf; f_s = 24 * 512.0 / float(M[0, 0]); cs = np.linalg.inv(np.asarray(M, np.float64)) @ np.array([511.5, 511.5, 1.0])
    Rc = cb.rot_onto_z(np.array([(cs[0] - 511.5) / f, (cs[1] - 511.5) / f, 1.0]))
    P = Rc.T @ np.array([Ps[0], Ps[1], Ps[2] * f / f_s]); R = Rc.T @ Rr
    g = lambda k, shp: np.asarray(raw[k], np.float32).reshape(shp)                                  # noqa: E731
    return {"R": R, "P": P, "body": g("body_pose", (21, 3)), "lh": g("left_hand_pose", (15, 3)), "rh": g("right_hand_pose", (15, 3)),
            "jaw": g("jaw_pose", (1, 3)), "exp": g("exp", 50), "eye": np.zeros((2, 3), np.float32)}


def target(c, shape, itf):
    tp = c["P"] - J0(shape)
    sm = {"global_pose": R2a(c["R"]).astype(np.float32), "body_pose": np.asarray(c["body"], np.float32).reshape(21, 3),
          "left_hand_pose": np.asarray(c["lh"], np.float32).reshape(15, 3), "right_hand_pose": np.asarray(c["rh"], np.float32).reshape(15, 3),
          "jaw_pose": np.asarray(c["jaw"], np.float32).reshape(1, 3), "exp": np.asarray(c["exp"], np.float32).reshape(50),
          "eye_pose": np.asarray(c["eye"], np.float32).reshape(2, 3), "shape": np.asarray(shape, np.float32).reshape(200),
          "camera_RT_params": np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)}
    return sm


def tdict(sm, itf, size=512):
    s = {k: torch.as_tensor(np.asarray(v).copy()) for k, v in sm.items()}
    s.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = s["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    t = {"smplx_coeffs": s, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c), "invtanfov": float(itf)}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], float(itf), size); return to_dev(F.collate1(t), dev)


@torch.no_grad()
def build(src):
    torch.manual_seed(0); vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); return av


@torch.no_grad()
def render(av, t):
    torch.manual_seed(0); return ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)[0]


def u8(x): return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def fit_track(fitter, img, prev_box, **kw):
    """SPRINT fit with a tracking fallback: if the detector finds no person, reuse the previous frame's box (else the whole image)."""
    try:
        return fitter.fit(img, **kw), 0
    except RuntimeError as e:
        if "no person" not in str(e): raise
        h, w = img.shape[:2]; box = prev_box if prev_box is not None else [0, 0, w - 1, h - 1]
        return fitter.fit(img, box=box, **kw), 1


def save(name, ds, clip, k, img):
    d = E / "renders" / name / ds / clip; d.mkdir(parents=True, exist_ok=True); torchvision.io.write_png(img, str(d / f"{k}.png"))


SWAPS = {"dB_gt_gt": [], "dB_gt_root": ["R"], "dB_gt_trans": ["P"], "dB_gt_body": ["body"], "dB_gt_hands": ["lh", "rh"], "dB_gt_face": ["jaw", "exp"],
         "dB_gt_sp": ["R", "P", "body", "lh", "rh", "jaw", "exp", "eye"]}
split = json.loads(Path(a.split).read_text())["clips"]
for c in split:
    ds, name = c["dataset"], c["clip"]
    if (E / "renders" / "dB_sp_actual" / ds / name / "run.json").exists(): continue
    cdir = E / "data" / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
    cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1); fbd = F.FBData(cfg, cl); recs = cl[0]["recs"]
    lm = fbd.lmdb.get(str(cdir)); img = lambda k: lm[f"{k}/body_image"].permute(1, 2, 0).contiguous().numpy()  # noqa: E731
    src = fj["source"]; t0 = time.time()
    cp = CACHE / f"{ds}__{name}.pkl"
    if cp.exists(): C = pickle.load(open(cp, "rb"))
    else:
        rs, _ = fit_track(fitter, img(src), None); C = {"src": {k: rs[k] for k in ("record", "shape", "raw_smplx", "raw_shape", "crop", "alpha", "M")}, "tgt": {}}
        pbox, C["fallback"] = rs["box"], 0
        for k in fj["test"]:
            r_, fb_ = fit_track(fitter, img(k), pbox, iters=0, shape_lock=rs["shape"]); pbox = r_["box"]; C["fallback"] += fb_
            C["tgt"][k] = {"raw": r_["raw_smplx"], "M": r_["M"], "record": r_["record"]}
        pickle.dump(C, open(cp, "wb"))
    t_fit = time.time() - t0
    gshape = recs[src]["smplx"]["shape"]; sshape = C["src"]["shape"]
    # avatars
    A = {"gt": build(to_dev(F.collate1(fbd.source_info(0, src)), dev))}
    def rec_with(shape_new, comp, itf):                                                              # GT source record rewritten (shape / pose)
        sm = target(comp, shape_new, itf); r = dict(recs[src]); r["smplx"] = sm; return r
    for nm, shp, cmp in (("shp", sshape, comp_gt(recs[src]["smplx"])),
                         ("pose", gshape, comp_sp(C["src"]["record"]["smplx_coeffs"], C["src"]["M"], sshape, recs[src]["invtanfov"]))):
        cl2 = [dict(cl[0])]; cl2[0]["recs"] = dict(recs); cl2[0]["recs"][src] = rec_with(shp, cmp, recs[src]["invtanfov"]); fb2 = F.FBData(cfg, cl2)
        A[nm] = build(to_dev(F.collate1(fb2.source_info(0, src)), dev))
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        rsd = dict(C["src"]); SAF.write_tracked(Path(td), name, src, rsd)
        fb3 = F.FBData(cfg, F.load_clips_parallel([(td, None, True)], workers=1)); A["sp"] = build(to_dev(F.collate1(fb3.source_info(0, src)), dev))
    for k in fj["test"]:
        itf = recs[k]["invtanfov"]; g = comp_gt(recs[k]["smplx"]); s = comp_sp(C["tgt"][k]["raw"], C["tgt"][k]["M"], sshape, itf)
        for vn, sw in SWAPS.items():
            cc = {kk: (s[kk] if kk in sw else g[kk]) for kk in g}; save(vn, ds, name, k, u8(render(A["gt"], tdict(target(cc, gshape, itf), itf))))
        save("dB_shp_gt", ds, name, k, u8(render(A["shp"], tdict(target(g, sshape, itf), itf))))
        save("dB_srcpose_gt", ds, name, k, u8(render(A["pose"], tdict(target(g, gshape, itf), itf))))
        save("dB_sp_gt", ds, name, k, u8(render(A["sp"], tdict(target(g, sshape, itf), itf))))
        save("dB_sp_sp", ds, name, k, u8(render(A["sp"], tdict(target(s, sshape, itf), itf))))
        # deployed path on the metric crop image: SPRINT's own camera (invtanfov 24 on its crop), rendered at 512, warped
        rr = dict(C["tgt"][k]["record"]); rr["smplx_coeffs"] = C["tgt"][k]["raw"]; comp = F.compact_record(rr, sshape)
        sm = dict(comp["smplx"]); r = render(A["sp"], tdict(sm, 24.0))
        W = S512 @ np.linalg.inv(np.asarray(C["tgt"][k]["M"], np.float64)) @ np.linalg.inv(S512)
        wp = cv2.warpPerspective(u8(r).permute(1, 2, 0).numpy(), W, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        save("dB_sp_actual", ds, name, k, torch.from_numpy(wp).permute(2, 0, 1).contiguous())
    for vn in list(SWAPS) + ["dB_shp_gt", "dB_srcpose_gt", "dB_sp_gt", "dB_sp_sp", "dB_sp_actual"]:
        (E / "renders" / vn / ds / name / "run.json").write_text(json.dumps({"method": vn, "diagnostic": vn != "dB_sp_actual", "sprint_input": "metric crop image",
                                                                             "sprint_cache_s": t_fit}))
    print("done", c["id"], f"{time.time() - t0:.0f} s", flush=True)
