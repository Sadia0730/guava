"""Orientation-hard evaluation on LSP-Extended and COCO-val (author's-protocol crops: tools/eval_author_protocol.py
HMR2Crops; 2D keypoints exactly as that evaluator: model -> mesh (--head) -> SMPL topology -> 44 HMR2 joints -> image px).

GROUND-TRUTH RULES (from the GT 2D keypoints ONLY; fixed before any model was run: `--gt-only` writes them and the
resulting subsets, rules_sha256 identifies the rule text below):
  keypoints   GT conf > 0 (= annotated; the npz stores 0/1). LSP-Extended: the 14 LSP joints in extra_keypoints_2d
              (44-index 25+: r_hip 27, l_hip 28, r_shoulder 33, l_shoulder 34); COCO-val: OpenPose-25 body_keypoints_2d
              (r_shoulder 2, l_shoulder 5, r_hip 9, l_hip 12, nose 0, r_eye 15, l_eye 16, r_ear 17, l_ear 18).
  torso       defined iff both shoulders and both hips are annotated. u = mid-shoulder - mid-hip (image px, y down).
  tilt        GT torso tilt = angle(u, image up (0, -1)) in [0, 180] deg.
  handedness  s = l_shoulder - r_shoulder; h = (u_x s_y - u_y s_x) / |u|  (signed shoulder separation perpendicular
              to the torso, px). Person-centric L/R labels => h > 0 when the chest faces the camera, at ANY in-plane
              rotation (as orient_lib / taxonomy.py `hand`). Defined only if |h| > 0.3 |u| (not side-on).
  front/back  LSP:  front = handedness defined and h > 0; back = defined and h < 0 (LSP has no face keypoints; only
                    head top / neck). Undefined otherwise (side-on, or torso undefined).
              COCO: front = nose and both eyes annotated; back = nose and both eyes NOT annotated, >= 1 ear annotated
                    and both shoulders annotated (COCO annotators only label locatable keypoints). Undefined otherwise.
  hard        torso defined and (tilt > 90 deg or back).
  bins        upright = tilt <= 90, inverted = tilt > 90 (finer: <45, 45-90, 90-135, >135); torso defined only.

MODEL QUANTITIES:
  facing      predicted body forward axis = column 2 of body_param["global_pose"] (camera axes, OpenCV: z forward;
              eval_3dpw_standard.predicted_smplx_vertices); predicted back-facing iff its z component > 0.
  flip        predicted back-facing != GT back, over images whose GT front/back is defined.
  torso dir   predicted torso up = mid-shoulder - mid-hip of the model's projected 2D joints (same joints as the GT
              rule); error = angle to the GT u, deg [0, 180]; secondary: the projected root up axis
              (global_pose column 1 = body +y, x / y components; image y down, so upright = (0, -1)) vs GT u.
  PCK         HMR2 / author protocol PCK@0.05 / 0.1 (tools/eval_pck_hmr2.pck, conf > 0.5, normaliser = box max side)
              on the full set (must equal the author-protocol summary) and on the hard subset.
Output: <output-dir>/summary.json, per_image.npz. `--table DIR ...` prints a markdown table of several summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_author_protocol as eap  # noqa: E402

RULES = __doc__.split("GROUND-TRUTH RULES")[1].split("MODEL QUANTITIES")[0]
RULES_SHA256 = hashlib.sha256(RULES.encode()).hexdigest()
DATASETS = ("LSP-EXTENDED", "COCO-VAL")
TORSO = {"LSP-EXTENDED": dict(rsho=33, lsho=34, rhip=27, lhip=28), "COCO-VAL": dict(rsho=2, lsho=5, rhip=9, lhip=12)}
SIDE_MARGIN = 0.3
FINE_BINS = (("upright<45", 0, 45), ("45-90", 45, 90), ("90-135", 90, 135), ("inverted>135", 135, 180.1))


def angle(a, b):
    """Unsigned angle (deg) between 2D vectors (N, 2)."""
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def torso_vectors(xy, J):
    u = 0.5 * (xy[:, J["rsho"]] + xy[:, J["lsho"]]) - 0.5 * (xy[:, J["rhip"]] + xy[:, J["lhip"]])
    s = xy[:, J["lsho"]] - xy[:, J["rsho"]]
    return u, s


def gt_labels(dataset, kp):
    """kp (N, 44, 3) GT; returns dict of per-image GT labels (rules in the module docstring)."""
    J = TORSO[dataset]; c = kp[..., 2] > 0; xy = kp[..., :2]
    torso = c[:, [J["rsho"], J["lsho"], J["rhip"], J["lhip"]]].all(1)
    u, s = torso_vectors(xy, J)
    un = np.linalg.norm(u, axis=1)
    tilt = np.where(torso, angle(u, np.tile([0.0, -1.0], (len(u), 1))), np.nan)
    h = (u[:, 0] * s[:, 1] - u[:, 1] * s[:, 0]) / (un + 1e-9)
    hand_def = torso & (np.abs(h) > SIDE_MARGIN * un)
    if dataset == "LSP-EXTENDED":
        front, back = hand_def & (h > 0), hand_def & (h < 0)
    else:
        face = c[:, 0] & c[:, 15] & c[:, 16]; noface = ~c[:, 0] & ~c[:, 15] & ~c[:, 16]
        front = face; back = noface & (c[:, 17] | c[:, 18]) & c[:, 2] & c[:, 5]
    hard = torso & ((np.nan_to_num(tilt) > 90) | back)
    return {"torso": torso, "tilt": tilt, "u": u, "hand_h": h, "hand_defined": hand_def, "front": front, "back": back,
            "fb_defined": front | back, "hard": hard}


def gt_counts(g):
    t = g["torso"]; tilt = np.nan_to_num(g["tilt"], nan=-1)
    out = {"N": int(len(t)), "torso_defined": int(t.sum()), "front": int(g["front"].sum()), "back": int(g["back"].sum()),
           "hard": int(g["hard"].sum()), "hard_by_tilt>90": int((t & (tilt > 90)).sum()), "hard_by_back": int(g["back"].sum()),
           "hard_both": int((t & (tilt > 90) & g["back"]).sum()),
           "upright": int((t & (tilt <= 90)).sum()), "inverted": int((t & (tilt > 90)).sum())}
    out.update({b: int((t & (tilt >= lo) & (tilt < hi)).sum()) for b, lo, hi in FINE_BINS})
    out["hard_index_sha256"] = hashlib.sha256(np.flatnonzero(g["hard"]).astype(np.int64).tobytes()).hexdigest()
    return out


def stats(x):
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
            ">45": float((x > 45).mean()) if len(x) else None, ">90": float((x > 90).mean()) if len(x) else None}


def predict(ev, model, ehm, ds, dataset, head, joints, mapping, device, batch_size, num_workers):
    pred2d = np.zeros((len(ds), 44, 2)); G = np.zeros((len(ds), 3, 3))
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True):
        idx = batch["index"].numpy()
        with torch.inference_mode():
            out = model(batch["image"].to(device).float() / 255.0)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            v = torch.stack([torch.sparse.mm(mapping, x.float()) for x in ev.predicted_smplx_vertices(out, ehm, head)])
            j = joints(v, dataset)
            px = ev.project_to_image(j, out["pd_cam"], batch["affine"].to(device).float())
        pred2d[idx] = px.cpu().numpy(); G[idx] = out["body_param"]["global_pose"][:, 0].cpu().numpy()
    return pred2d, G


def evaluate(dataset, ds, g, pred2d, G, pck_mod):
    J = TORSO[dataset]; t = g["torso"]; tilt = np.nan_to_num(g["tilt"], nan=-1)
    box = ds.box_wh.max(1)
    def pck(m):
        r = pck_mod.pck(pred2d[m], ds.kp2d[m, :, :2], ds.kp2d[m, :, 2], box[m])
        return {str(k): v["pck"] for k, v in r.items()}
    away = G[:, 2, 2] > 0
    fb = g["fb_defined"]; flip = away != g["back"]
    pu, _ = torso_vectors(pred2d, J)
    err = np.where(t, angle(pu, g["u"]), np.nan)
    root_up = G[:, :2, 1]                                        # body +y in camera axes; upright -> (0, -1) (y down)
    err_root = np.where(t, angle(root_up, g["u"]), np.nan)
    res = {"pck_all": pck(np.ones(len(ds), bool)), "pck_hard": pck(g["hard"]),
           "pck_not_hard": pck(~g["hard"])}
    def flips(m):
        m = m & fb
        return {"n": int(m.sum()), "flip_rate": float(flip[m].mean()) if m.sum() else None,
                "front_pred_away": float(away[m & g["front"]].mean()) if (m & g["front"]).sum() else None,
                "back_pred_toward": float((~away[m & g["back"]]).mean()) if (m & g["back"]).sum() else None}
    res["flip"] = {"all": flips(np.ones(len(ds), bool)), "hard": flips(g["hard"]),
                   "upright": flips(t & (tilt <= 90)), "inverted": flips(t & (tilt > 90))}
    bins = {"all": t, "hard": g["hard"], "upright": t & (tilt <= 90), "inverted": t & (tilt > 90)}
    bins.update({b: t & (tilt >= lo) & (tilt < hi) for b, lo, hi in FINE_BINS})
    res["torso_dir_err_deg"] = {b: stats(err[m]) for b, m in bins.items()}
    res["root_up_err_deg"] = {b: stats(err_root[m]) for b, m in bins.items()}
    res["pred_back_facing_rate"] = float(away.mean())
    return res, {"pred2d": pred2d.astype(np.float32), "global_pose": G.astype(np.float32), "pred_away": away, "torso_err": err,
                 "root_up_err": err_root}


def table(dirs):
    S = [json.loads((Path(d) / "summary.json").read_text()) for d in dirs]
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for ds in DATASETS:
        print(f"\n### {ds}  (hard n = {S[0]['gt'][ds]['counts']['hard']}, front/back defined n = "
              f"{S[0]['gt'][ds]['counts']['front'] + S[0]['gt'][ds]['counts']['back']})\n")
        print("| model | head | PCK@.05 all | PCK@.1 all | PCK@.05 hard | PCK@.1 hard | flip all | flip hard | torso err upright mean / med | torso err inverted mean / med | root-up err inverted med |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for d, s in zip(dirs, S):
            r = s["results"][ds]; te = r["torso_dir_err_deg"]
            print(f"| {s['name']} | {s['head']} | {f(r['pck_all']['0.05'])} | {f(r['pck_all']['0.1'])} | {f(r['pck_hard']['0.05'])} | "
                  f"{f(r['pck_hard']['0.1'])} | {f(r['flip']['all']['flip_rate'])} | {f(r['flip']['hard']['flip_rate'])} | "
                  f"{f(te['upright']['mean'], 1)} / {f(te['upright']['median'], 1)} | {f(te['inverted']['mean'], 1)} / {f(te['inverted']['median'], 1)} | "
                  f"{f(r['root_up_err_deg']['inverted']['median'], 1)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pear", "student", "student_vitpose", "student_vitpose_mq", "student_vitpose_mq_handcrop", "sprint", "sprint_raw"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--head", choices=("ehm", "smplx"), default="ehm")
    p.add_argument("--name", default=None, help="row name in the table (default: output-dir parent name)")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gt-only", action="store_true", help="write the GT rules / subsets only (no model)")
    p.add_argument("--table", nargs="+", default=None, help="print a markdown table of these output dirs and exit")
    a = p.parse_args()
    if a.table:
        return table(a.table)
    t0 = time.time()
    ev = eap.load_module("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    pck_mod = eap.load_module("guava_eval_pck_hmr2", ROOT / "tools" / "eval_pck_hmr2.py")
    data, gts = {}, {}
    for dsn in DATASETS:
        npz, img_root, _ = eap.DATASETS[dsn]
        labels = eap.EVAL_ROOT / "hmr2_evaluation_data" / npz
        data[dsn] = eap.HMR2Crops(ev, labels, img_root)
        g = gt_labels(dsn, data[dsn].kp2d)
        gts[dsn] = {"g": g, "counts": gt_counts(g), "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest()}
    gt_report = {dsn: {"counts": v["counts"], "labels_sha256": v["labels_sha256"]} for dsn, v in gts.items()}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.gt_only:
        (a.output_dir / "gt_rules.json").write_text(json.dumps({"rules": RULES, "rules_sha256": RULES_SHA256, "gt": gt_report,
                                                               "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
        print(RULES); print(json.dumps(gt_report, indent=1)); return
    head = ev.effective_head(a.model, a.head)
    device = torch.device(a.device)
    checkpoint = (a.checkpoint or ev.DEFAULT_TEACHER).resolve()
    cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
    try:
        model, load_report = ev.load_model(a.model, checkpoint, a.student_config, a.batch_size, device)
        from models.modules.ehm import EHM_v2
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    finally:
        os.chdir(cwd)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    joints = eap.AuthorJoints(ev, eap.reference_tables(), device)
    results, per = {}, {}
    for dsn in DATASETS:
        pred2d, G = predict(ev, model, ehm, data[dsn], dsn, head, joints, mapping, device, a.batch_size, a.num_workers)
        results[dsn], arr = evaluate(dsn, data[dsn], gts[dsn]["g"], pred2d, G, pck_mod)
        per.update({f"{dsn}/{k}": v for k, v in arr.items()})
        per.update({f"{dsn}/gt_{k}": v for k, v in gts[dsn]["g"].items()})
        print(dsn, json.dumps({"pck_all": results[dsn]["pck_all"], "pck_hard": results[dsn]["pck_hard"],
                               "flip": results[dsn]["flip"]["all"]["flip_rate"]}), flush=True)
    np.savez_compressed(a.output_dir / "per_image.npz", **per)
    summ = {"name": a.name or a.output_dir.parent.name, "model": a.model, "head": head, "checkpoint": str(checkpoint),
            "checkpoint_sha256": ev.sha256(checkpoint), "load_report": load_report,
            "rules_sha256": RULES_SHA256, "rules": RULES, "gt": gt_report, "results": results,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git": {"guava": ev.git_state(ev.ROOT), "pear": ev.git_state(ev.PEAR_ROOT)},
            "command": " ".join(sys.argv), "seconds": round(time.time() - t0, 1)}
    (a.output_dir / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    table([a.output_dir])


if __name__ == "__main__":
    main()
