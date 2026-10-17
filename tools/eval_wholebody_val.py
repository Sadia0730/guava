"""COCO-WholeBody val2017: face (68) and hand (2 x 21) 2D keypoint accuracy of the EVALUATED (EHM) mesh (DNT F-a).

Persons: iscrowd 0, >= 6 body keypoints, box max side >= 64 px (the COCO training-crop filter), at least one valid part
(face_valid / lefthand_valid / righthand_valid). Crop: the annotation bbox through PEAR's process_bbox (x1.25) -> 256 px
(the author's-protocol crop). Prediction: EHM vertices / joints of the model output -> DWPose-134 face / hand keypoints
(smplx_joints_to_dwpose), projected with the predicted crop camera, mapped back to image pixels.
Metric: PCK@a, a in {0.1, 0.2}, threshold a x the max side of the part's GT keypoint box (visible keypoints), over the
visible keypoints of valid parts; per part (face, hands), plus the mean pixel error normalised by the same box side (NME)."""
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); import eval_3dpw_standard as ev
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True, choices=("pear", "student_vitpose")); p.add_argument("--checkpoint", type=Path, default=None)
p.add_argument("--student-config", default="configs/student_l70_v2.yaml"); p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--wholebody", type=Path, default=Path("/raid/ubx858/datasets/train2d/coco_wholebody/coco_wholebody_val_v1.0.json"))
p.add_argument("--images", type=Path, default=Path("/raid/ubx858/datasets/eval_suite/coco/val2017")); p.add_argument("--device", default="cuda:0")
a = p.parse_args(); t0 = time.time(); dev = torch.device(a.device)
W = json.loads(a.wholebody.read_text()); imgs = {im["id"]: im for im in W["images"]}
items = []
for an in W["annotations"]:
    if an.get("iscrowd", 0) or an.get("num_keypoints", 0) < 6 or max(an["bbox"][2:]) < 64: continue
    parts = {k: np.asarray(an[f], np.float64).reshape(n, 3) for k, f, n, v in (("face", "face_kpts", 68, "face_valid"), ("lhand", "lefthand_kpts", 21, "lefthand_valid"), ("rhand", "righthand_kpts", 21, "righthand_valid")) if an.get(v)}
    parts = {k: v for k, v in parts.items() if (v[:, 2] > 0).sum() >= (20 if k == "face" else 6)}
    if parts: items.append((imgs[an["image_id"]]["file_name"], np.asarray(an["bbox"], np.float32), parts))
cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT); sys.path.insert(0, str(ev.PEAR_ROOT))
for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]: sys.modules.pop(k)
from models.modules.ehm import EHM_v2
from models.smplx.smplx_head import smplx_joints_to_dwpose
import train_pear_student_distill as tpd
model, rep = ev.load_model(a.model, (a.checkpoint or ev.DEFAULT_TEACHER).resolve(), a.student_config, 32, dev)
ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval(); os.chdir(cwd)
hits = {k: {0.1: [], 0.2: []} for k in ("face", "hands")}; nme = {"face": [], "hands": []}
for s in range(0, len(items), 32):
    batch = items[s:s + 32]; crops, affs = [], []
    for fn, box, _ in batch:
        im = cv2.imread(str(a.images / fn)); h, w = im.shape[:2]
        bb = ev.process_bbox(box, w, h, (ev.IMAGE_SIZE, ev.IMAGE_SIZE), ratio=1.25); A = ev.box_to_affine(bb)
        crops.append(cv2.cvtColor(cv2.warpAffine(im, A, (ev.IMAGE_SIZE, ev.IMAGE_SIZE), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)); affs.append(A)
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(dev).float() / 255.0
    with torch.inference_mode():
        o = model(x); o = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float()) for k, v in o.items()}
        dw = smplx_joints_to_dwpose(ehm(o["body_param"], o["flame_param"], pose_type="rotmat")["joints"])[0][..., :3]
        uv = (tpd.project_joints(o["pd_cam"], dw) * ev.IMAGE_SIZE).cpu().numpy()            # crop px
    for b, (fn, box, parts) in enumerate(batch):
        Ainv = cv2.invertAffineTransform(np.asarray(affs[b], np.float64)); P = uv[b] @ Ainv[:, :2].T + Ainv[:, 2]
        for k, gt in parts.items():
            pr = P[{"face": slice(24, 92), "lhand": slice(92, 113), "rhand": slice(113, 134)}[k]]; v = gt[:, 2] > 0
            side = max(float(np.ptp(gt[v, 0])), float(np.ptp(gt[v, 1])), 1.0); d = np.linalg.norm(pr[v] - gt[v, :2], axis=1) / side
            key = "face" if k == "face" else "hands"
            for al in (0.1, 0.2): hits[key][al].extend((d <= al).tolist())
            nme[key].append(float(d.mean()))
summ = {"model": a.model, "checkpoint": str(a.checkpoint), "load_report": rep, "persons": len(items),
        "parts": {k: len(v) for k, v in nme.items()}, **{f"pck{int(al*100):02d}_{k}": float(np.mean(hits[k][al])) for k in hits for al in (0.1, 0.2)},
        **{f"nme_{k}": float(np.mean(v)) for k, v in nme.items()}, "seconds": time.time() - t0, "evaluator": "tools/eval_wholebody_val.py"}
a.output_dir.mkdir(parents=True, exist_ok=True); (a.output_dir / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
