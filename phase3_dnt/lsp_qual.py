"""LSP-Extended qualitative grid (as PEAR Fig. 9): each checkpoint's SMPL-X mesh overlaid on the author's-protocol input
crop (tools/eval_author_protocol.py HMR2Crops, 256 px, rendered at 512). Image choice is model-independent, from the GT 2D
keypoints only: 3 most inverted bodies (head below hips), 3 widest leg spreads (ankle distance / torso), 6 random (seed 0);
every image needs >= 12 visible keypoints. Columns: input, then one per --model NAME=KIND:CKPT (KIND pear: CKPT ignored).
Output: <out>/lsp_qual_<k>.jpg (6 images per sheet) + selection.json."""
import argparse, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_author_protocol as eap
ap = argparse.ArgumentParser(); ap.add_argument("--model", action="append", required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--head", choices=("ehm", "smplx"), default="ehm", help="mesh head of the student models (PEAR: ehm)"); a = ap.parse_args()
ev = eap.load_module("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
npz, img_root, _ = eap.DATASETS["LSP-EXTENDED"]
ds = eap.HMR2Crops(ev, eap.EVAL_ROOT / "hmr2_evaluation_data" / npz, img_root)
# --- selection (GT only). OpenPose-25 part of the 44: 1 neck, 8 mid-hip, 11 / 14 ankles; SPIN extra (25+): 25 / 30 ankles, 27 / 28 hips, 37 neck, 38 head top
kp = ds.kp2d; c = kp[..., 2] > 0
def pt(i, cand):
    for j in cand:
        if c[i, j]: return kp[i, j, :2]
    return None
inv, spread, ok = [], [], []
for i in range(len(ds)):
    if c[i].sum() < 12: continue
    head, hip = pt(i, [38, 37, 1]), None
    hs = [pt(i, [9, 27]), pt(i, [12, 28])]
    if hs[0] is not None and hs[1] is not None: hip = (hs[0] + hs[1]) / 2
    ra, la = pt(i, [11, 25]), pt(i, [14, 30])
    if head is None or hip is None: continue
    ok.append(i); torso = np.linalg.norm(head - hip) + 1e-6
    inv.append(((head[1] - hip[1]) / torso, i))
    if ra is not None and la is not None: spread.append((np.linalg.norm(ra - la) / torso, i))
sel = {"inverted": [i for _, i in sorted(inv)[::-1][:3]]}
sel["leg_spread"] = [i for _, i in sorted(spread)[::-1] if i not in sel["inverted"]][:3]
rest = [i for i in ok if i not in sel["inverted"] + sel["leg_spread"]]
sel["random"] = [int(i) for i in np.random.default_rng(0).choice(rest, 6, replace=False)]
order = [int(i) for k in ("inverted", "leg_spread", "random") for i in sel[k]]
faces = np.load(ev.PEAR_ROOT / "assets/SMPLX/smplx_faces.npy").astype(np.int64)
dev = torch.device("cuda:0"); S = a.size
crops = [ds[i] for i in order]
x = torch.stack([b["image"] for b in crops]).to(dev).float() / 255.0
def overlay(img, v_cam, rt):
    cam = v_cam @ rt[:3, :3].T + rt[:3, 3]
    ndc = 24.0 * cam[:, :2] / np.clip(cam[:, 2:3], 1e-4, None)
    p = (1.0 - ndc) * 0.5 * S                                      # crop pixels (PEAR projection, as ev.project_to_image)
    tri = cam[faces]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = np.abs(n @ np.array([0.0, 0.0, 1.0])) * 0.75 + 0.25
    layer = img.copy(); mask = np.zeros((S, S), np.uint8)
    for f in np.argsort(-tri[:, :, 2].mean(1)):                   # painter's algorithm, far first
        poly = np.round(p[faces[f]] * 4).astype(np.int32)
        col = tuple(int(v * shade[f]) for v in (200, 205, 215))
        cv2.fillConvexPoly(layer, poly, col, lineType=cv2.LINE_AA, shift=2); cv2.fillConvexPoly(mask, poly, 1, shift=2)
    m = mask[..., None].astype(np.float32) * 0.9
    return (img * (1 - m) + layer * m).astype(np.uint8)
base = [cv2.resize(b["image"].permute(1, 2, 0).numpy(), (S, S), interpolation=cv2.INTER_CUBIC) for b in crops]
cols = [("Input", base)]
cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT); sys.path.insert(0, str(ev.PEAR_ROOT))
from models.modules.ehm import EHM_v2
ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval()
for spec in a.model:
    name, rest_ = spec.split("=", 1); kind, ck = rest_.split(":", 1)
    model, _ = ev.load_model(kind, Path(ck) if kind != "pear" else ev.DEFAULT_TEACHER, "configs/student_l70_v2.yaml", len(order), dev)
    with torch.inference_mode():
        out = model(x)
        out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float()) for k, v in out.items()}
        v = ev.predicted_smplx_vertices(out, ehm, ev.effective_head(kind, a.head)).cpu().numpy(); rt = out["pd_cam"].cpu().numpy()
    cols.append((name, [overlay(base[k], v[k], rt[k]) for k in range(len(order))]))
    del model; torch.cuda.empty_cache(); print("rendered", name, flush=True)
os.chdir(cwd); a.out.mkdir(parents=True, exist_ok=True)
lab = np.full((40, S, 3), 255, np.uint8)
for s in range(0, len(order), 6):
    rows = [np.concatenate([np.concatenate([cv2.putText(lab.copy(), n, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA) for n, _ in cols], 1)])]
    for k in range(s, min(s + 6, len(order))):
        rows.append(np.concatenate([np.pad(im[k], ((2, 2), (2, 2), (0, 0)), constant_values=255) [2:-2, 2:-2] for _, im in cols], 1))
    cv2.imwrite(str(a.out / f"lsp_qual_{s // 6}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump({"selection": sel, "images": [ds.imgname[i] for i in order], "models": a.model,
           "head": {s.split("=", 1)[0]: ev.effective_head(s.split("=", 1)[1].split(":", 1)[0], a.head) for s in a.model}}, open(a.out / "selection.json", "w"), indent=1)
print("done", a.out)
