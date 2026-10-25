"""Avatar task A, helper: GUAVA (a20) face parameters for every EHM-tracked frame of the C1 test clips -- the same conversion
as phase3_dnt/guava_a20_conversion.py (third_party/PEAR/tools/build_face_labels_smplx2020.py: build_models / lstsq_map /
convert "refit", 60 LBFGS iterations = per-row LBS refit of the tracker's FLAME expression + jaw + eyelids onto the SMPL-X
2020 exp50 + jaw). The C1 clips have no SPRINT labels, so every frame is refitted here (the training conversion's check:
refit vs labels 0.16 mm head error). Eyes: the training conversion's decision (face_a20.npz eye_copy) is reused.
Output: <out>/face_a20_c1.npz (clip, frame_key, exp50, jaw, eye_copy), <out>/report.json."""
import argparse, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c1/tracked"); ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/avatar_A/conversion_c1")
ap.add_argument("--train-conv", default="/raid/ubx858/outputs/dnt/guava_a20/conversion/face_a20.npz")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0"); t0 = time.time()
spec = importlib.util.spec_from_file_location("bfl2020", str(ROOT / "third_party/PEAR/tools/build_face_labels_smplx2020.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
M = T.build_models(dev)
W_fl = T.lstsq_map(M["x2020"].B.cpu().numpy().astype(np.float64), M["flame"].B.cpu().numpy().astype(np.float64), M["fitmask"])[0]
C, K, FE, FJ, FL = [], [], [], [], []
for c in sorted(p.parent for p in Path(a.tracked).glob("*/optim_tracking_ehm.pkl")):
    d = pickle.load(open(c / "optim_tracking_ehm.pkl", "rb"))
    for k in sorted(d):
        fc = d[k]["flame_coeffs"]
        C.append(c.name); K.append(k); FE.append(np.asarray(fc["expression_params"], np.float32).reshape(50))
        FJ.append(np.asarray(fc["jaw_params"], np.float32).reshape(3)); FL.append(np.asarray(fc["eyelid_params"], np.float32).reshape(2))
FE, FJ, FL = map(np.asarray, (FE, FJ, FL))
E, J = T.convert(M, "flame", W_fl, FE, FJ, FL, "refit", dev, iters=60)
ev = T.evaluate(M, "ubody_ehm_gt", {"e": FE, "jaw": FJ, "lid": FL, "model": "flame"}, W_fl, np.arange(len(FE)), E, J, dev)[0]
eye_copy = bool(np.load(a.train_conv)["eye_copy"])
np.savez(out / "face_a20_c1.npz", clip=np.array(C), frame_key=np.array(K), exp50=E.astype(np.float32), jaw=J.astype(np.float32), eye_copy=np.array(eye_copy))
rep = {"frames": len(C), "clips": len(set(C)), "head_err_mm": ev["vertex_err_mm"], "src_disp_mm": ev.get("src_disp_mm"), "eye_copy": eye_copy, "elapsed_s": time.time() - t0}
(out / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
