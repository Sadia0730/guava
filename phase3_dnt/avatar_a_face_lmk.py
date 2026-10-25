"""Avatar task A, helper: SCRFD-10G (insightface antelopev2, scrfd_10g_bnkps.onnx, CPU onnxruntime) face box + 5 landmarks
on every tracked frame of the C1 clips (EHM union-box body crop, 1024 px, unmasked; coordinates divided by 2 -> the 512 px
render / metric frame). Used by phase3_dnt/avatar_a_compare.py to align the ArcFace crops: the REAL frame's landmarks are
applied to every render of that frame (one crop per frame, identical for every variant, like C1's face boxes).
Output: <out>/<clip>.npz (keys, score, box (N, 4), kps (N, 5, 2); score 0 / NaN = no face)."""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--split", default="/raid/ubx858/outputs/dnt/c1/split.json"); ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c1/tracked")
ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/avatar_A/face_lmk"); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--antelope", default="/home/ubx858/VACE/models/VACE-Annotators/face/models/antelopev2")
ap.add_argument("--scrfd-py", default="/home/ubx858/miniconda3/envs/vace/lib/python3.10/site-packages/insightface/model_zoo/scrfd.py")
a = ap.parse_args()
import onnxruntime as ort, lmdb, cv2
spec = importlib.util.spec_from_file_location("scrfd_mod", a.scrfd_py); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
det = M.SCRFD(str(Path(a.antelope) / "scrfd_10g_bnkps.onnx"), session=ort.InferenceSession(str(Path(a.antelope) / "scrfd_10g_bnkps.onnx"), so, providers=["CPUExecutionProvider"]))
det.prepare(-1, input_size=(640, 640), det_thresh=0.5)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for c in json.loads(Path(a.split).read_text())["clips"]:
    cd = Path(a.tracked) / c["clip"]; f = out / f"{c['clip']}.npz"
    if not (cd / "optim_tracking_ehm.pkl").exists() or f.exists():
        continue
    t0 = time.time(); vi = json.loads((cd / "videos_info.json").read_text()); keys = list(vi.values())[0]["frames_keys"]
    env = lmdb.open(str(cd / "img_lmdb"), readonly=True, lock=False); tx = env.begin()
    S, B, K = [], [], []
    for k in keys:
        buf = tx.get(f"{k}/body_image".encode())
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)                       # BGR, as insightface expects
        bb, kp = det.detect(img, input_size=(640, 640))
        if len(bb):
            i = int(np.argmax(bb[:, 4] * (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))          # score x area
            S.append(float(bb[i, 4])); B.append(bb[i, :4] / 2.0); K.append(kp[i] / 2.0)
        else:
            S.append(0.0); B.append(np.full(4, np.nan)); K.append(np.full((5, 2), np.nan))
    env.close()
    np.savez(f, keys=np.array(keys), score=np.array(S, np.float32), box=np.array(B, np.float32), kps=np.array(K, np.float32))
    print(c["clip"], len(keys), "frames, faces", int((np.array(S) > 0).sum()), f"{time.time() - t0:.0f} s", flush=True)
print("FACE_LMK_DONE", flush=True)
