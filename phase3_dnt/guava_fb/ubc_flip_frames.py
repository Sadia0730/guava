"""UBC front/back flip frames (not a registered criterion; information + training exclusion): fitted root facing (> 0: body faces
the camera; fit_ubc.py) vs DWPose shoulder order in the 1024 crop (EHM 134 layout, OpenPose body: 2 = right shoulder, 5 = left
shoulder; facing the camera <=> right shoulder left of the left shoulder). Checked only when both shoulder scores >= 0.5, the
shoulder x-gap >= 40 px and |facing| >= 0.5 (the rule in reports/ubc/report.json). Writes O/data/ubc/drop_frames.json
{clip: [frame_key, ...]} for the kept (train) videos."""
import json, pickle
from pathlib import Path
import numpy as np
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
idx = json.loads((O / "data/ubc/index.json").read_text()); drop, n_chk = {}, 0
for e in idx["clips"]:
    if e["split"] != "train": continue
    fr = np.load(O / "runs/ubc/clips" / e["name"] / "frames.npz"); fac = dict(zip(fr["keys"], fr["facing"]))
    recs = pickle.load(open(O / "data/ubc" / e["name"] / "optim_tracking_ehm.pkl", "rb"))
    for k, r in recs.items():
        d = r["dwpose_raw"]; kp, sc = np.asarray(d["keypoints_crop1024"]), np.asarray(d["scores"])
        if k not in fac or sc[2] < 0.5 or sc[5] < 0.5 or abs(kp[2, 0] - kp[5, 0]) < 40 or abs(fac[k]) < 0.5: continue
        n_chk += 1
        if (kp[2, 0] < kp[5, 0]) != (fac[k] > 0): drop.setdefault(e["name"], []).append(k)
(O / "data/ubc/drop_frames.json").write_text(json.dumps(drop, indent=1))
print("checked", n_chk, "flipped", sum(map(len, drop.values())), "videos", len(drop))
