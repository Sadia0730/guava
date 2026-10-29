"""4D-DRESS rows to skip in training / validation: person truncated by the 1280 x 940 image (record["dress4d"]["trunc_frac"] =
fraction of mesh vertices outside the image) > 0.05; their crops are black-padded and StyleMatte often misses the half-visible
person. Writes O/data/dress4d/drop_frames.json {clip: [frame_key, ...]} (read by guava_fb_train.py)."""
import json, pickle
from pathlib import Path
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/data/dress4d")
drop, n, nd = {}, 0, {"train": 0, "val": 0}
for e in json.loads((O / "index.json").read_text())["clips"]:
    recs = pickle.load(open(O / e["name"] / "optim_tracking_ehm.pkl", "rb"))
    for k, r in recs.items():
        n += 1
        if r["dress4d"]["trunc_frac"] > 0.05: drop.setdefault(e["name"], []).append(k); nd[e["split"]] += 1
(O / "drop_frames.json").write_text(json.dumps(drop, indent=1))
print("rows", n, "dropped", nd, "clips", len(drop))
