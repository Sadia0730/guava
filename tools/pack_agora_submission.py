#!/usr/bin/env python
"""Zip a folder of AGORA prediction pickles (tools/eval_agora.py) for the AGORA evaluation server.
Layout required by the official check_pred_format.py: a top-level folder `predictions/` holding `<image>_personId_<k>.pkl`.
Every file is checked against the official rules for the joints-and-vertices route (SMPL-X): `joints` ndarray (>= 24, >= 2),
`verts` ndarray (10475, 3), `allSmplJoints3d` ndarray (>= 127, 3), all finite; then stored with pickle protocol 3 (plain bytes;
protocol 2 escapes the array bytes and is about 1.5 x larger; the official loader reads either) after checking that the arrays
are bit-identical.
Usage: python tools/pack_agora_submission.py --pred-dir DIR --out predictions.zip"""
import argparse
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--pred-dir", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); files = sorted(a.pred_dir.glob("*.pkl")); assert files, "no predictions"
images = set(); n = 0
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
    for f in files:
        d = pickle.load(open(f, "rb"), encoding="latin1")
        j, v, j3 = d["joints"], d["verts"].squeeze(), d["allSmplJoints3d"].squeeze()
        assert isinstance(j, np.ndarray) and j.ndim == 2 and j.shape[0] >= 24 and j.shape[1] >= 2, (f.name, "joints")
        assert isinstance(v, np.ndarray) and v.shape == (10475, 3), (f.name, "verts")
        assert isinstance(j3, np.ndarray) and j3.ndim == 2 and j3.shape[1] == 3 and j3.shape[0] >= 127, (f.name, "allSmplJoints3d")
        assert np.isfinite(j).all() and np.isfinite(v).all() and np.isfinite(j3).all(), (f.name, "non-finite")
        assert "_personId_" in f.name
        b = pickle.dumps(d, protocol=3); e = pickle.loads(b)
        assert all(np.array_equal(d[k], e[k]) and d[k].dtype == e[k].dtype for k in d), f.name
        z.writestr(f"predictions/{f.name}", b); images.add(f.name.split("_personId_")[0]); n += 1
h = hashlib.sha256()
with open(a.out, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
info = {"zip": str(a.out), "files": n, "images_with_predictions": len(images), "bytes": a.out.stat().st_size, "sha256": h.hexdigest(),
        "keys": sorted(d), "dtypes": {k: str(d[k].dtype) for k in d}, "joints_max_xy": [float(x) for x in np.asarray(d["joints"]).max(0)]}
a.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n"); print(json.dumps(info, indent=1))
