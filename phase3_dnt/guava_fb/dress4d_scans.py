"""GUAVA-FB pilot D: per-clip cache of the 4D-DRESS scans used by the converted rows (scan_geom.py describes the layout).
Input (read-only): 4D-DRESS extracted/<subj>/<outfit>/<take>/ Meshes_pkl/mesh-f*.pkl (80k-face scan), Semantic/labels/label-f*.pkl
(per-vertex garment label), Capture/cameras.pkl; the converted clip's optim_tracking_ehm.pkl (record["dress4d"]: frame, R_virtual;
smplx_coeffs.camera_RT_params). Output: <out>/<clip>/ (written to .tmp, then renamed). Train and val clips (val: evaluation only).
The scan pickles reference trimesh.TrackedArray (an ndarray subclass); a stub ndarray subclass unpickles them without trimesh.
Usage: python dress4d_scans.py --data O/data/dress4d --out O/data/dress4d_scan [--workers 16] [--limit N]"""
import argparse, json, os, pickle, shutil, sys, time
from pathlib import Path
import numpy as np

DATA = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/4d_dress/extracted")
VMAX, NF = 45000, 80000
FLIP = np.diag([-1.0, -1.0, 1.0])


class _Tracked(np.ndarray):
    pass


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        return _Tracked if m.startswith("trimesh") else super().find_class(m, n)


def row_A(rec, cam):
    """4D-DRESS world -> GUAVA world of the row (3 x 4)."""
    Rv = np.asarray(rec["dress4d"]["R_virtual"], np.float64); E = np.asarray(cam["extrinsics"], np.float64); Rc, tc = E[:, :3], E[:, 3]
    tp = FLIP @ np.asarray(rec["smplx_coeffs"]["camera_RT_params"], np.float64)[:, 3]
    return np.c_[Rv @ Rc, Rv @ tc - tp]


def one(args):
    clip_dir, out_root = args; clip_dir = Path(clip_dir); name = clip_dir.name; out = Path(out_root) / name
    if (out / "meta.json").exists(): return name, "skip", 0.0
    t0 = time.time(); tmp = Path(out_root) / f".{name}.tmp"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    recs = pickle.load(open(clip_dir / "optim_tracking_ehm.pkl", "rb"))
    d0 = next(iter(recs.values()))["dress4d"]; T = DATA / d0["subject"] / d0["outfit"] / d0["take"]
    cams = pickle.load(open(T / "Capture" / "cameras.pkl", "rb"))
    frames = sorted({r["dress4d"]["frame"] for r in recs.values()})
    A = {k: row_A(r, cams[k.split("_", 1)[0]]).tolist() for k, r in recs.items()}
    V = np.zeros((len(frames), VMAX, 3), np.float32); L = np.zeros((len(frames), VMAX), np.uint8)
    Fc = np.zeros((len(frames), NF, 3), np.uint16); nv = np.zeros(len(frames), np.int32)
    for i, fr in enumerate(frames):
        m = _U(open(T / "Meshes_pkl" / f"mesh-f{fr}.pkl", "rb")).load()
        v = np.asarray(m["vertices"], np.float32); f = np.asarray(m["faces"], np.int64)
        lab = np.asarray(pickle.load(open(T / "Semantic" / "labels" / f"label-f{fr}.pkl", "rb"))["scan_labels"])
        assert len(v) <= VMAX and f.shape == (NF, 3) and f.max() < len(v) and len(lab) == len(v), (name, fr, v.shape, f.shape, lab.shape)
        V[i, :len(v)] = v; Fc[i] = f; L[i, :len(v)] = lab.astype(np.uint8); nv[i] = len(v)
    for n_, x in (("verts", V), ("faces", Fc), ("labels", L), ("nverts", nv)): np.save(tmp / f"{n_}.npy", x)
    (tmp / "meta.json").write_text(json.dumps({"clip": name, "take_dir": str(T), "frames": frames, "A": A, "vmax": VMAX}))
    if out.exists(): shutil.rmtree(out)
    tmp.rename(out); return name, "ok", time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    clips = [Path(a.data) / c["name"] for c in json.loads((Path(a.data) / "index.json").read_text())["clips"]]
    if a.limit: clips = clips[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for name, st, dt in ex.map(one, [(str(c), a.out) for c in clips]):
            n += 1; print(f"{n}/{len(clips)} {name} {st} {dt:.1f}s elapsed {time.time() - t0:.0f}s", flush=True)
