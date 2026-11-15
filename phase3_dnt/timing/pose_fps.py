"""Pose-regressor speed for the pose table's FPS column, with one harness on any GPU.

Harness (the L40S measurement of RUNLOG 148, made reproducible): batch 1, 256 x 256 crop (+ camera token for our model, focal =
image width as in the table), forward passes with torch.cuda.synchronize after each, 40 warm-up + 400 timed per setting,
median / p95 ms and FPS = 1000 / median. Settings: network only and network + the body model of each model's evaluated path
(ours: pure SMPL-X 2020 layer; PEAR: its EHM layer), each in FP32 (the table's number) and FP16 autocast. Repeated in rounds
with a pause, so laptop throttling shows up. Inputs are 64 real COCO-val person crops from the bundle (the evaluation's own
crops), kept on the GPU (no copies inside the timed loop). Before timing, each model's vertices on those crops are compared with
the server's reference, so a wrong checkpoint or broken install is caught.

  make-inputs (server only): python phase3_dnt/timing/pose_fps.py make-inputs --out DIR [--with-pear]
  run (any machine):         python pose_fps.py run --root GUAVA_ROOT --inputs pose_fps_inputs.npz --out RESULTS_DIR
                                 [--pear-ckpt pear_model.pt] [--models sprint pear] [--rounds 3] [--pause 20]
GUAVA_ROOT is this repository or the laptop package (guava_live): it must contain tools/eval_3dpw_standard.py and
third_party/PEAR. Writes RESULTS_DIR/pose_fps_<gpu>.json and prints a summary. Never writes inside GUAVA_ROOT."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

STUDENT_CONFIG = "configs/student_l70_v2.yaml"
KIND = {"sprint": ("sprint_raw", "smplx", True), "pear": ("pear", "ehm", False)}     # loader kind, body head, camera token
SERVER_SPRINT = Path("/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
HF_PEAR = Path.home() / (".cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
                         "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt")


def load_ev(root: Path):
    spec = importlib.util.spec_from_file_location("guava_eval_3dpw_standard", root / "tools" / "eval_3dpw_standard.py")
    ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
    return ev


class Model:
    """One pose model as the pose table evaluates it (tools/eval_author_protocol.py via phase3_dnt/pose_qual.Runner)."""

    def __init__(self, ev, which, ckpt, dev):
        self.ev, self.which, self.dev = ev, which, dev
        self.kind, head, self.camera = KIND[which]
        self.head = ev.effective_head(self.kind, head)
        cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
        try:
            self.model, self.report = ev.load_model(self.kind, Path(ckpt), STUDENT_CONFIG, 1, dev)
            self.model.eval()
            self.ehm = None
            if self.head == "ehm":
                from models.modules.ehm import EHM_v2
                self.ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval()
        finally:
            os.chdir(cwd)
        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)

    def forward(self, x, cam, body: bool, fp16: bool):
        """fp16 = autocast on the network only; the body model always runs in FP32 (its index ops need one dtype)."""
        kw = self.ev.camera_kwargs(self.kind, self.camera, cam, self.dev)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                out = self.model(x, **kw)
            if body:
                out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                       for k, v in out.items()}
                return self.ev.predicted_smplx_vertices(out, self.ehm, self.head)
        return out


def gpu_state(index: int) -> dict:
    q = "name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu,memory.used"
    try:
        r = subprocess.run(["nvidia-smi", f"--id={index}", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(zip(q.split(","), [v.strip() for v in r.split(",")]))
    except Exception as e:                                                                          # noqa: BLE001
        return {"error": repr(e)}


def ac_power():
    """Linux laptops: True / False if a mains adapter reports its state, else None."""
    for p in Path("/sys/class/power_supply").glob("*/online"):
        with contextlib.suppress(Exception):
            if (p.parent / "type").read_text().strip() == "Mains": return p.read_text().strip() == "1"
    return None


def timed(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); t[i] = time.perf_counter() - t0
    t *= 1000
    return {"median_ms": round(float(np.median(t)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2),
            "fps": round(1000 / float(np.median(t)), 1)}


def make_inputs(a):
    """Server: 64 COCO-val crops of the author's protocol (evenly spaced over persons with >= 12 visible joints) + reference
    vertices of each model (FP32) on this machine."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pose_qual as pq                                                                         # noqa: E402
    ev, _, dev = pq.setup(a.device)
    ds = pq.dataset(ev, "COCO-VAL"); ds.camera = True; ds.focals = None
    nvis = np.load(Path(a.npz_dir) / "pear_COCO-VAL.npz")["nvis"]
    elig = np.where(nvis >= 12)[0]; idx = elig[np.linspace(0, len(elig) - 1, a.n).round().astype(int)]
    imgs, cams = [], []
    for i in idx:
        b = ds[int(i)]; imgs.append(np.asarray(b["image"], np.uint8)); cams.append(np.asarray(b["cam_feat"], np.float32))
    imgs, cams = np.stack(imgs), np.stack(cams)
    out = {"images": imgs, "cam": cams, "index": idx, "imgname": np.array([ds.imgname[int(i)] for i in idx])}
    x = torch.from_numpy(imgs).to(dev).float() / 255.0; c = torch.from_numpy(cams).to(dev)
    for w in (("sprint", "pear") if a.with_pear else ("sprint",)):
        m = Model(ev, w, SERVER_SPRINT if w == "sprint" else HF_PEAR, dev)
        out[f"ref_verts_{w}"] = torch.cat([m.forward(x[k:k + 1], c[k:k + 1], True, False) for k in range(len(x))]).cpu().numpy().astype(np.float32)
        out[f"ref_step_{w}"] = np.array(m.report.get("step", -1))
    a.out.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out / "pose_fps_inputs.npz", **out)
    print("wrote", a.out / "pose_fps_inputs.npz", {k: v.shape for k, v in out.items()})


def run(a):
    root = a.root.resolve(); ev = load_ev(root); dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = False
    D = np.load(a.inputs)
    x_all = torch.from_numpy(D["images"]).to(dev).float() / 255.0; c_all = torch.from_numpy(D["cam"]).to(dev)
    gi = dev.index or 0
    res = {"gpu": torch.cuda.get_device_name(dev), "gpu_state_start": gpu_state(gi), "ac_power": ac_power(),
           "host": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
           "harness": {"batch": 1, "crop": 256, "warmup": a.warmup, "iters": a.iters, "rounds": a.rounds, "pause_s": a.pause,
                       "inputs": f"{len(x_all)} COCO-val crops on the GPU, cycled"},
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    ckpts = {"sprint": a.sprint_ckpt or next((p for p in (root / "live_assets/checkpoints/sprint_scratch_ema128k.pt", SERVER_SPRINT)
                                              if p.exists()), None),
             "pear": a.pear_ckpt or (HF_PEAR if HF_PEAR.exists() else None)}
    for w in a.models:
        if ckpts[w] is None or not Path(ckpts[w]).exists():
            print(f"[{w}] skipped: checkpoint not found ({ckpts[w]})", flush=True); res["models"][w] = {"skipped": f"no checkpoint: {ckpts[w]}"}
            continue
        try:
            m = Model(ev, w, ckpts[w], dev)
        except Exception as e:                                                                      # noqa: BLE001
            print(f"[{w}] could not be loaded: {e!r}", flush=True); res["models"][w] = {"error": repr(e)}; continue
        r = {"checkpoint": str(ckpts[w]), "step": m.report.get("step"), "params_M": m.params_m, "check": None, "rounds": []}
        ref = D[f"ref_verts_{w}"] if f"ref_verts_{w}" in D.files else None
        if ref is not None:                                                                         # correctness before timing
            V = torch.cat([m.forward(x_all[k:k + 1], c_all[k:k + 1], True, False) for k in range(len(x_all))]).cpu().numpy()
            d = np.linalg.norm(V - ref, axis=-1).mean(1) * 1000                                       # mean vertex distance per person, mm
            r["check"] = {"median_mm": round(float(np.median(d)), 3), "max_mm": round(float(d.max()), 3),
                          "pass": bool(np.median(d) < 1.0 and d.max() < 5.0)}
            print(f"[{w}] check vs server reference: median {np.median(d):.3f} mm, max {d.max():.3f} mm ->",
                  "PASS" if r["check"]["pass"] else "FAIL (wrong checkpoint or install?)", flush=True)
        k = [0]
        def step(body, fp16):
            i = k[0] % len(x_all); k[0] += 1
            return m.forward(x_all[i:i + 1], c_all[i:i + 1], body, fp16)
        for rd in range(a.rounds):
            if rd: time.sleep(a.pause)
            row = {}
            for body in (False, True):
                for prec in ("fp32", "fp16"):
                    row[f"{'with_body_model' if body else 'network'}_{prec}"] = timed(lambda: step(body, prec == "fp16"), a.warmup, a.iters)
            row["gpu_state_after"] = gpu_state(gi); r["rounds"].append(row)
            print(f"[{w}] round {rd + 1}/{a.rounds}: network fp32 {row['network_fp32']['median_ms']} ms ({row['network_fp32']['fps']} FPS), "
                  f"+ body model {row['with_body_model_fp32']['fps']} FPS, fp16 {row['network_fp16']['fps']} FPS; "
                  f"GPU {row['gpu_state_after'].get('temperature.gpu')} C, SM {row['gpu_state_after'].get('clocks.sm')} MHz", flush=True)
        for key in ("network_fp32", "network_fp16", "with_body_model_fp32", "with_body_model_fp16"):
            med = float(np.median([rr[key]["median_ms"] for rr in r["rounds"]]))
            r[key] = {"median_ms_over_rounds": round(med, 2), "fps": round(1000 / med, 1)}
        res["models"][w] = r; del m; torch.cuda.empty_cache()
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    slug = re.sub(r"[^a-z0-9]+", "_", res["gpu"].lower()).strip("_")
    a.out.mkdir(parents=True, exist_ok=True); f = a.out / f"pose_fps_{slug}.json"
    if f.exists(): f = a.out / f"pose_fps_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.json"            # never overwrite a result
    f.write_text(json.dumps(res, indent=1) + "\n")
    print("\nFPS (batch 1, FP32, network only = the table's column):")
    for w, r in res["models"].items():
        if "network_fp32" in r:
            print(f"  {w:6s} {r['network_fp32']['fps']:6.1f} FPS ({r['network_fp32']['median_ms_over_rounds']} ms)   "
                  f"+ body model {r['with_body_model_fp32']['fps']} FPS   check {'PASS' if (r['check'] or {}).get('pass') else r['check']}")
    print("wrote", f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-inputs"); m.add_argument("--out", type=Path, required=True); m.add_argument("--n", type=int, default=64)
    m.add_argument("--with-pear", action="store_true"); m.add_argument("--device", default="cuda:0")
    m.add_argument("--npz-dir", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/pose_qual")
    r = sub.add_parser("run"); r.add_argument("--root", type=Path, required=True); r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True); r.add_argument("--device", default="cuda:0")
    r.add_argument("--models", nargs="+", default=["sprint", "pear"], choices=["sprint", "pear"])
    r.add_argument("--sprint-ckpt", type=Path, default=None); r.add_argument("--pear-ckpt", type=Path, default=None)
    r.add_argument("--rounds", type=int, default=3); r.add_argument("--pause", type=float, default=20.0)
    r.add_argument("--warmup", type=int, default=40); r.add_argument("--iters", type=int, default=400)
    a = p.parse_args()
    make_inputs(a) if a.cmd == "make-inputs" else run(a)


if __name__ == "__main__":
    main()
