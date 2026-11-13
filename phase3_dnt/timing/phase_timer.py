"""Synchronized phase diagnostics, independent serial totals, and timing JSON provenance.

Per-phase synchronization perturbs execution. `total` is the instrumented diagnostic
loop, NOT live application latency/FPS. `measure_serial` runs a second pass with no
phase barriers; replay FPS is obtained separately from the unmodified live app.
Missing stages are null, never fabricated zeros. All units are milliseconds.
"""
from __future__ import annotations
import json
import math
import os
import platform
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np

PROTOCOL = {"warmup": 10, "iters": 50, "recon_warmup": 1, "recon_reps": 3,
            "precision": "as evaluated", "render_px": 512}
RUN_ID = uuid.uuid4().hex


def stats(ms):
    values = [float(x) for x in ms]
    if not values:
        return None
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("timings must be finite and nonnegative")
    a = np.asarray(values)
    return {"median_ms": round(float(np.median(a)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "mean_ms": round(float(a.mean()), 3), "n": len(values), "samples_ms": values}


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Phases:
    def __init__(self):
        self.t = {}; self._rec = False; self._t0 = None

    def start_run(self, record=True):
        sync(); self._rec = record; self._t0 = time.perf_counter()

    @contextmanager
    def __call__(self, name):
        sync(); t0 = time.perf_counter()
        try:
            yield
        finally:
            sync()
            if self._rec:
                self.t.setdefault(name, []).append((time.perf_counter() - t0) * 1000)

    def add(self, name, ms):
        if self._rec:
            self.t.setdefault(name, []).append(float(ms))

    def end_run(self):
        if self._t0 is None:
            raise RuntimeError("start_run must precede end_run")
        sync()
        if self._rec:
            self.t.setdefault("total", []).append((time.perf_counter() - self._t0) * 1000)
        self._t0 = None

    def summary(self):
        return {k: stats(v) for k, v in self.t.items()}

    def combined(self, *names):
        arrays = [self.t[k] for k in names if k in self.t]
        if not arrays:
            return None
        if len({len(v) for v in arrays}) != 1:
            raise ValueError("cannot combine unpaired stage samples")
        return stats([sum(v) for v in zip(*arrays)])


class NoPhases:
    """Same instrumentation interface, with no synchronization inside the pipeline."""
    def __call__(self, _name):
        return nullcontext()
    def add(self, _name, _ms):
        pass


def measure_serial(fn, warmup, iters):
    if warmup < 0 or iters < 1:
        raise ValueError("warmup >= 0 and iters >= 1 required")
    result = []
    p = NoPhases()
    for i in range(warmup + iters):
        sync(); t0 = time.perf_counter(); fn(p, i); sync()
        if i >= warmup:
            result.append((time.perf_counter() - t0) * 1000)
    return stats(result)


def env_info(power_note=""):
    info = {"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": platform.processor() or platform.machine(), "platform": platform.platform(),
            "power_note": power_note or os.environ.get("TIMING_POWER_NOTE", "")}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,power.limit",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    here = Path(__file__).resolve().parent
    try:
        info["git_sha"] = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=10).stdout.strip() or os.environ.get("TIMING_GIT_SHA", "")
        info["git_dirty"] = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", str(here)],
                                capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        info["git_sha"] = os.environ.get("TIMING_GIT_SHA", "")
    return info


def write(out, method, row, clips, notes, protocol=None, power_note=""):
    out = Path(out).resolve()
    if out.is_relative_to('/raid') or out.is_relative_to('/vault'):
        raise ValueError("write timing outputs to a new local directory, not /raid or /vault")
    if out.exists() and json.loads(out.read_text()).get("run_id") != RUN_ID:
        raise FileExistsError(f"refusing to overwrite a prior run: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"schema_version": 2, "run_id": RUN_ID, "method": method, "row": row, "env": env_info(power_note),
           "protocol": protocol or PROTOCOL, "clips": clips, "notes": notes,
           "measurement_note": "phase totals contain instrumentation barriers; inverse diagnostic latency is not observed live FPS"}
    tmp = out.with_name(out.name + '.' + RUN_ID + '.tmp')
    with tmp.open('x') as f:
        json.dump(res, f, indent=1, allow_nan=False)
    os.replace(tmp, out)
    print("wrote", out, flush=True)
    return res
