"""Pause a long RAGA check while the laptop is too hot, then resume.

The quality check took this TensorBook's CPU to 100 C (its limit) twice, so long scripts call guard() once per item:
when the CPU package or the GPU reaches its limit, it sleeps until both are a few degrees below, then returns.

    from heat_guard import HeatGuard
    hg = HeatGuard(max_cpu=92, max_gpu=88)
    for item in items:
        hg.guard()
        ...
    print(hg.summary())
"""
import time
from pathlib import Path


def _cpu_reader():
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (h / "name").read_text().strip() == "coretemp":            # temp1 = "Package id 0"
                f = h / "temp1_input"
                return lambda: int(f.read_text()) / 1000.0
        except OSError:
            continue
    return lambda: None


def _gpu_reader():
    try:
        import pynvml
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: float(pynvml.nvmlDeviceGetTemperature(h, 0))
    except Exception:
        return lambda: None


class HeatGuard:
    def __init__(self, max_cpu=92.0, max_gpu=88.0, resume_margin=7.0, poll_s=3.0):
        self.max_cpu, self.max_gpu, self.margin, self.poll = max_cpu, max_gpu, resume_margin, poll_s
        self.cpu, self.gpu = _cpu_reader(), _gpu_reader()
        self.paused_s, self.pauses, self.peak_cpu, self.peak_gpu = 0.0, 0, 0.0, 0.0

    def temps(self):
        c, g = self.cpu(), self.gpu()
        self.peak_cpu = max(self.peak_cpu, c or 0.0); self.peak_gpu = max(self.peak_gpu, g or 0.0)
        return c, g

    def _hot(self, margin):
        c, g = self.temps()
        return (bool(self.max_cpu) and c is not None and c >= self.max_cpu - margin) or \
               (bool(self.max_gpu) and g is not None and g >= self.max_gpu - margin), c, g

    def guard(self):
        """Return at once when cool enough; otherwise sleep until both temperatures are `resume_margin` below the limits."""
        hot, c, g = self._hot(0.0)
        if not hot:
            return 0.0
        t0 = time.perf_counter(); self.pauses += 1
        print(f"  heat guard: CPU {c:.0f} C, GPU {g:.0f} C -> pausing", flush=True)
        while self._hot(self.margin)[0]:
            time.sleep(self.poll)
        dt = time.perf_counter() - t0; self.paused_s += dt
        c, g = self.temps(); print(f"  heat guard: resumed after {dt:.0f} s (CPU {c:.0f} C, GPU {g:.0f} C)", flush=True)
        return dt

    def summary(self):
        return (f"heat guard: {self.pauses} pauses, {self.paused_s:.0f} s paused; peak CPU {self.peak_cpu:.0f} C, "
                f"peak GPU {self.peak_gpu:.0f} C (limits {self.max_cpu:.0f} / {self.max_gpu:.0f} C)")
