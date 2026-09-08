from __future__ import annotations

import contextlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import torch


def percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class StageSummary:
    count: int
    mean_ms: float
    p95_ms: float
    p99_ms: float
    clock: str


class StageProfiler:
    """CUDA-event timings for GPU stages and perf-counter timings for host stages."""

    def __init__(self, window: int = 300, warmup_frames: int = 10):
        self.window = window
        self.warmup_frames = warmup_frames
        self.frame_index = 0
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._clock: dict[str, str] = {}
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def next_frame(self) -> None:
        self.frame_index += 1

    @contextlib.contextmanager
    def cpu(self, name: str):
        start = time.perf_counter()
        yield
        if self.frame_index >= self.warmup_frames:
            self._samples[name].append((time.perf_counter() - start) * 1000.0)
            self._clock[name] = "cpu_perf_counter"

    @contextlib.contextmanager
    def cuda(self, name: str):
        if not torch.cuda.is_available():
            with self.cpu(name):
                yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        if self.frame_index >= self.warmup_frames:
            self._pending.append((name, start, end))
            self._clock[name] = "cuda_event"

    def collect_cuda(self, synchronize: bool = True) -> None:
        if not self._pending:
            return
        if synchronize:
            torch.cuda.synchronize()
        remaining = []
        for name, start, end in self._pending:
            if end.query():
                self._samples[name].append(start.elapsed_time(end))
            else:
                remaining.append((name, start, end))
        self._pending = remaining

    def add(self, name: str, milliseconds: float, clock: str = "cpu_perf_counter") -> None:
        if self.frame_index >= self.warmup_frames:
            self._samples[name].append(float(milliseconds))
            self._clock[name] = clock

    def summary(self) -> dict[str, StageSummary]:
        result = {}
        for name, values in sorted(self._samples.items()):
            samples = list(values)
            result[name] = StageSummary(
                count=len(samples),
                mean_ms=sum(samples) / len(samples) if samples else 0.0,
                p95_ms=percentile(samples, 0.95),
                p99_ms=percentile(samples, 0.99),
                clock=self._clock.get(name, "unknown"),
            )
        return result

    def as_dict(self) -> dict[str, dict[str, float | int | str]]:
        return {
            name: {
                "count": item.count,
                "mean_ms": item.mean_ms,
                "p95_ms": item.p95_ms,
                "p99_ms": item.p99_ms,
                "clock": item.clock,
            }
            for name, item in self.summary().items()
        }
