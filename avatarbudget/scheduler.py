from __future__ import annotations

import time
from dataclasses import dataclass

from .config import SchedulerConfig


@dataclass(frozen=True)
class DeadlineResult:
    work_ms: float
    frame_ms: float
    slept_ms: float
    deadline_missed: bool


class HardFrameScheduler:
    """Pace output at a fixed deadline and expose misses instead of hiding them."""

    def __init__(self, config: SchedulerConfig, clock=time.perf_counter, sleeper=time.sleep):
        self.config = config
        self.clock = clock
        self.sleeper = sleeper
        self._start: float | None = None
        self.frames = 0
        self.deadline_misses = 0
        self.overrun_ema_ms = 0.0

    def begin_frame(self) -> None:
        if self._start is not None:
            raise RuntimeError("finish the current frame before beginning another")
        self._start = self.clock()

    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (self.clock() - self._start) * 1000.0

    def remaining_ms(self) -> float:
        return max(0.0, self.config.deadline_ms - self.elapsed_ms())

    def finish_frame(self) -> DeadlineResult:
        if self._start is None:
            raise RuntimeError("begin_frame() was not called")
        work_ms = self.elapsed_ms()
        slept_ms = 0.0
        if self.config.sleep_to_rate and work_ms < self.config.deadline_ms:
            slept_ms = self.config.deadline_ms - work_ms
            self.sleeper(slept_ms / 1000.0)
        frame_ms = self.elapsed_ms()
        missed = work_ms > self.config.deadline_ms
        overrun = max(0.0, work_ms - self.config.deadline_ms)
        self.overrun_ema_ms = 0.8 * self.overrun_ema_ms + 0.2 * overrun
        self.frames += 1
        self.deadline_misses += int(missed)
        self._start = None
        return DeadlineResult(work_ms, frame_ms, slept_ms, missed)

    @property
    def achieved_deadline_rate(self) -> float:
        return 1.0 - self.deadline_misses / self.frames if self.frames else 0.0

    @property
    def routing_budget_ms(self) -> float:
        """Tighten the next action budget after measured deadline overruns."""
        return max(1.0, self.config.deadline_ms - self.overrun_ema_ms)
