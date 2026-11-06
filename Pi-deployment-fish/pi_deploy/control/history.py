"""Recent robot14 readings (written by the 100 Hz control loop) with time interpolation.

Used for (a) the robot state at the observation time and (b) the v3 velocity feature
(Pi least-squares slope over the past 100 ms).  Times are time.monotonic() seconds.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class StateHistory:
    def __init__(self, seconds: float = 3.0, rate_hz: float = 100.0):
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max(4, int(seconds * rate_hz * 1.5)))
        self._lock = threading.Lock()

    def add(self, t: float, state) -> None:
        with self._lock:
            if self._buf and t <= self._buf[-1][0]:
                return
            self._buf.append((t, np.asarray(state, dtype=np.float64).copy()))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            return (self._buf[-1][0], self._buf[-1][1].copy()) if self._buf else None

    def covers(self, t: float) -> bool:
        with self._lock:
            return bool(self._buf) and self._buf[0][0] <= t

    def velocity(self, t: float, window_s: float) -> tuple[np.ndarray, bool]:
        """Pi timeseries.causal_velocity: least-squares over (t-window,t], fallback last two."""
        with self._lock:
            samples = [(ts, v.copy()) for ts, v in self._buf if ts <= t]
        if len(samples) < 2:
            return np.zeros(14), True
        in_window = [(ts, v) for ts, v in samples if ts > t - window_s]
        selected = in_window if len(in_window) >= 2 else samples[-2:]
        times = np.array([ts - selected[-1][0] for ts, _ in selected])
        x = np.stack([v for _, v in selected])
        tc = times - times.mean()
        denom = float((tc * tc).sum())
        if denom <= 0:
            return np.zeros(14), True
        return (tc[:, None] * (x - x.mean(0))).sum(0) / denom, False

    def at(self, t: float) -> np.ndarray:
        with self._lock:
            if not self._buf:
                raise RuntimeError("状态历史为空")
            times = np.fromiter((s[0] for s in self._buf), dtype=np.float64, count=len(self._buf))
            states = [s[1] for s in self._buf]
        if t <= times[0]:
            return states[0].copy()
        if t >= times[-1]:
            return states[-1].copy()
        i = int(np.searchsorted(times, t))
        a = (t - times[i - 1]) / (times[i] - times[i - 1])
        return states[i - 1] + (states[i] - states[i - 1]) * a
