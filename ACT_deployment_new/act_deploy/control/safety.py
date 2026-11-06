"""Last line of software safety before a target is sent to the arms (runs every 10 ms).

Order: range clamp -> leash to measured pose -> optional low-pass -> velocity limit.
  range    model-driven dims stay inside the demonstration range +- margin; grippers 0..1
  leash    (policy only) a joint target may not be further than max_deviation from where the
           joint actually is -> a wild prediction cannot yank the arm
  low-pass first-order filter with time constant tau (0 = off)
  velocity per-step change limited to v_max * dt (joints rad/s, grippers 1/s)
"""

from __future__ import annotations

import math

import numpy as np

from act_deploy import GRIPPER_IDX, JOINT_IDX


class SafetyFilter:
    def __init__(self, *, max_joint_velocity: float, max_gripper_velocity: float,
                 max_deviation: float, lo=None, hi=None, smoothing_tau_s: float = 0.0):
        self.vmax = np.full(14, float(max_joint_velocity))
        self.vmax[list(GRIPPER_IDX)] = float(max_gripper_velocity)
        self.max_dev = float(max_deviation)
        self.lo = np.full(14, -np.inf) if lo is None else np.asarray(lo, float).copy()
        self.hi = np.full(14, np.inf) if hi is None else np.asarray(hi, float).copy()
        self.lo[list(GRIPPER_IDX)] = np.maximum(self.lo[list(GRIPPER_IDX)], 0.0)
        self.hi[list(GRIPPER_IDX)] = np.minimum(self.hi[list(GRIPPER_IDX)], 1.0)
        self.tau = float(smoothing_tau_s)
        self.last: np.ndarray | None = None
        self.counts = {"range": 0, "leash": 0, "rate": 0}
        self._joints = np.array(JOINT_IDX)

    def reset(self, current) -> None:
        self.last = np.asarray(current, dtype=np.float64).copy()

    def set_limits(self, lo, hi) -> None:
        self.lo = np.asarray(lo, float).copy()
        self.hi = np.asarray(hi, float).copy()

    def __call__(self, target, measured, dt: float, *, leash: bool = False, clamp_range: bool = True) -> np.ndarray:
        x = np.asarray(target, dtype=np.float64).copy()
        if clamp_range:
            y = np.clip(x, self.lo, self.hi)
            if not np.allclose(x, y):
                self.counts["range"] += 1
            x = y
        else:
            x[[6, 13]] = np.clip(x[[6, 13]], 0.0, 1.0)
        if leash and measured is not None:
            j = self._joints
            m = np.asarray(measured, dtype=np.float64)
            y = np.clip(x[j], m[j] - self.max_dev, m[j] + self.max_dev)
            if not np.allclose(x[j], y):
                self.counts["leash"] += 1
            x[j] = y
        if self.last is None:
            self.last = x.copy()
            return x
        if self.tau > 0:
            a = 1.0 - math.exp(-max(dt, 1e-4) / self.tau)
            x = self.last + (x - self.last) * a
        step = self.vmax * max(dt, 1e-3)
        y = np.clip(x, self.last - step, self.last + step)
        if not np.allclose(x, y):
            self.counts["rate"] += 1
        self.last = y
        return y.copy()
