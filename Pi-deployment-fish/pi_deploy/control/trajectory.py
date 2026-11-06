"""Smooth point-to-point motion for initialisation / return-to-zero.

Quintic "minimum-jerk" profile s(u) = 10u^3 - 15u^4 + 6u^5: position, velocity AND acceleration
are continuous and zero at both ends, so the arm starts and stops gently.  The duration is the
smallest one that respects the joint velocity / acceleration and gripper velocity limits:
    peak velocity     = 1.875 * distance / T
    peak acceleration = 5.774 * distance / T^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pi_deploy import GRIPPER_IDX, JOINT_IDX


@dataclass
class Segment:
    start: np.ndarray
    goal: np.ndarray
    duration: float

    def at(self, t: float) -> np.ndarray:
        u = min(max(t / self.duration, 0.0), 1.0) if self.duration > 0 else 1.0
        s = u ** 3 * (10 + u * (-15 + 6 * u))
        return self.start + (self.goal - self.start) * s


def plan_duration(start, goal, *, max_joint_velocity, max_joint_acceleration,
                  max_gripper_velocity, min_duration_s) -> float:
    d = np.abs(np.asarray(goal, float) - np.asarray(start, float))
    dj = float(d[list(JOINT_IDX)].max())
    dg = float(d[list(GRIPPER_IDX)].max())
    return max(min_duration_s,
               1.875 * dj / max_joint_velocity,
               math.sqrt(5.774 * dj / max_joint_acceleration),
               1.875 * dg / max_gripper_velocity)


class Route:
    """Chain of quintic segments through waypoints; evaluated by wall time."""

    def __init__(self, start, waypoints: list, limits: dict, t0: float):
        self.segments: list[Segment] = []
        cur = np.asarray(start, dtype=np.float64).copy()
        for wp in waypoints:
            goal = np.asarray(wp, dtype=np.float64).copy()
            dur = plan_duration(cur, goal, **limits)
            self.segments.append(Segment(cur, goal, dur))
            cur = goal
        self.t0 = t0
        self.duration = sum(s.duration for s in self.segments)
        self.goal = cur

    def at(self, t: float) -> np.ndarray:
        tau = t - self.t0
        for seg in self.segments:
            if tau <= seg.duration:
                return seg.at(tau)
            tau -= seg.duration
        return self.goal.copy()

    def done(self, t: float) -> bool:
        return t - self.t0 >= self.duration

    def remaining(self, t: float) -> float:
        return max(0.0, self.duration - (t - self.t0))
