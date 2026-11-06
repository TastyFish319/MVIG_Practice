"""Turn checkpoint-FPS action chunks ("plans") into a smooth 100 Hz target stream.

Timing model (all time.monotonic() seconds):
    a plan predicted from an observation taken at t_obs has knot k at
        t_obs + (k + action_offset_frames) / fps          k = 0 .. horizon-1
    between knots we use cubic (Catmull-Rom) interpolation -> continuous velocity, no coarse action stair.
    Inference latency simply means the plan arrives a bit later; it is still evaluated at the
    true wall time, so nothing is "replayed late".

Two execution modes:
  chunk     commit to the newest plan; ask for the next one so that it ARRIVES after
            `exec_steps` steps (request is sent `latency` earlier); cross-fade old->new over
            `blend_s`.  Robust: the arm follows one coherent plan at a time.
  ensemble  ask for a new plan at `rate_hz`; average every plan covering t with weights
            exp(-m*i), i=0 = oldest (ACT temporal ensembling).  Smoothest, slightly laggier.
Both modes start from a constant "hold" value (the pose when the episode started) and fade into
the first plan, so starting the policy never jumps.
"""

from __future__ import annotations

import math
import threading

import numpy as np


class Plan:
    def __init__(self, t_obs: float, fps: float, offset: int, points: np.ndarray, t_arrive: float):
        self.t_obs = t_obs
        self.dt = 1.0 / fps
        self.offset = offset
        self.p = np.asarray(points, dtype=np.float64)
        self.t_arrive = t_arrive
        self.n = len(self.p)

    @property
    def t_last(self) -> float:
        return self.t_obs + (self.n - 1 + self.offset) * self.dt

    def at(self, t: float) -> np.ndarray:
        f = (t - self.t_obs) / self.dt - self.offset
        if f <= 0:
            return self.p[0].copy()
        if f >= self.n - 1:
            return self.p[-1].copy()
        i = int(math.floor(f))
        u = f - i
        p0 = self.p[max(i - 1, 0)]
        p1 = self.p[i]
        p2 = self.p[i + 1]
        p3 = self.p[min(i + 2, self.n - 1)]
        # Catmull-Rom (cubic Hermite with central-difference tangents)
        m1 = (p2 - p0) / 2
        m2 = (p3 - p1) / 2
        u2, u3 = u * u, u * u * u
        return ((2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * m1
                + (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * m2)


class ActionScheduler:
    def __init__(self, *, mode: str, fps: float, horizon: int, offset: int,
                 exec_steps: int = 10, blend_s: float = 0.15, ensemble_m: float = 0.01,
                 rate_hz: float = 15.0):
        if mode not in ("chunk", "ensemble"):
            raise ValueError(f"未知执行方式 {mode}")
        self.mode, self.fps, self.horizon, self.offset = mode, fps, horizon, offset
        self.dt = 1.0 / fps
        self.exec_steps = int(exec_steps)
        self.blend_s = float(blend_s)
        self.m = float(ensemble_m)
        self.rate_hz = float(rate_hz)
        if mode == "chunk":
            spare = (horizon - 1 + offset) * self.dt - exec_steps * self.dt - blend_s
            if exec_steps < 1 or spare < 0.1:
                raise ValueError(f"exec_steps={exec_steps} + blend_s={blend_s} 超出 chunk 长度 "
                                 f"{horizon}步，至少留 0.1s 余量")
        self._lock = threading.Lock()
        self._plans: list[Plan] = []
        self._base: np.ndarray | None = None
        self._pending = False
        self._last_request = -1e9

    # ------------------------------------------------------------------ feeding
    def reset(self, hold14) -> None:
        with self._lock:
            self._plans = []
            self._base = np.asarray(hold14, dtype=np.float64).copy()
            self._pending = False
            self._last_request = -1e9

    def mark_request(self, t: float) -> None:
        with self._lock:
            self._pending = True
            self._last_request = t

    def request_failed(self) -> None:
        with self._lock:
            self._pending = False

    def add(self, plan: Plan) -> None:
        with self._lock:
            self._plans.append(plan)
            self._pending = False
            if len(self._plans) > self.horizon:
                self._plans = self._plans[-self.horizon:]

    def wants_inference(self, now: float, latency_s: float) -> bool:
        with self._lock:
            if self._pending:
                return False
            if not self._plans:
                return True
            if self.mode == "ensemble":
                return now - self._last_request >= 1.0 / self.rate_hz
            cur = self._plans[-1]
            lead = latency_s + 0.015
            return now - cur.t_obs >= self.exec_steps * self.dt - lead

    # ------------------------------------------------------------------ output
    def _alpha(self, plan: Plan, t: float) -> float:
        if self.blend_s <= 0:
            return 1.0 if t >= plan.t_arrive else 0.0
        return min(max((t - plan.t_arrive) / self.blend_s, 0.0), 1.0)

    def target_at(self, t: float, measured_hold=None) -> tuple[np.ndarray | None, dict]:
        """(target14 or None if never fed, info{starved_s, n_plans, plan_age_s}).
        starved_s > 0: the newest plan is used up (holding its last point) for that long."""
        with self._lock:
            plans = list(self._plans)
            base = None if self._base is None else self._base.copy()
        if base is None and not plans:
            return None, {"starved_s": 0.0, "n_plans": 0, "plan_age_s": None}
        info = {"starved_s": max(0.0, t - plans[-1].t_last) if plans else 0.0,
                "n_plans": len(plans),
                "plan_age_s": (t - plans[-1].t_obs) if plans else None}
        expired = [p for p in plans if t > p.t_last]
        plans = [p for p in plans if t <= p.t_last]
        if expired and base is None and measured_hold is not None:
            base = np.asarray(measured_hold, dtype=np.float64).copy()
        if not plans and expired:
            return None, info
        if not plans:
            return base, info
        if self.mode == "chunk":
            # drop everything older than the newest fully-faded-in plan
            j = 0
            for k in range(len(plans) - 1, -1, -1):
                if self._alpha(plans[k], t) >= 1.0:
                    j = k
                    break
            if j > 0 or self._alpha(plans[0], t) >= 1.0:
                with self._lock:
                    if self._plans[:len(plans)] == plans:
                        self._plans = self._plans[j:]
                        self._base = None
                plans, base = plans[j:], None
            value = base if base is not None else plans[0].at(t)
            for k, plan in enumerate(plans):
                if k == 0 and base is None:
                    continue
                a = self._alpha(plan, t)
                value = value * (1 - a) + plan.at(t) * a
            return value, info
        # ensemble
        live = [p for p in plans if t <= p.t_last] or plans[-1:]
        # each plan's weight also fades in over blend_s, so a new plan never enters abruptly
        w = np.exp(-self.m * np.arange(len(live))) * np.array([self._alpha(p, t) for p in live])
        if w.sum() <= 1e-9:
            w = np.ones(len(live))
        ens = sum(wi * p.at(t) for wi, p in zip(w, live)) / w.sum()
        if base is not None:
            a = self._alpha(plans[0], t)
            if a >= 1.0:
                with self._lock:
                    self._base = None
            ens = base * (1 - a) + ens * a
        return ens, info
