"""Engine: the robot state machine, the 100 Hz control loop and the policy (inference) thread.

    IDLE --Enter--> PREPARING --> READY --Enter--> RUNNING --f / time limit--> RETURNING --> ZERO
     ^                (smooth move to start pose)      |  p <-> PAUSED            (via start pose)  |
     |                                                 '-- space / error --> ESTOP (hold here)     |
     '-------------------------------------------- Enter (next episode) <-------------------------'
    z: smooth return to zero      i: smooth move to start pose     r/q: return to zero, release, exit
    worker FAULT / lost link -> FAULT (the worker holds by itself); z or i recovers first

Threads:
  control (this module, 100 Hz)  the ONLY one that commands the arms; all phase changes happen here
  policy                          camera + state history -> model server -> plan -> scheduler
  console (app/console.py)        keys -> Engine.command(); renders Engine.snapshot()
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from act_deploy import GRIPPER_IDX, JOINT_IDX
from act_deploy.app.recorder import EpisodeRecorder
from act_deploy.control.history import StateHistory
from act_deploy.control.safety import SafetyFilter
from act_deploy.control.scheduler import ActionScheduler, Plan
from act_deploy.control.trajectory import Route
from act_deploy.hw.arms import ArmCommError, ArmFault
from act_deploy.model.features import build_state, decode_actions, letterbox_rgb

# phase -> (Chinese name, colour, what the operator can do next)
PHASES = {
    "IDLE": ("待命（上电原位）", "cyan", "Enter 移动到初始姿态 · r 直接释放退出"),
    "PREPARING": ("正在移动到初始姿态", "yellow", "空格 急停"),
    "READY": ("已就绪（初始姿态）", "green", "摆好场景后按 Enter 开始执行 · z 回零位"),
    "RUNNING": ("策略执行中", "magenta", "f 结束并回零 · p 暂停 · 空格 急停"),
    "PAUSED": ("已暂停（原地保持）", "yellow", "p/Enter 继续 · f 结束并回零 · z 回零位"),
    "RETURNING": ("正在回零位", "yellow", "空格 急停"),
    "ZERO": ("零位保持", "green", "Enter 下一次（先回初始姿态）· r 释放并退出"),
    "ESTOP": ("急停·原地保持", "red", "z 紧急回零 · i 回初始姿态 · r 回零后释放退出"),
    "FAULT": ("故障·worker 自行保持", "red", "z 复位并回零 · i 复位并回初始姿态 · r 回零后释放退出"),
    "RELEASING": ("正在释放电机", "yellow", "请稍候"),
    "RELEASED": ("已释放", "cyan", "程序即将退出"),
}
MOVING = ("PREPARING", "RUNNING", "RETURNING")


@dataclass
class EngineOptions:
    mode: str  # real | shadow | sim
    rate_hz: float
    motion: dict
    poses: dict
    safety: dict
    policy: dict
    preset: dict
    recording: dict
    max_frame_age_s: float
    start_pose: np.ndarray
    max_episode_s: float
    session_dir: Path
    meta: dict


class Engine:
    def __init__(self, opts: EngineOptions, spec, arms, cams, client):
        self.o, self.spec, self.arms, self.cams, self.client = opts, spec, arms, cams, client
        self.dt = 1.0 / opts.rate_hz
        self.zero = np.asarray(opts.poses["zero"], np.float64)
        self.start_pose = np.asarray(opts.start_pose, np.float64)
        self.limits = {k: opts.motion[k] for k in ("max_joint_velocity", "max_joint_acceleration",
                                                   "max_gripper_velocity", "min_duration_s")}
        self.history = StateHistory(seconds=3.0, rate_hz=opts.rate_hz)
        p = opts.preset
        self.scheduler = ActionScheduler(
            mode=p["mode"], fps=spec.fps, horizon=spec.horizon, offset=spec.action_offset_frames,
            exec_steps=p.get("exec_steps", 10), blend_s=p.get("blend_s", 0.15),
            ensemble_m=p.get("ensemble_m", 0.01), rate_hz=p.get("rate_hz", 15.0))
        lo, hi = self._workspace()
        s = opts.safety
        self.safety = SafetyFilter(max_joint_velocity=s["max_joint_velocity"],
                                   max_gripper_velocity=s["max_gripper_velocity"],
                                   max_deviation=s["max_deviation"], lo=lo, hi=hi,
                                   smoothing_tau_s=p.get("smoothing_tau_s", 0.0))
        self.phase = "STARTING"
        self.hold: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.state: np.ndarray | None = None
        self.policy_target: np.ndarray | None = None  # what the policy wants (also in shadow mode)
        self.route: Route | None = None
        self.route_next = "READY"
        self.route_label = ""
        self.settle_since: float | None = None
        self.settle_ok = 0
        self.release_requested = False
        self.fault_reason: str | None = None
        self.episode_index = 0
        self.episode_id = 0  # changes whenever an episode starts/ends -> stale plans are dropped
        self.ep_start = 0.0
        self.ep_hold: np.ndarray | None = None
        self.recorder: EpisodeRecorder | None = None
        self.results: list[dict] = []
        self.events: deque = deque(maxlen=200)
        self.n_events = 0
        self.cmd_q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.comm_fail = 0
        self.submit_fail = 0
        self.loop_late_ms = deque(maxlen=500)
        # policy statistics
        self.lat = deque(maxlen=50)  # request -> plan arrival [s]
        self.infer_ms = deque(maxlen=50)
        self.n_infer = 0
        self.policy_fail = 0
        self.last_policy_error: str | None = None
        self.velocity_padded = False
        self.warmup_pending = opts.policy["live_warmup_iterations"]
        self.warmup_report: str | None = None
        self.warmup_attempts = 0
        self.sched_info: dict = {}
        self._threads: list[threading.Thread] = []

    def _workspace(self):
        """Joint range the policy may command: demo range +- margin for model-driven joints."""
        lo, hi = np.full(14, -np.inf), np.full(14, np.inf)
        if self.spec.state_min is not None and self.spec.state_max is not None:
            m = self.o.safety["workspace_margin"]
            for i in self.spec.controlled:
                if i not in GRIPPER_IDX:
                    lo[i] = self.spec.state_min[i] - m
                    hi[i] = self.spec.state_max[i] + m
        return lo, hi

    # ================================================================== public API
    def start(self) -> None:
        state = self.arms.read(allow_fault=True)
        if getattr(self.arms, "fault", None):
            self.arms.reset_fault()
            state = self.arms.read()
        self.arms.begin(state)
        self.state = state
        self.hold = state.copy()
        self.cmd = state.copy()
        self.safety.reset(state)
        self.history.add(time.monotonic(), state)
        self.phase = "IDLE"
        self.event("info", "机械臂已上电，原地保持（上电原位）")
        for target, name in ((self._control_loop, "control"), (self._policy_loop, "policy")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def command(self, name: str, arg=None) -> None:
        self.cmd_q.put((name, arg))

    def event(self, level: str, msg: str) -> None:
        with self.lock:
            self.events.append((time.strftime("%H:%M:%S"), level, msg))
            self.n_events += 1
        try:
            with open(self.o.session_dir / "events.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")
        except OSError:
            pass

    def snapshot(self) -> dict:
        now = time.monotonic()
        route = self.route
        with self.lock:
            events = list(self.events)[-8:]
        lat = np.asarray(self.lat) * 1000 if self.lat else None
        snap = {
            "phase": self.phase, "mode": self.o.mode, "events": events,
            "state": None if self.state is None else self.state.copy(),
            "cmd": None if self.cmd is None else self.cmd.copy(),
            "policy_target": None if self.policy_target is None else self.policy_target.copy(),
            "episode": self.episode_index, "results": list(self.results[-5:]),
            "elapsed": now - self.ep_start if self.phase in ("RUNNING", "PAUSED") else None,
            "max_episode_s": self.o.max_episode_s,
            "route_remaining": route.remaining(now) if route is not None else None,
            "route_label": self.route_label,
            "n_infer": self.n_infer, "policy_fail": self.policy_fail, "policy_error": self.last_policy_error,
            "lat_ms": None if lat is None else (float(np.median(lat)), float(lat.max())),
            "infer_ms": float(np.median(self.infer_ms)) if self.infer_ms else None,
            "velocity_padded": self.velocity_padded, "sched": dict(self.sched_info),
            "safety": dict(self.safety.counts), "fault": self.fault_reason,
            "loop_late_ms": float(np.max(self.loop_late_ms)) if self.loop_late_ms else 0.0,
            "arm_age_ms": getattr(self.arms, "last_sample_age_ms", 0.0),
            "cams": self.cams.status() if self.cams else {}, "warmup": self.warmup_report,
            "release_requested": self.release_requested,
        }
        return snap

    def wait_done(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)

    # ================================================================== control loop
    def _control_loop(self) -> None:
        next_t = time.monotonic()
        try:
            while not self.stop.is_set():
                self._tick()
                if self.phase == "RELEASED":
                    break
                next_t += self.dt
                late = time.monotonic() - next_t
                self.loop_late_ms.append(max(0.0, late * 1000))
                if late > 0.05:
                    next_t = time.monotonic()
                else:
                    time.sleep(max(0.0, next_t - time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - never die silently: the workers would FAULT-hold
            self.event("error", f"控制循环崩溃: {exc}")
            traceback.print_exc()
            self.fault_reason = f"控制循环崩溃: {exc}"
            self.phase = "FAULT"
        finally:
            self.done.set()

    def _tick(self) -> None:
        now = time.monotonic()
        # 1. measure ---------------------------------------------------------------------------
        try:
            state = self.arms.read(allow_fault=self.phase == "FAULT")
            self.comm_fail = 0
        except ArmFault as exc:
            self._enter_fault(f"worker 故障: {exc}")
            state = None
        except ArmCommError as exc:
            self.comm_fail += 1
            if self.comm_fail == 1:
                self.event("warn", f"机械臂通信异常: {exc}")
            if self.comm_fail >= 10 and self.phase != "FAULT":
                self._enter_fault(f"机械臂通信中断: {exc}")
            state = None
        if state is not None:
            self.state = state
            self.history.add(time.monotonic(), state)
        # 2. operator / thread commands ---------------------------------------------------------
        while True:
            try:
                name, arg = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._handle(name, arg, now)
        if self.phase in ("FAULT", "RELEASED") or self.state is None:
            return
        if self.phase == "RELEASING":
            self._do_release()
            return
        # 3. target ------------------------------------------------------------------------------
        target, leash = self._target(now)
        if self.phase in ("FAULT", "RELEASING", "RELEASED"):
            return
        safe = self.safety(target, self.state, self.dt, leash=leash, clamp_range=self.phase == "RUNNING")
        # 4. send --------------------------------------------------------------------------------
        try:
            self.arms.submit(safe)
            self.submit_fail = 0
        except (ArmCommError, ArmFault) as exc:
            self.submit_fail += 1
            if self.submit_fail >= 5:
                self._enter_fault(f"指令发送失败: {exc}")
            return
        self.cmd = safe
        if self.recorder is not None:
            self.recorder.control(now, self.state, self.policy_target, safe)

    def _target(self, now: float) -> tuple[np.ndarray, bool]:
        ph = self.phase
        if ph in ("PREPARING", "RETURNING"):
            return self._route_target(now), False
        if ph == "RUNNING":
            if now - self.ep_start > self.o.max_episode_s:
                self._finish(f"达到时长上限 {self.o.max_episode_s:.0f}s")
                return self._route_target(now), False
            tgt, info = self.scheduler.target_at(now)
            self.sched_info = info
            if info.get("starved_s", 0.0) > self.o.policy["starvation_timeout_s"]:
                self._estop(f"动作计划耗尽 {info['starved_s']:.2f}s（推理跟不上）")
                return self.hold, False
            self.policy_target = tgt
            if tgt is None or self.o.mode == "shadow":
                return self.ep_hold, False
            return tgt, True
        return self.hold, False

    def _route_target(self, now: float) -> np.ndarray:
        r = self.route
        tgt = r.at(now)
        err = np.abs(self.state - tgt)[list(JOINT_IDX)].max()
        if err > self.o.motion["max_tracking_error"]:
            self._estop(f"轨迹跟随误差 {err:.2f} rad 超限（碰撞或卡住？）")
            return self.hold
        if not r.done(now):
            return tgt
        # settle at the goal
        goal = r.goal
        if self.settle_since is None:
            self.settle_since, self.settle_ok = now, 0
        d = np.abs(self.state - goal)
        ok = (d[list(JOINT_IDX)].max() <= self.o.motion["settle_tolerance"]
              and d[list(GRIPPER_IDX)].max() <= self.o.motion["gripper_tolerance"])
        self.settle_ok = self.settle_ok + 1 if ok else 0
        timed_out = now - self.settle_since > self.o.motion["settle_timeout_s"]
        if self.settle_ok >= 3 or timed_out:
            if timed_out and not ok:
                self.event("warn", f"{self.route_label}: 未完全到位（最大偏差 关节 {d[list(JOINT_IDX)].max():.3f} rad, "
                                   f"夹爪 {d[list(GRIPPER_IDX)].max():.2f}），继续")
            self._arrive(goal)
        return goal

    def _arrive(self, goal: np.ndarray) -> None:
        self.hold = goal.copy()
        self.route = None
        nxt = self.route_next
        if nxt == "ZERO" and self.release_requested:
            nxt = "RELEASING"
        self.phase = nxt
        self.event("ok", f"{self.route_label} 完成")

    # ================================================================== transitions
    def _handle(self, name: str, arg, now: float) -> None:
        ph = self.phase
        if name == "estop":
            if ph in MOVING or ph == "PAUSED":
                self._estop(arg or "操作者急停")
            elif ph not in ("FAULT", "RELEASING", "RELEASED"):
                self.event("info", "当前已静止，无需急停")
        elif name == "next":
            if ph in ("IDLE", "ZERO"):
                self._go(self.start_pose, "READY", "移动到初始姿态", now)
            elif ph == "READY":
                self._start_episode(now)
            elif ph == "PAUSED":
                self._resume(now)
            elif ph in ("ESTOP", "FAULT"):
                self.event("info", "急停/故障状态下请按 z 回零位 或 i 回初始姿态")
        elif name == "pause":
            if ph == "RUNNING":
                self.phase, self.hold = "PAUSED", self.cmd.copy()
                self.event("info", "已暂停，原地保持")
            elif ph == "PAUSED":
                self._resume(now)
        elif name == "finish":
            if ph in ("RUNNING", "PAUSED"):
                self._finish("操作者结束")
        elif name == "zero":
            if ph == "ZERO":
                self.event("info", "已在零位")
            elif ph in ("IDLE", "READY", "PAUSED", "ESTOP", "FAULT"):
                if ph in ("PAUSED",):
                    self._finish("操作者回零")
                else:
                    emergency = ph in ("ESTOP", "FAULT")
                    self._go_zero(now, emergency=emergency)
            elif ph in MOVING:
                self.event("info", "运动中：先按空格急停，再按 z")
        elif name == "init":
            if ph in ("IDLE", "ZERO", "ESTOP", "FAULT", "PAUSED"):
                if ph == "PAUSED":
                    self._end_episode("操作者中止（回初始姿态）")
                self._go(self.start_pose, "READY", "移动到初始姿态", now)
            elif ph == "READY":
                self.event("info", "已在初始姿态")
        elif name == "release":
            self.release_requested = True
            if ph == "IDLE":
                self.phase = "RELEASING"
            elif ph == "ZERO":
                self.phase = "RELEASING"
            elif ph in ("RUNNING", "PAUSED"):
                self._finish("操作者释放资源")
            elif ph in ("READY", "ESTOP", "FAULT"):
                self._go_zero(now, emergency=ph != "READY")
            elif ph == "PREPARING":
                self._go_zero(now, emergency=True)
            self.event("info", "将回到零位后释放电机并退出")
        elif name == "force_release":
            self.event("warn", "强制释放：电机立即断电")
            self.release_requested = True
            self._end_episode("强制释放")
            self.phase = "RELEASING"
        elif name == "policy_error":
            self._estop(arg)

    def _go(self, goal, next_phase: str, label: str, now: float, via=None) -> None:
        if self.phase == "FAULT" and not self._recover():
            return
        start = self.cmd if self.cmd is not None else self.state
        waypoints = ([via] if via is not None else []) + [goal]
        self.route = Route(start, waypoints, self.limits, now)
        self.route_next, self.route_label = next_phase, label
        self.settle_since = None
        self.phase = "PREPARING" if next_phase == "READY" else "RETURNING"
        self.event("info", f"{label}（约 {self.route.duration:.1f}s）")

    def _go_zero(self, now: float, emergency: bool) -> None:
        route = self.o.poses["emergency_route" if emergency else "return_route"]
        via = self.start_pose if route == "via_start" else None
        self._go(self.zero, "ZERO", "紧急回零" if emergency else "回零位", now, via=via)

    def _recover(self) -> bool:
        try:
            self.arms.reset_fault()
            state = self.arms.read()
            self.arms.begin(state)
        except Exception as exc:  # noqa: BLE001
            self.event("error", f"复位失败: {exc}（可再试；或 Ctrl+C 退出后 bash start.sh --rescue）")
            return False
        self.state, self.cmd, self.hold = state, state.copy(), state.copy()
        self.safety.reset(state)
        self.fault_reason = None
        self.phase = "ESTOP"
        self.event("ok", "worker 已复位，重新接管")
        return True

    def _enter_fault(self, reason: str) -> None:
        if self.phase in ("FAULT", "RELEASED"):
            return
        self._end_episode(f"故障: {reason}")
        self.fault_reason = reason
        self.phase = "FAULT"
        self.route = None
        self.event("error", reason)

    def _estop(self, reason: str) -> None:
        if self.phase in ("FAULT", "RELEASING", "RELEASED"):
            return
        self._end_episode(f"急停: {reason}")
        hold = self.cmd.copy() if self.cmd is not None else self.state.copy()
        if self.state is not None:
            hold[list(JOINT_IDX)] = self.state[list(JOINT_IDX)]  # stop where the joints ARE
        self.hold = hold
        self.route = None
        self.phase = "ESTOP"
        self.event("error", f"急停：{reason}")

    def _start_episode(self, now: float) -> None:
        self.episode_index += 1
        self.episode_id += 1
        self.ep_start = now
        self.ep_hold = self.cmd.copy()
        self.hold = self.cmd.copy()
        self.scheduler.reset(self.ep_hold)
        self.safety.reset(self.cmd)
        self.safety.counts = {k: 0 for k in self.safety.counts}
        self.policy_fail = 0
        self.policy_target = None
        if self.o.recording["enabled"]:
            self.recorder = EpisodeRecorder(self.o.session_dir / "episodes", self.episode_index,
                                            {**self.o.meta, "episode": self.episode_index},
                                            self.o.recording["save_images"])
        self.phase = "RUNNING"
        shadow = "（影子模式：只推理不动作）" if self.o.mode == "shadow" else ""
        self.event("ok", f"第 {self.episode_index} 次执行开始{shadow}")

    def _resume(self, now: float) -> None:
        self.scheduler.reset(self.cmd)
        self.phase = "RUNNING"
        self.event("info", "继续执行（重新规划）")

    def _end_episode(self, reason: str) -> None:
        if self.phase not in ("RUNNING", "PAUSED"):
            return
        self.episode_id += 1
        dur = time.monotonic() - self.ep_start
        self.results.append({"episode": self.episode_index, "reason": reason, "duration": dur})
        if self.recorder is not None:
            rec, self.recorder = self.recorder, None
            try:
                path = rec.close(reason)
                self.event("info", f"已记录 {path.name}")
            except Exception as exc:  # noqa: BLE001
                self.event("warn", f"记录保存失败: {exc}")
        self.policy_target = None
        self.event("info", f"第 {self.episode_index} 次执行结束（{reason}，{dur:.1f}s）")

    def _finish(self, reason: str) -> None:
        self._end_episode(reason)
        self.phase = "PAUSED"  # transient; _go_zero switches to RETURNING
        self._go_zero(time.monotonic(), emergency=False)

    def _do_release(self) -> None:
        self.event("info", "释放电机（零力矩）…")
        try:
            self.arms.release()
            self.event("ok", "电机已释放")
        except Exception as exc:  # noqa: BLE001
            self.event("error", f"释放失败: {exc}")
        self.phase = "RELEASED"
        self.stop.set()

    # ================================================================== policy thread
    def _policy_loop(self) -> None:
        cams = [im.camera for im in self.spec.images]
        while not self.stop.is_set():
            ph = self.phase
            if self.warmup_pending and ph in ("IDLE", "READY", "ZERO"):
                self._live_warmup(cams)
                continue
            if ph != "RUNNING":
                time.sleep(0.005)
                continue
            now = time.monotonic()
            lead = max(self.lat) if self.lat else self.o.policy["latency_init_s"]
            if not self.scheduler.wants_inference(now, lead):
                time.sleep(0.002)
                continue
            ep = self.episode_id
            self.scheduler.mark_request(now)
            try:
                plan_inputs = self._observe(cams)
                raw, reply = self.client.predict(plan_inputs["state_vec"], plan_inputs["images"])
                t_done = time.monotonic()
                points = decode_actions(self.spec, raw, plan_inputs["robot"], self.ep_hold)
                if ep != self.episode_id or self.phase != "RUNNING":
                    self.scheduler.request_failed()
                    continue
                self.scheduler.add(Plan(plan_inputs["t_obs"], self.spec.fps, self.spec.action_offset_frames,
                                        points, t_done))
                self.lat.append(t_done - now)
                self.infer_ms.append(float(reply.get("infer_ms", 0.0)))
                self.n_infer += 1
                self.policy_fail = 0
                self.velocity_padded = plan_inputs["velocity_padded"]
                rec = self.recorder
                if rec is not None:
                    rec.inference(now, plan_inputs["t_obs"], t_done, plan_inputs["state_vec"], raw,
                                  points[:, list(self.spec.controlled)], reply.get("infer_ms", 0.0),
                                  plan_inputs["velocity_padded"], plan_inputs["images"])
            except Exception as exc:  # noqa: BLE001
                self.scheduler.request_failed()
                self.policy_fail += 1
                self.last_policy_error = f"{type(exc).__name__}: {exc}"
                self.event("warn", f"推理失败({self.policy_fail}): {self.last_policy_error}")
                if self.policy_fail >= self.o.policy["max_consecutive_failures"] and ep == self.episode_id:
                    self.command("policy_error", f"推理连续失败 {self.policy_fail} 次")
                time.sleep(0.02)

    def _observe(self, cams) -> dict:
        latest = self.history.latest()
        if latest is None:
            raise RuntimeError("还没有机械臂状态")
        t_obs = latest[0]
        if time.monotonic() - t_obs > 0.1:
            raise RuntimeError("机械臂状态过期")
        frames, _ = self.cams.grab(cams, self.o.max_frame_age_s)
        state_vec, robot, info = build_state(self.spec, self.history, t_obs,
                                             self.ep_start if self.phase in ("RUNNING", "PAUSED") else -1e9)
        jpeg = self.o.policy["jpeg_quality"] or None
        images = {im.key: letterbox_rgb(frames[im.camera], im.height, im.width, jpeg) for im in self.spec.images}
        return {"t_obs": t_obs, "state_vec": state_vec, "robot": robot, "images": images,
                "velocity_padded": info["velocity_padded"]}

    def _live_warmup(self, cams) -> None:
        """A few real camera+state inferences (discarded): first-call latency happens here,
        not in the first second of the task."""
        n = self.warmup_pending
        times = []
        for _ in range(n):
            if self.stop.is_set():
                return
            try:
                t0 = time.monotonic()
                obs = self._observe(cams)
                self.client.predict(obs["state_vec"], obs["images"])
                times.append((time.monotonic() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                self.warmup_attempts += 1
                if self.warmup_attempts >= 5:
                    self.warmup_pending = 0
                    self.warmup_report = f"失败: {exc}"
                    self.event("warn", f"实时预热失败（已重试 5 次）: {exc}")
                time.sleep(0.5)
                return
        self.warmup_pending = 0
        self.lat.extend(t / 1000 for t in times[1:])
        self.warmup_report = f"首次 {times[0]:.0f}ms，之后中位 {np.median(times[1:] or times):.0f}ms"
        self.event("ok", f"实时预热完成（相机+状态+模型全链路）: {self.warmup_report}")
