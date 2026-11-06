"""Both follower arms seen as one 14-value device.

FollowerArms  starts (or attaches to) one worker process per arm and talks portal RPC to it.
SimArms       in-process stand-in for sim mode (first-order lag, no hardware, no portal).

The console's 100 Hz control loop is the ONLY caller of submit(); it is also the heartbeat that
keeps the workers' configured command lease alive.

Worker control session:  begin(hold) -> [ALIGNING frame = hold, wait until applied] -> ACTIVE
                         -> submit() every tick -> end() (= worker holds the pose itself)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from pi_deploy import PROJECT_ROOT

SIDES = ("left", "right")
WORKER_MODULE = "pi_deploy.hw.follower.worker"


class ArmFault(RuntimeError):
    """A worker reported FAULT (it is holding its pose)."""


class ArmCommError(RuntimeError):
    """RPC failure / stale feedback."""


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def pid_alive(pid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def load_ports(hw_config: Path) -> dict[str, int]:
    from pi_deploy.hw.follower.config import load_teleoperation_config

    cfg = load_teleoperation_config(hw_config)
    return {s: cfg.arms[s].follower.rpc_port for s in SIDES}


def load_can_channels(hw_config: Path) -> dict[str, str]:
    from pi_deploy.hw.follower.config import load_teleoperation_config

    cfg = load_teleoperation_config(hw_config)
    return {s: cfg.arms[s].follower.can_channel for s in SIDES}


def leftover_workers(pid_dir: Path) -> list[dict]:
    """Workers of an earlier run of THIS project that are still alive (pid files)."""
    out = []
    for f in sorted(Path(pid_dir).glob("*.json")):
        try:
            info = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if info.get('project_root') != str(PROJECT_ROOT) or info.get('side') not in SIDES:
            continue
        if pid_alive(int(info.get("pid", -1))):
            out.append(info)
        else:
            f.unlink(missing_ok=True)
    return out


class FollowerArms:
    def __init__(self, hw_config: Path, *, log_dir: Path | None, pid_dir: Path, fake: bool = False,
                 attach: bool = False, rpc_timeout: float = 0.1, start_timeout: float = 40.0,
                 feedback_max_age_s: float = 0.1, sides: tuple[str, ...] = SIDES):
        import portal

        self.hw_config = Path(hw_config).resolve()
        if not sides or len(set(sides)) != len(sides) or any(x not in SIDES for x in sides):
            raise ValueError('sides must be a non-empty unique subset of left/right')
        self.sides = tuple(sides)
        self.ports = {s: p for s, p in load_ports(self.hw_config).items() if s in self.sides}
        self.identities = {}
        self.pid_handles = {}
        self.timeout = rpc_timeout
        self.feedback_max_age_s = feedback_max_age_s
        self.log_dir, self.pid_dir, self.fake = log_dir, Path(pid_dir), fake
        self.procs: dict[str, subprocess.Popen] = {}
        self.clients: dict = {}
        self.session: str | None = None
        self.seq = 0
        self._lock = threading.RLock()
        self.last_sample_age_ms = 0.0
        self.fault: str | None = None
        try:
            if not attach:
                self._spawn(start_timeout)
            else:
                missing = [s for s, p in self.ports.items() if not port_open(p)]
                if missing:
                    raise RuntimeError(f"没有找到运行中的 worker: {missing}")
            self.clients = {s: portal.Client(f"127.0.0.1:{self.ports[s]}", name=f"arm_{s}") for s in self.sides}
            for s in self.sides:
                identity = self._call(s, 'identity', timeout=3.0)
                if (identity.get('project_root') != str(PROJECT_ROOT) or identity.get('side') != s
                        or identity.get('config') != str(self.hw_config)):
                    raise RuntimeError(f'{s} worker 归属不匹配，禁止接管')
                self.identities[s] = identity
                if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                    self.pid_handles[s] = os.pidfd_open(identity['pid'])
                    if self._call(s, 'identity', timeout=1.0) != identity:
                        raise RuntimeError('worker 身份在接管时变化')
                elif attach:
                    raise RuntimeError('安全救援要求Linux pidfd支持；当前系统不支持，不发送进程信号')
                if attach:
                    info = json.loads((self.pid_dir / f'{s}.json').read_text(encoding='utf-8'))
                    if identity.get('pid') != info.get('pid'):
                        raise RuntimeError(f'{s} worker PID 已变化，禁止接管')
        except BaseException:
            # A healthy worker may already hold an arm at an arbitrary pose. Do not drop it
            # just because the other worker / startup handshake failed. start.sh offers rescue.
            for client in self.clients.values():
                try:
                    client.close(timeout=0.2)
                except Exception:
                    pass
            raise

    # ------------------------------------------------------------------ process
    def _spawn(self, start_timeout: float) -> None:
        busy = [f"{s}:{p}" for s, p in self.ports.items() if port_open(p)]
        if busy:
            left = leftover_workers(self.pid_dir)
            if left:
                raise RuntimeError("检测到上一次运行遗留的机械臂 worker（手臂可能仍通电保持）。"
                                   "请先执行救援：bash start.sh --rescue")
            raise RuntimeError(f"机械臂 RPC 端口被占用 ({', '.join(busy)})：可能是 SI 采集 WebUI 或其他程序"
                               "正在控制机械臂，请先关闭它")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        for side in self.sides:
            out = open(self.log_dir / f"worker_{side}.log", "w") if self.log_dir else subprocess.DEVNULL  # noqa: SIM115
            cmd = [sys.executable, "-m", WORKER_MODULE, "--config", str(self.hw_config), "--side", side,
                   "--pid-dir", str(self.pid_dir)] + (["--fake"] if self.fake else [])
            kwargs = {"start_new_session": True} if os.name == "posix" else {}
            self.procs[side] = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=out,
                                                stderr=subprocess.STDOUT, **kwargs)
        deadline = time.monotonic() + start_timeout
        pending = set(self.sides)
        while pending:
            for s, p in self.procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{s} 臂 worker 启动失败(退出码 {p.returncode})，见 {self._log_hint(s)}")
            for s in list(pending):
                if port_open(self.ports[s]):
                    pending.discard(s)
            if pending and time.monotonic() > deadline:
                raise TimeoutError(f"worker 启动超时 {sorted(pending)}（CAN 不通/电机未上电？见 {self._log_hint('right')}）")
            time.sleep(0.2)

    def _log_hint(self, side: str) -> str:
        return str(self.log_dir / f"worker_{side}.log") if self.log_dir else "worker 输出"

    def workers_alive(self) -> bool:
        if self.procs:
            return all(p.poll() is None for p in self.procs.values())
        return all(port_open(p) for p in self.ports.values())

    # ------------------------------------------------------------------ rpc
    def _call(self, side: str, method: str, *args, timeout: float | None = None):
        try:
            r = getattr(self.clients[side], method)(*args).result(timeout=timeout or self.timeout)
        except TimeoutError as exc:
            raise ArmCommError(f"{side} 臂 RPC {method} 超时") from exc
        if not isinstance(r, dict) or not r.get("ok"):
            raise ArmCommError(f"{side} 臂 {method} 失败: {r.get('error') if isinstance(r, dict) else r}")
        return r["value"]

    def _both(self, method: str, args_per_side: dict | None = None, timeout: float | None = None) -> dict:
        futures = {s: getattr(self.clients[s], method)(*(args_per_side or {}).get(s, ())) for s in self.sides}
        out = {}
        for s, f in futures.items():
            try:
                r = f.result(timeout=timeout or self.timeout)
            except TimeoutError as exc:
                raise ArmCommError(f"{s} 臂 RPC {method} 超时") from exc
            if not isinstance(r, dict) or not r.get("ok"):
                raise ArmCommError(f"{s} 臂 {method} 失败: {r.get('error') if isinstance(r, dict) else r}")
            out[s] = r["value"]
        return out

    def read(self, allow_fault: bool = False) -> np.ndarray:
        """robot14 state.  Raises ArmFault if a worker is in FAULT (unless allow_fault),
        ArmCommError on RPC failure / stale data."""
        with self._lock:
            samples = self._both("get_sample")
        now = time.monotonic_ns()
        faults = [f"{s}: {samples[s].get('fault_reason')}" for s in self.sides if samples[s].get("phase") == "FAULT"]
        self.fault = "; ".join(faults) or None
        if faults and not allow_fault:
            raise ArmFault(self.fault)
        age = max((now - int(samples[s]["sampled_at_ns"])) / 1e9 for s in self.sides)
        self.last_sample_age_ms = age * 1000
        if age > self.feedback_max_age_s:
            raise ArmCommError(f"机械臂反馈过期 {age * 1000:.0f} ms")
        state = np.zeros(14)
        for side in self.sides:
            i = SIDES.index(side) * 7
            state[i:i+7] = np.asarray(samples[side]['state'], np.float64).reshape(7)
        if not np.isfinite(state).all():
            raise ArmCommError('机械臂反馈不是有限值')
        return state

    def phases(self) -> dict[str, str]:
        with self._lock:
            return {s: v.get("phase") for s, v in self._both("get_status", timeout=0.5).items()}

    def begin(self, hold14) -> None:
        """New control session holding hold14 (the current pose)."""
        hold14 = np.asarray(hold14, np.float64)
        with self._lock:
            self.end()
            self.session = uuid.uuid4().hex
            self.seq = 0
            self._both("begin_control_session", {s: (self.session,) for s in self.sides}, timeout=1.0)
            self._send(hold14, "ALIGNING")
            deadline = time.monotonic() + 1.0
            while True:
                st = self._both("get_status", timeout=0.5)
                if all(int(st[s]["last_applied_sequence"]) >= 0 for s in self.sides):
                    break
                if time.monotonic() > deadline:
                    raise ArmCommError("worker 未确认保持帧")
                self._send(hold14, "ALIGNING")  # keep the lease alive while waiting
                time.sleep(0.01)
            self._both("set_phase", {s: (self.session, "ACTIVE") for s in self.sides}, timeout=1.0)

    def _send(self, target14, phase: str) -> None:
        if self.session is None:
            raise ArmCommError("没有控制会话")
        self.seq += 1
        issued = time.monotonic_ns()
        args = {s: ({"session_id": self.session, "sequence": self.seq, "issued_at_ns": issued,
                     "phase": phase, "joint_position": np.asarray(target14[i * 7:(i + 1) * 7], np.float64)},)
                for s in self.sides for i in [SIDES.index(s)]}
        self._both("submit_control_frame", args)

    def submit(self, target14) -> None:
        with self._lock:
            self._send(np.asarray(target14, np.float64), "ACTIVE")

    def end(self) -> None:
        """Leave control; the workers hold the current pose themselves (motors stay on)."""
        with self._lock:
            if self.session is None:
                return
            for s in self.sides:
                try:
                    self._call(s, "end_control_session", self.session, timeout=1.0)
                except Exception:  # noqa: BLE001 - faulted worker has no session
                    pass
            self.session = None

    def reset_fault(self) -> None:
        with self._lock:
            self.session = None
            self._both("reset_fault", timeout=1.0)

    def release(self) -> None:
        """Stop the workers => motors ZERO TORQUE.  Only call at a pose where that is safe."""
        try:
            self.end()
        except Exception:  # noqa: BLE001
            pass
        pids = {}
        for side, identity in self.identities.items():
            current = self._call(side, 'identity', timeout=1.0)
            if current != identity:
                raise RuntimeError(f'{side} worker 身份已变化；禁止释放')
            pids[side] = identity['pid']
        # Retain verified identities; never rescan pid_dir during release.
        for client in self.clients.values():
            t = threading.Thread(target=client.close, kwargs={'timeout': 1.0}, daemon=True)
            t.start()
            t.join(timeout=2.0)
        self.clients = {}
        for s, pid in pids.items():
            try:
                if s in self.procs:
                    if self.procs[s].poll() is None:
                        self.procs[s].terminate()
                else:
                    signal.pidfd_send_signal(self.pid_handles[s], signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        deadline = time.monotonic() + 10
        for s, pid in pids.items():
            while time.monotonic() < deadline:
                alive = self.procs[s].poll() is None if s in self.procs else pid_alive(pid)
                if not alive:
                    break
                time.sleep(0.1)
            else:
                try:
                    (self.procs[s].kill() if s in self.procs else signal.pidfd_send_signal(self.pid_handles[s], signal.SIGKILL))
                except (OSError, ProcessLookupError):
                    pass
        for fd in self.pid_handles.values():
            os.close(fd)
        self.pid_handles = {}


class SimArms:
    """Sim mode: 14-value first-order lag towards the target (~130 ms like the real follower)."""

    fake = True

    def __init__(self, initial14, tau_s: float = 0.13):
        self.state = np.asarray(initial14, np.float64).copy()
        self.target = self.state.copy()
        self.tau = tau_s
        self.t = time.monotonic()
        self.session = None
        self.last_sample_age_ms = 0.0
        self._lock = threading.Lock()

    fault = None

    def read(self, allow_fault: bool = False) -> np.ndarray:
        with self._lock:
            now = time.monotonic()
            self.state += (self.target - self.state) * (1 - np.exp(-(now - self.t) / self.tau))
            self.t = now
            return self.state.copy()

    def phases(self) -> dict[str, str]:
        return {s: ("ACTIVE" if self.session else "SAFE_HOLD") for s in SIDES}

    def begin(self, hold14) -> None:
        self.read()
        self.target = np.asarray(hold14, np.float64).copy()
        self.session = "sim"

    def submit(self, target14) -> None:
        self.read()
        self.target = np.asarray(target14, np.float64).copy()

    def end(self) -> None:
        self.session = None

    def reset_fault(self) -> None:
        self.session = None

    def workers_alive(self) -> bool:
        return True

    def release(self) -> None:
        self.session = None
