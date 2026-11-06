"""Offline self-test (no hardware, no torch needed):  python -m act_deploy.tools.selftest

Checks the pure parts that decide what the robot does: checkpoint contract parsing, state /
velocity construction, image preprocessing, action decoding, plan scheduling smoothness,
trajectory limits, safety filter, the loopback protocol (mock model), and the worker runtime
with a fake arm (if portal is installed).
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np

from act_deploy import JOINT_IDX, PROJECT_ROOT

RESULTS: list[tuple[str, bool, str]] = []


def case(fn):
    try:
        detail = fn() or ""
        RESULTS.append((fn.__name__, True, detail))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()
    return fn


def _ckpts():
    return sorted((PROJECT_ROOT / "checkpoints").glob("*/*/pretrained_model"))


def _velocity_ckpt():
    """A checkpoint whose state includes the 100 ms velocity (v3); None if there is none."""
    from act_deploy.model.spec import OBS_CONTRACT

    return next((p for p in _ckpts() if (p / OBS_CONTRACT).is_file()), None)


@case
def spec_from_checkpoints():
    from act_deploy.model.spec import resolve_spec

    from act_deploy.model.spec import ACTION_CONTRACT, OBS_CONTRACT, normalizer_stats

    paths = _ckpts()
    assert paths, "checkpoints/ 下没有权重"
    kinds: dict[str, list[str]] = {}
    for p in paths:
        s = resolve_spec(p)  # raises on inconsistent files
        name = f"{p.parent.parent.name}/{p.parent.name}"
        # what the checkpoint files say must be exactly what the spec uses
        stats = normalizer_stats(p)
        assert stats["observation.state.mean"].size == s.state_dim, name
        assert stats["action.mean"].size == s.action_dim, name
        has_vel = (p / OBS_CONTRACT).is_file()
        assert bool(s.velocity_segments) == has_vel, name
        if has_vel:
            v = s.velocity_segments[0]
            assert s.state_dim == 14 + v.size and v.lag_s > 0, name
        assert bool(s.relative_dims) == (p / ACTION_CONTRACT).is_file(), name
        assert all(s.action_targets[d] not in (6, 13) for d in s.relative_dims), name
        assert s.state_min is not None, f"{name}: 没读到示范范围（安全范围裁剪会失效）"
        vel = f"+速度{s.velocity_segments[0].size}" if has_vel else ""
        kind = (f"state{s.state_dim}(14{vel}) -> {s.horizon}x{s.action_dim}, "
                f"{'增量' if s.relative_dims else '绝对'}关节")
        kinds.setdefault(kind, []).append(name)
    return "; ".join(f"{len(v)}个 {k}" for k, v in kinds.items())


@case
def state_and_velocity():
    from act_deploy.control.history import StateHistory
    from act_deploy.model.features import build_state
    from act_deploy.model.spec import resolve_spec

    if _velocity_ckpt() is None:
        return "跳过（没有带速度输入的 v3 checkpoint）"
    s = resolve_spec(_velocity_ckpt())
    h = StateHistory()
    vel = np.zeros(14)
    vel[7:14] = [0.5, -0.2, 0.1, 0, 0, 0.3, -1.0]
    t0 = 100.0
    for k in range(101):  # 1 s at 100 Hz, constant velocity
        h.add(t0 + k * 0.01, np.full(14, 0.5) + vel * k * 0.01)
    t = t0 + 1.0
    st, robot, info = build_state(s, h, t, episode_start=t0)
    assert np.allclose(st[14:], vel[7:14], atol=1e-5), st[14:]
    assert not info["velocity_padded"]
    st2, _, info2 = build_state(s, h, t, episode_start=t - 0.05)  # episode younger than 100 ms
    assert info2["velocity_padded"] and np.all(st2[14:] == 0)
    assert np.allclose(st[:14], robot)
    return "速度 = (s(t)-s(t-0.1))/0.1 正确；开局 <100ms 补零"


@case
def image_preprocessing_matches_training():
    import cv2

    from act_deploy.model.features import letterbox_rgb

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = letterbox_rgb(rgb, 256, 256, 90)
    # training: recorder encodes RGB->BGR JPEG q90; converter decodes BGR, prepare_rgb()
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    resized = cv2.resize(bgr, (256, 192), interpolation=cv2.INTER_AREA)
    ref = np.zeros((256, 256, 3), np.uint8)
    ref[32:224] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    assert out.shape == (256, 256, 3) and np.array_equal(out, ref)
    assert out[:32].max() == 0 and out[224:].max() == 0
    return "640x480 -> JPEG q90 -> 256x192 + 上下各 32 行黑边，与训练逐像素一致"


@case
def action_decoding():
    from act_deploy.model.features import decode_actions
    from act_deploy.model.spec import resolve_spec

    s = next((resolve_spec(p) for p in _ckpts() if resolve_spec(p).relative_dims), None)
    if s is None:
        return "跳过（没有增量动作 checkpoint）"
    raw = np.zeros((32, 7))
    raw[:, :6] = 0.1
    raw[:, 6] = 1.3
    anchor = np.arange(14) / 10
    hold = np.full(14, -1.0)
    hold[[6, 13]] = 0.5
    out = decode_actions(s, raw, anchor, hold)
    assert np.allclose(out[:, 7:13], anchor[7:13] + 0.1)
    assert np.allclose(out[:, 13], 1.0)  # gripper absolute, clipped
    assert np.allclose(out[:, :7], hold[:7])  # left arm untouched
    return "右关节 = 观测 + 增量；右夹爪绝对值(裁剪0..1)；左臂保持"


@case
def scheduler_smooth():
    from act_deploy.control.scheduler import ActionScheduler, Plan

    # The fake "model" plans a 1 rad/s sine (=0.01 rad per 10 ms tick) and every plan gets its
    # own random offset (consecutive predictions disagree a bit).  A correct scheduler can then
    # step at most  0.01 + |offset jump| / (blend ticks)  per tick in steady state.
    out = {}
    rng = np.random.default_rng(0)
    blend_s, dt = 0.15, 0.01
    for mode in ("chunk", "ensemble"):
        sch = ActionScheduler(mode=mode, fps=20, horizon=32, offset=1, exec_steps=10, blend_s=blend_s,
                              rate_hz=15, ensemble_m=0.01)
        hold = np.zeros(14)
        sch.reset(hold)
        t = 0.0
        traj = lambda tt: np.sin(tt) * np.ones(14)  # noqa: E731 - what the "model" plans
        prev, n_plans = None, 0
        start_step = steady_step = 0.0
        offsets = []
        pending = None
        for _ in range(600):
            if pending and t >= pending[0]:
                sch.add(pending[1])
                pending = None
                n_plans += 1
            if pending is None and sch.wants_inference(t, 0.03):
                sch.mark_request(t)
                t_obs = t
                offsets.append(0.02 * rng.standard_normal())
                pts = np.stack([traj(t_obs + (k + 1) * 0.05) + offsets[-1] for k in range(32)])
                pending = (t + 0.03, Plan(t_obs, 20, 1, pts, t + 0.03))
            v, info = sch.target_at(t)
            assert v is not None and info["starved_s"] == 0.0
            if prev is not None:
                step = float(np.abs(v - prev).max())
                if t < 0.3:  # fade-in from the hold pose into the first plan
                    start_step = max(start_step, step)
                else:
                    steady_step = max(steady_step, step)
            prev = v
            t += dt
        jump = float(np.abs(np.diff(offsets)).max()) if len(offsets) > 1 else 0.0
        bound = 0.01 + jump / (blend_s / dt) + 0.002
        assert steady_step <= bound, f"{mode}: steady step {steady_step:.4f} > bound {bound:.4f}"
        # fade-in: hold -> first plan (value |offset0| + sin) over blend_s, plus plan motion
        start_bound = 0.01 + (abs(offsets[0]) + 0.3) / (blend_s / dt) + 0.005
        assert start_step <= start_bound, f"{mode}: start step {start_step:.4f} > {start_bound:.4f}"
        out[mode] = f"{n_plans}计划, 稳态最大步长 {steady_step:.4f}(上限{bound:.4f}), 起步 {start_step:.4f}"
    return "; ".join(f"{k}: {v}" for k, v in out.items())


@case
def trajectory_limits():
    from act_deploy.control.trajectory import Route

    limits = dict(max_joint_velocity=0.4, max_joint_acceleration=0.8, max_gripper_velocity=0.5, min_duration_s=2.0)
    a = np.zeros(14)
    a[[6, 13]] = 0.99
    b = np.array([-1.14, 0.82, 1.79, -1.44, -0.17, 0.12, 0.99, 0.83, 1.6, 2.42, -1.53, 0.03, -0.02, 0.99])
    r = Route(a, [b, a], limits, 0.0)
    ts = np.arange(0, r.duration + 0.01, 0.01)
    x = np.stack([r.at(t) for t in ts])
    v = np.diff(x, axis=0) / 0.01
    acc = np.diff(v, axis=0) / 0.01
    vj = np.abs(v[:, list(JOINT_IDX)]).max()
    aj = np.abs(acc[:, list(JOINT_IDX)]).max()
    assert vj <= 0.4 * 1.01 and aj <= 0.8 * 1.05, (vj, aj)
    assert np.allclose(x[0], a) and np.allclose(x[-1], a)
    return f"零位->初始姿态->零位 {r.duration:.1f}s, 峰值速度 {vj:.3f} rad/s, 峰值加速度 {aj:.3f} rad/s²"


@case
def safety_filter():
    from act_deploy.control.safety import SafetyFilter

    f = SafetyFilter(max_joint_velocity=1.0, max_gripper_velocity=2.0, max_deviation=0.2)
    f.reset(np.zeros(14))
    y = f(np.full(14, 5.0), np.zeros(14), 0.01, leash=True)
    assert np.allclose(y[list(JOINT_IDX)], 0.01) and np.allclose(y[[6, 13]], 0.02)
    assert f.counts["leash"] == 1 and f.counts["rate"] == 1
    return "牵引/限速/夹爪裁剪生效"


@case
def ipc_with_mock_server():
    from act_deploy.ipc import PolicyClient
    from act_deploy.model.features import dummy_inputs
    from act_deploy.server.policy_server import Server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    args = SimpleNamespace(checkpoint=str(_ckpts()[0]), port=port, device="cpu", backend="mock",
                           warmup=2, keep_warm_s=0.0, exit_with_parent=False)
    srv = Server(args)
    th = threading.Thread(target=srv.serve, daemon=True)
    th.start()
    cl = PolicyClient("127.0.0.1", port, 1.0)
    deadline = time.monotonic() + 10
    while True:
        try:
            if cl.health()["status"] == "ready":
                break
        except OSError:
            pass
        assert time.monotonic() < deadline, "mock server not ready"
        time.sleep(0.05)
    state, images = dummy_inputs(srv.spec)
    rtts = []
    for _ in range(20):
        actions, reply = cl.predict(state, images)
        rtts.append(reply["roundtrip_ms"] - reply["infer_ms"])
    assert actions.shape == (32, 7)
    try:
        cl.predict(state[:5], images)
        raise AssertionError("bad state accepted")
    except RuntimeError:
        pass
    cl.shutdown()
    th.join(3)
    return f"往返协议开销中位 {np.median(rtts):.2f} ms；错误输入被拒绝"


@case
def worker_runtime_fake_arm():
    try:
        import portal  # noqa: F401
    except ImportError:
        return "跳过（本环境没有 portal；在部署机设备环境中会运行）"
    import tempfile
    from pathlib import Path

    from act_deploy.config import HARDWARE_YAML
    from act_deploy.hw.arms import FollowerArms, load_ports, port_open

    if any(port_open(p) for p in load_ports(HARDWARE_YAML).values()):
        return "跳过（worker 端口被占用）"
    with tempfile.TemporaryDirectory() as d:
        arms = FollowerArms(HARDWARE_YAML, log_dir=Path(d), pid_dir=Path(d) / "pids", fake=True)
        try:
            s = arms.read()
            arms.begin(s)
            goal = s.copy()
            goal[7] += 0.2
            for k in range(60):
                arms.submit(s + (goal - s) * min(1, k / 30))
                time.sleep(0.01)
            s2 = arms.read()
            assert abs(s2[7] - goal[7]) < 0.1, s2[7]
            time.sleep(0.5)  # no frames -> lease expires -> FAULT (hold)
            arms.read(allow_fault=True)
            assert arms.fault and "lease" in arms.fault, arms.fault
            arms.reset_fault()
            arms.begin(arms.read())
            arms.submit(arms.read())
        finally:
            arms.release()
    return "worker 进程 + RPC + 租约超时故障 + 复位 正常"


def main() -> int:
    from act_deploy.app.console import c

    print()
    for name, ok, detail in RESULTS:
        print(f" {c(' PASS', 'green') if ok else c(' FAIL', 'red')}  {name:<38} {detail}")
    bad = sum(not ok for _, ok, _ in RESULTS)
    print(c(f"\n {len(RESULTS) - bad}/{len(RESULTS)} 通过", "green" if not bad else "red"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
