"""Entry point of the deployment console (run by start.sh in the device environment).

    python -m act_deploy.app.main --model-python <py312>              interactive menus
    python -m act_deploy.app.main --task block_into_cup --checkpoint v3_long2_017000 \
        --mode sim --preset chunk_10 --headless --script "enter,wait:READY,enter,sleep:3,f,wait:ZERO,r"

Start-up order:
    menus -> session dir -> model server process (loads + warms up in the background)
          -> arm workers + cameras -> wait for the server READY -> engine -> dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from act_deploy import PROJECT_ROOT
from act_deploy.app import console as ui
from act_deploy.config import HARDWARE_YAML, ConfigError, load_deploy, load_tasks
from act_deploy.ipc import PolicyClient
from act_deploy.model.spec import resolve_spec
from act_deploy.registry import scan_checkpoints

LOG_ROOT = PROJECT_ROOT / "logs"
PID_DIR = LOG_ROOT / ".workers"


def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("找不到空闲端口")


class ServerProcess:
    def __init__(self, python: str, ckpt: Path, cfg: dict, log_path: Path, backend: str, device: str):
        self.port = free_port(cfg["port"])
        self.log_path = log_path
        cmd = [python, "-m", "act_deploy.server.policy_server", "--checkpoint", str(ckpt),
               "--port", str(self.port), "--device", device, "--backend", backend,
               "--warmup", str(cfg["warmup_iterations"]), "--keep-warm-s", str(cfg["keep_warm_s"]),
               "--exit-with-parent"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HUB_OFFLINE", "1")
        self.log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, **kwargs)
        self.client = PolicyClient("127.0.0.1", self.port, cfg["request_timeout_s"])

    def alive(self) -> bool:
        return self.proc.poll() is None

    def tail(self, n: int = 12) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def health(self) -> dict | None:
        try:
            return self.client.health(timeout_s=2.0)
        except Exception:  # noqa: BLE001 - not listening yet
            return None

    def stop(self) -> None:
        if self.alive():
            self.client.shutdown()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        self.client.close()
        self.log.close()


# ---------------------------------------------------------------------- menus
def pick(args, deploy, tasks):
    names = list(tasks)
    if args.task:
        if args.task not in tasks:
            raise SystemExit(f"未知任务 {args.task}，可选 {names}")
        task = tasks[args.task]
    else:
        opts = [(f"{t.label}  ({t.name})", "权重目录: " + ", ".join(str(p) for p in t.checkpoint_dirs), t.enabled)
                for t in tasks.values()]
        default = next((i for i, t in enumerate(tasks.values()) if t.enabled), 0)
        task = tasks[names[ui.choose("① 选择任务", opts, default)]]

    ckpts = scan_checkpoints(task)
    if not ckpts:
        raise SystemExit(f"任务 {task.name} 的权重目录里没有找到完整的 pretrained_model: "
                         + ", ".join(str(p) for p in task.checkpoint_dirs))
    if args.checkpoint:
        match = [c for c in ckpts if c.name == args.checkpoint or str(c.path) == args.checkpoint]
        if not match:
            raise SystemExit(f"没有权重 {args.checkpoint}，可选 {[c.name for c in ckpts]}")
        ckpt = match[0]
    else:
        ckpt = ckpts[ui.choose("② 选择权重", [(c.describe(), str(c.path), True) for c in ckpts], len(ckpts) - 1)]

    spec = resolve_spec(ckpt.path, task.model_overrides)
    print(ui.rule("模型输入/输出（从 checkpoint 契约解析）"))
    for line in spec.summary():
        print("   " + (ui.c(line, "yellow") if line.startswith("注意") else line))
    missing = [i.camera for i in spec.images if i.camera not in deploy["cameras"]["serials"]]
    if missing:
        raise SystemExit(f"模型需要相机 {missing}，但 configs/deploy.yaml cameras.serials 里没有")
    print()

    modes = [("real", "真机执行", "初始化 → 模型控制机械臂执行任务 → 回零"),
             ("shadow", "影子模式", "机械臂会做初始化/回零，但执行阶段只推理、显示模型想怎么动，手臂不动（首次上机推荐）"),
             ("sim", "模拟演练", "无硬件：模拟机械臂 + 合成相机画面 + 真实模型推理（检查链路与延迟）")]
    if args.mode:
        mode = args.mode
    else:
        mode = modes[ui.choose("③ 选择运行模式", [(f"{n}", d, True) for _, n, d in modes], 0)][0]

    presets = deploy["execution"]["presets"]
    keys = list(presets)
    if args.preset:
        if args.preset not in presets:
            raise SystemExit(f"未知执行方式 {args.preset}，可选 {keys}")
        preset_name = args.preset
    else:
        def detail(p):
            if p["mode"] == "chunk":
                return f"每个计划执行 {p['exec_steps']} 步({p['exec_steps'] / spec.fps:.2f}s)，新旧计划淡入 {p['blend_s']}s"
            return f"{p['rate_hz']}Hz 重新规划，加权 exp(-{p['ensemble_m']}·i)"
        default = keys.index(task.execution)
        preset_name = keys[ui.choose("④ 选择执行方式（动作平滑策略）",
                                     [(presets[k]["label"], detail(presets[k]), True) for k in keys], default)]
    return task, ckpt, spec, mode, preset_name


# ---------------------------------------------------------------------- startup helpers
def wait_server(server: ServerProcess, timeout: float, spec, say) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if not server.alive():
            raise RuntimeError(f"模型服务进程退出（退出码 {server.proc.returncode}）:\n{server.tail()}")
        h = server.health()
        status = h.get("status") if h else "starting"
        if status != last:
            say({"starting": "模型服务启动中…", "loading": "加载模型权重…",
                 "warming": "模型预热中（首次 CUDA 初始化较慢）…", "ready": "模型服务 READY"}.get(status, status))
            last = status
        if status == "ready":
            hs = h["spec"]
            mine = {"state_dim": spec.state_dim, "action_dim": spec.action_dim, "horizon": spec.horizon}
            if any(hs[k] != v for k, v in mine.items()):
                raise RuntimeError(f"服务端与本地对模型的理解不一致: {hs} vs {mine}")
            return h
        if status == "error":
            raise RuntimeError(f"模型服务加载失败: {h.get('error')}\n{server.tail()}")
        time.sleep(0.3)
    raise TimeoutError(f"模型服务 {timeout:.0f}s 内未就绪:\n{server.tail()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-python", default=os.environ.get("MODEL_PY", sys.executable))
    p.add_argument("--task")
    p.add_argument("--checkpoint")
    p.add_argument("--mode", choices=("real", "shadow", "sim"))
    p.add_argument("--preset")
    p.add_argument("--device", help="override policy_server.device (cuda/cpu)")
    p.add_argument("--backend", choices=("lerobot", "mock"), default="lerobot",
                   help="mock = fake model without torch (UI tests only)")
    p.add_argument("--yes", action="store_true", help="skip the final confirmation")
    p.add_argument("--sim-hardware", action="store_true",
                   help="dev: simulated arms + synthetic cameras in any mode (tests real/shadow logic offline)")
    p.add_argument("--headless", action="store_true", help="print events instead of the dashboard")
    p.add_argument("--script", default="", help="comma separated keys / wait:PHASE[:s] / sleep:s (tests)")
    args = p.parse_args()

    try:
        deploy = load_deploy()
        tasks = load_tasks(deploy)
    except ConfigError as exc:
        raise SystemExit(f"配置错误: {exc}") from exc

    ui.banner([ui.c("流程：选任务 → 选权重 → 选模式 → 选执行方式 → 启动", "grey"),
               ui.c(f"设备环境 {sys.executable}", "grey"), ui.c(f"模型环境 {args.model_python}", "grey")])
    task, ckpt, spec, mode, preset_name = pick(args, deploy, tasks)
    preset = deploy["execution"]["presets"][preset_name]
    device = args.device or deploy["policy_server"]["device"]

    print(ui.rule("确认"))
    print(f"   任务  {task.label}\n   权重  {ckpt.name}  ({ckpt.path})\n   模式  {ui.MODE_NAMES[mode]}"
          f"\n   执行  {preset['label']}\n   推理  {args.backend} on {device}"
          f"\n   初始姿态 {ui.fmt_pose(task.start_pose)}\n   零位     {ui.fmt_pose(deploy['poses']['zero'])}")
    if mode == "real":
        print(ui.c("   ⚠ 真机模式：请确认急停按钮在手边、工作空间无人。", "red", "bold"))
    if not args.yes:
        try:
            if input(ui.c("   按 Enter 启动，输入 q 退出: ", "cyan")).strip().lower() == "q":
                return
        except EOFError:
            return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = LOG_ROOT / f"{stamp}_{task.name}_{mode}"
    session.mkdir(parents=True, exist_ok=True)
    meta = {"task": task.name, "checkpoint": str(ckpt.path), "checkpoint_name": ckpt.name, "mode": mode,
            "preset": preset_name, "preset_values": preset, "backend": args.backend, "device": device,
            "spec": spec.summary(), "device_python": sys.executable, "model_python": args.model_python}
    (session / "session.json").write_text(json.dumps({**meta, "deploy": deploy}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    def say(msg):
        print(f"   {ui.c('›', 'cyan')} {msg}", flush=True)

    server = arms = cams = engine = keys = None
    released = False
    sigint = {"n": 0}

    def on_sigint(_s, _f):
        sigint["n"] += 1
        if engine is None:
            raise KeyboardInterrupt

    try:
        # ---- model server (loads in the background while the hardware starts)
        if not Path(args.model_python).exists() and os.sep in args.model_python:
            raise RuntimeError(f"模型环境解释器不存在: {args.model_python}（在 local.env 设置 MODEL_PY）")
        say(f"启动模型服务（日志 {session / 'server.log'}）")
        server = ServerProcess(args.model_python, ckpt.path, deploy["policy_server"], session / "server.log",
                               args.backend, device)

        # ---- hardware
        cam_names = sorted({i.camera for i in spec.images})
        cc = deploy["cameras"]
        if mode == "sim" or args.sim_hardware:
            from act_deploy.hw.arms import SimArms
            from act_deploy.hw.cameras import CameraRig

            arms = SimArms(np.asarray(deploy["poses"]["zero"]))
            cams = CameraRig.fake(cam_names, cc["width"], cc["height"], cc["fps"])
        else:
            from act_deploy.hw.arms import FollowerArms, load_can_channels
            from act_deploy.hw.cameras import CameraRig
            from act_deploy.hw.preflight import check_can

            problems = check_can(load_can_channels(HARDWARE_YAML))
            if problems:
                raise RuntimeError("CAN 检查未通过:\n   " + "\n   ".join(problems))
            say("CAN 检查通过；启动左右从臂 worker（电机上电，原地保持）…")
            arms = FollowerArms(HARDWARE_YAML, log_dir=session, pid_dir=PID_DIR)
            say("机械臂 worker 就绪；打开相机…")
            cams = CameraRig.realsense({n: cc["serials"][n] for n in cam_names}, cc["width"], cc["height"], cc["fps"])
        cams.wait_ready()
        say("相机就绪；等待模型服务…")
        h = wait_server(server, deploy["policy_server"]["startup_timeout_s"], spec, say)
        say(f"模型服务就绪: {h.get('backend')}  预热 {h.get('warmup_ms')} ms")

        # ---- engine
        from act_deploy.app.engine import Engine, EngineOptions

        opts = EngineOptions(mode=mode, rate_hz=deploy["control"]["rate_hz"], motion=deploy["motion"],
                             poses=deploy["poses"], safety=deploy["safety"], policy=deploy["policy"],
                             preset=preset, recording=deploy["recording"],
                             max_frame_age_s=cc["max_frame_age_s"], start_pose=np.asarray(task.start_pose),
                             max_episode_s=task.max_episode_s, session_dir=session, meta=meta)
        engine = Engine(opts, spec, arms, cams, server.client)
        signal.signal(signal.SIGINT, on_sigint)
        engine.start()
        released = run_console(engine, server, args, deploy, task, ckpt, mode, preset, sigint)
    except KeyboardInterrupt:
        print(ui.c("\n   已中断", "yellow"))
    except Exception as exc:  # noqa: BLE001
        print(ui.c(f"\n   ✖ 错误: {exc}", "red"))
    finally:
        if engine is not None and not released and engine.phase != "RELEASED":
            if engine.done.is_set() or sigint["n"] >= 3:
                print(ui.c("   ⚠ 机械臂 worker 仍在运行并保持通电。start.sh 将自动进入救援（回零后断电）；"
                           "或手动 bash start.sh --rescue", "red", "bold"))
            else:
                engine.command("release")
                print("   正在回零位并释放…")
                engine.wait_done(120)
        elif engine is None and arms is not None:
            print("   机械臂未运动过，直接释放")
            arms.release()
        if cams is not None:
            cams.close()
        if server is not None:
            server.stop()
        print(f"   日志目录: {session}")


def run_console(engine, server, args, deploy, task, ckpt, mode, preset, sigint) -> bool:
    header = {"task": task.label, "ckpt": ckpt.name, "preset": preset["label"]}
    dash = ui.Dashboard(engine, header, server.alive)
    steps = [s.strip() for s in args.script.split(",") if s.strip()]
    keys = None if steps else ui.KeyReader()
    period = 1.0 / deploy["ui"]["refresh_hz"]
    seen_events = 0
    step_no = 0
    handled_sigint = 0
    server_dead_reported = False
    if not args.headless:
        ui.enter_screen()
    try:
        next_draw = 0.0
        while not engine.done.is_set():
            # Ctrl+C escalation
            if sigint["n"] > handled_sigint:
                handled_sigint = sigint["n"]
                if handled_sigint == 1:
                    engine.command("estop", "Ctrl+C")
                    dash.extra.append("Ctrl+C：已急停。再按一次 = 回零并释放退出")
                elif handled_sigint == 2:
                    engine.command("release")
                    dash.extra.append("Ctrl+C×2：回零后释放退出。再按一次 = 立即退出（worker 保持通电）")
                else:
                    return False
            if not server.alive() and not server_dead_reported:
                server_dead_reported = True
                engine.command("policy_error", "模型服务进程退出")
                engine.event("error", "模型服务进程意外退出，见 server.log")
            # input
            if steps:
                step = steps[0]
                if step.startswith("wait:"):
                    parts = step.split(":")
                    if engine.phase == parts[1]:
                        steps.pop(0)
                        step_no += 1
                    elif len(parts) > 2 and time.monotonic() > _deadline(f"{step_no}", float(parts[2])):
                        raise RuntimeError(f"脚本等待 {parts[1]} 超时（当前 {engine.phase}）")
                elif step.startswith("sleep:"):
                    if time.monotonic() > _deadline(f"{step_no}", float(step.split(":")[1])):
                        steps.pop(0)
                        step_no += 1
                else:
                    k = {"enter": "next", "space": "estop"}.get(step, ui.KEYMAP.get(step, step))
                    engine.command(k)
                    steps.pop(0)
                    step_no += 1
                time.sleep(0.05)
            elif keys is None:  # script finished: just watch the engine
                time.sleep(0.05)
            else:
                k = keys.get(timeout=0.05)
                if k == "h":
                    dash.show_help = not dash.show_help
                elif k in ui.KEYMAP:
                    engine.command(ui.KEYMAP[k])
            # output
            if args.headless:
                with engine.lock:
                    evs, total = list(engine.events), engine.n_events
                new = min(total - seen_events, len(evs))
                for t, level, msg in (evs[-new:] if new > 0 else []):
                    print(f"   {t} [{level}] [{engine.phase}] {msg}", flush=True)
                seen_events = total
            elif time.monotonic() >= next_draw:
                next_draw = time.monotonic() + period
                sys.stdout.write(dash.render())
                sys.stdout.flush()
        return engine.phase == "RELEASED"
    finally:
        if keys is not None:
            keys.close()
        if not args.headless:
            ui.leave_screen()
        s = engine.snapshot()
        print(ui.rule("本次会话"))
        for r in engine.results:
            print(f"   #{r['episode']}  {r['duration']:5.1f}s  {r['reason']}")
        for t, level, msg in s["events"]:
            print(f"   {t} [{level}] {msg}")


_DEADLINES: dict[str, float] = {}


def _deadline(key: str, seconds: float) -> float:
    if key not in _DEADLINES:
        _DEADLINES[key] = time.monotonic() + seconds
    return _DEADLINES[key]


if __name__ == "__main__":
    main()
