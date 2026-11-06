"""Entry point of the deployment console (run by start.sh in the device environment).

    python -m pi_deploy.app.main --model-python <py312>              interactive menus
    python -m pi_deploy.app.main --task block_into_cup --checkpoint v3_long2_017000 \
        --mode sim --preset chunk_10 --headless --script "enter,wait:READY,enter,sleep:3,f,wait:ZERO,r"

Start-up order:
    menus -> session dir -> model server process (loads + completes synthetic warmup before hardware)
          -> arm workers + cameras -> engine + live warmup -> dashboard
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
import uuid
from pathlib import Path

import numpy as np

from pi_deploy import PROJECT_ROOT
from pi_deploy.app import console as ui
from pi_deploy.config import HARDWARE_YAML, DEPLOY_YAML, TASKS_YAML, ConfigError, load_deploy, load_tasks
from pi_deploy.ipc import PolicyClient
from pi_deploy.model.spec import resolve_spec
from pi_deploy.registry import scan_checkpoints

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
    def __init__(self, python: str, ckpt: Path, cfg: dict, log_path: Path, backend: str, device: str,
                 inference: dict, overrides: dict):
        self.port = free_port(cfg["port"])
        self.log_path = log_path
        self.instance_id = uuid.uuid4().hex
        self.verified = False
        self.expected_backend = backend
        from pi_deploy.inference.options import gripper_options
        self.expected_gripper = gripper_options(resolve_spec(ckpt, overrides).contract, inference['gripper_floor'])
        cmd = [python, "-m", "pi_deploy.server.policy_server", "--checkpoint", str(ckpt),
               "--port", str(self.port), "--device", device, "--backend", backend,
               "--warmup", str(cfg["warmup_iterations"]), "--keep-warm-s", str(cfg["keep_warm_s"]),
               '--tokenizer', str(inference['tokenizer']), '--gripper-floor', inference['gripper_floor'],
               '--seed', str(inference['seed']), '--model-overrides', json.dumps(overrides),
               '--instance-id', self.instance_id,
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
            h = self.client.health(timeout_s=2.0)
        except Exception:  # noqa: BLE001 - not listening yet
            return None
        if h.get('pid') != self.proc.pid or h.get('instance_id') != self.instance_id:
            raise RuntimeError('连接到的模型服务不属于本次子进程；拒绝推理或shutdown')
        self.verified = True
        backend = h.get('backend')
        if backend is not None:
            if backend.get('backend') != self.expected_backend:
                raise RuntimeError('模型服务后端不匹配')
            if self.expected_backend == 'lerobot' and backend.get('gripper_filter') != self.expected_gripper:
                raise RuntimeError('模型服务的夹爪floor参数未按本次选择生效')
        return h

    def stop(self) -> None:
        if self.alive():
            if self.verified:
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
        if not task.enabled:
            raise SystemExit(f'任务 {task.name} 未启用')
    else:
        opts = [(f"{t.label}  ({t.name})", "权重目录: " + ", ".join(str(p) for p in t.checkpoint_dirs), t.enabled)
                for t in tasks.values()]
        default = next((i for i, t in enumerate(tasks.values()) if t.enabled), 0)
        task = tasks[names[ui.choose("① 选择任务", opts, default)]]

    ckpts = scan_checkpoints(task)
    if not ckpts:
        raise SystemExit(f"任务 {task.name} 的权重目录里没有找到完整的 Pi checkpoint: "
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
        def resolved_detail(p):
            p = dict(p)
            if p.get('exec_steps') == 0:
                p['exec_steps'] = spec.n_action_steps
            return detail(p)
        preset_name = keys[ui.choose("④ 选择执行方式（动作平滑策略）",
                                     [(presets[k]["label"], resolved_detail(presets[k]), True) for k in keys], default)]
    floor = args.gripper_floor
    if floor is None:
        options = [('checkpoint', '沿用权重配置（推荐）'), ('on', '启用夹爪软下限'), ('off', '关闭软下限（允许完全闭合）')]
        floor = options[ui.choose('⑤ 夹爪 floor 滤波', [(label, key, True) for key, label in options],
                                  [k for k, _ in options].index(deploy['inference']['gripper_floor']))][0]
    return task, ckpt, spec, mode, preset_name, floor


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
            if hs.get('fingerprint') != spec.fingerprint or hs.get('output_semantics') != 'absolute_filtered':
                raise RuntimeError('模型进程与设备进程的契约/绝对输出语义不一致')
            return h
        if status == "error":
            raise RuntimeError(f"模型服务加载失败: {h.get('error')}\n{server.tail()}")
        time.sleep(0.3)
    raise TimeoutError(f"模型服务 {timeout:.0f}s 内未就绪:\n{server.tail()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-python", default=os.environ.get("MODEL_PY", sys.executable))
    p.add_argument('--deploy-config', type=Path, default=DEPLOY_YAML)
    p.add_argument('--tasks-config', type=Path, default=TASKS_YAML)
    p.add_argument("--task")
    p.add_argument("--checkpoint")
    p.add_argument("--mode", choices=("real", "shadow", "sim"))
    p.add_argument("--preset")
    p.add_argument("--device", help="override policy_server.device (cuda/cpu)")
    p.add_argument('--gripper-floor', choices=('checkpoint', 'on', 'off'),
                   help='夹爪软下限：沿用权重 / 强制开启 / 关闭（关闭仍裁剪到0..1）')
    p.add_argument("--backend", choices=("lerobot", "mock"), default="lerobot",
                   help="mock = fake model without torch (UI tests only)")
    p.add_argument("--yes", action="store_true", help="skip the final confirmation")
    p.add_argument("--sim-hardware", action="store_true",
                   help="dev: simulated arms + synthetic cameras in any mode (tests real/shadow logic offline)")
    p.add_argument("--headless", action="store_true", help="print events instead of the dashboard")
    p.add_argument("--script", default="", help="comma separated keys / wait:PHASE[:s] / sleep:s (tests)")
    args = p.parse_args()
    if args.backend == 'mock' and args.mode in ('real', 'shadow') and not args.sim_hardware:
        raise SystemExit('mock仅支持模拟硬件')

    try:
        deploy = load_deploy(args.deploy_config)
        tasks = load_tasks(deploy, args.tasks_config)
    except ConfigError as exc:
        raise SystemExit(f"配置错误: {exc}") from exc

    ui.banner([ui.c("流程：选任务 → 选权重 → 选模式 → 选执行方式 → 启动", "grey"),
               ui.c(f"设备环境 {sys.executable}", "grey"), ui.c(f"模型环境 {args.model_python}", "grey")])
    task, ckpt, spec, mode, preset_name, floor = pick(args, deploy, tasks)
    from pi_deploy.config import resolve_path
    from pi_deploy.inference.options import gripper_options
    from pi_deploy.control.scheduler import ActionScheduler
    preset = dict(deploy["execution"]["presets"][preset_name])
    if preset.get('exec_steps') == 0:
        preset['exec_steps'] = spec.n_action_steps
    ActionScheduler(mode=preset['mode'], fps=spec.fps, horizon=spec.horizon, offset=0,
                    **{k: v for k, v in preset.items() if k not in ('label', 'mode', 'smoothing_tau_s')})
    inference = dict(deploy['inference'], tokenizer=str(resolve_path(deploy['inference']['tokenizer'])), gripper_floor=floor)
    effective_gripper = gripper_options(spec.contract, floor)
    if args.backend == 'mock' and mode != 'sim' and not args.sim_hardware:
        raise SystemExit('mock 后端仅允许模拟硬件；禁止用于真实机械臂')
    device = args.device or deploy["policy_server"]["device"]

    print(ui.rule("确认"))
    print(f"   任务  {task.label}\n   权重  {ckpt.name}  ({ckpt.path})\n   模式  {ui.MODE_NAMES[mode]}"
          f"\n   执行  {preset['label']}\n   推理  {args.backend} on {device}"
          f"\n   初始姿态 {ui.fmt_pose(task.start_pose)}\n   零位     {ui.fmt_pose(deploy['poses']['zero'])}"
          f"\n   夹爪 floor: {floor} → {effective_gripper}（仅真实 Pi 后端应用）")
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
    meta.update(inference=inference, gripper_filter=effective_gripper, contract_fingerprint=spec.fingerprint,
                simulated_hardware=(mode == 'sim' or args.sim_hardware))
    (session / "session.json").write_text(json.dumps({**meta, "deploy": deploy}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    def say(msg):
        print(f"   {ui.c('›', 'cyan')} {msg}", flush=True)

    server = arms = cams = engine = keys = None
    released = False
    exit_code = 0
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
                               args.backend, device, inference, task.model_overrides)
        h = wait_server(server, deploy['policy_server']['startup_timeout_s'], spec, say)
        say(f'模型已加载且预热完成: {h.get("backend")}，预热耗时 {h.get("warmup_ms")} ms')

        # ---- hardware
        cam_names = sorted({i.camera for i in spec.images})
        cc = deploy["cameras"]
        if mode == "sim" or args.sim_hardware:
            from pi_deploy.hw.arms import SimArms
            from pi_deploy.hw.cameras import CameraRig

            arms = SimArms(np.asarray(deploy["poses"]["zero"]))
            cams = CameraRig.fake(cam_names, cc["width"], cc["height"], cc["fps"])
        else:
            from pi_deploy.hw.arms import FollowerArms, load_can_channels
            from pi_deploy.hw.cameras import CameraRig
            from pi_deploy.hw.preflight import check_can
            from pi_deploy.hw.arms import leftover_workers
            if leftover_workers(PID_DIR):
                raise RuntimeError('检测到本工程遗留worker。请先明确运行 bash start.sh --rescue，完成后重新启动。')

            problems = check_can(load_can_channels(HARDWARE_YAML))
            if problems:
                raise RuntimeError("CAN 检查未通过:\n   " + "\n   ".join(problems))
            say("CAN 检查通过；启动左右从臂 worker（电机上电，原地保持）…")
            arms = FollowerArms(HARDWARE_YAML, log_dir=session, pid_dir=PID_DIR)
            say("机械臂 worker 就绪；打开相机…")
            cams = CameraRig.realsense({n: cc["serials"][n] for n in cam_names}, cc["width"], cc["height"], cc["fps"])
        cams.wait_ready()
        say("相机就绪；开始真实观测预热…")

        # ---- engine
        from pi_deploy.app.engine import Engine, EngineOptions

        opts = EngineOptions(mode=mode, rate_hz=deploy["control"]["rate_hz"], motion=deploy["motion"],
                             poses=deploy["poses"], safety=deploy["safety"], policy=deploy["policy"],
                             preset=preset, recording=deploy["recording"],
                             max_frame_age_s=cc["max_frame_age_s"], start_pose=np.asarray(task.start_pose),
                             max_episode_s=task.max_episode_s, session_dir=session, meta=meta)
        engine = Engine(opts, spec, arms, cams, server.client)
        signal.signal(signal.SIGINT, on_sigint)
        engine.start()
        released = run_console(engine, server, args, deploy, task, ckpt, mode, preset, sigint)
        if not released:
            exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
        print(ui.c("\n   已中断", "yellow"))
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        print(ui.c(f"\n   ✖ 错误: {exc}", "red"))
    finally:
        if engine is not None and not released and engine.phase != "RELEASED":
            if engine.done.is_set() or sigint["n"] >= 3:
                print(ui.c('   ⚠ worker仍在运行并保持；请显式运行 bash start.sh --rescue 检查和回零释放', 'red', 'bold'))
            else:
                engine.command('detach', '控制台异常退出，停止策略并保持')
                engine.wait_done(3)
                print('   已请求原地保持；未回零或释放。需要时运行 bash start.sh --rescue')
        elif engine is None and arms is not None:
            if mode == 'sim' or args.sim_hardware:
                arms.release()
            else:
                print('   启动未完成，worker 保持通电；请使用 start.sh --rescue 检查并回零后释放')
                arms.end()
        if engine is not None:
            for thread in engine._save_threads:
                thread.join(timeout=30)
        if cams is not None:
            cams.close()
        if server is not None:
            server.stop()
        print(f"   日志目录: {session}")
    return exit_code


def run_console(engine, server, args, deploy, task, ckpt, mode, preset, sigint) -> bool:
    header = {"task": task.label, "ckpt": ckpt.name, "preset": preset["label"],
              'floor': engine.o.meta.get('gripper_filter', {})}
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
    sys.exit(main())
