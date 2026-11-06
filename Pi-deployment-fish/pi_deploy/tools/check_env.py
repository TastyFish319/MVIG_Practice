"""Read-only environment check (never powers anything).

    python -m pi_deploy.tools.check_env --model-python <py312>
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

from pi_deploy import PROJECT_ROOT
from pi_deploy.app.console import c, rule

MODEL_SNIPPET = r"""
import json, importlib.metadata as m
import torch
out = {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
       "cuda": torch.cuda.is_available(), "lerobot": m.version("lerobot"), "numpy": m.version("numpy")}
if out["cuda"]:
    out["gpu"] = torch.cuda.get_device_name(0)
    out['cuda_build'] = torch.version.cuda
    out['capability'] = torch.cuda.get_device_capability(0)
    out['bf16'] = torch.cuda.is_bf16_supported()
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    out["matmul_ok"] = bool(torch.isfinite(x @ x).all())
print(json.dumps(out))
"""


def line(ok, name, detail="") -> bool:
    mark = c("  OK ", "green") if ok else c(" FAIL", "red")
    print(f" {mark} {name:<28} {c(detail, 'grey') if ok else c(detail, 'red')}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-python", default=os.environ.get("MODEL_PY"))
    p.add_argument('--skip-hardware', action='store_true', help='仅软件检查，不枚举CAN/相机')
    p.add_argument('--task', help='只检查指定任务的权重')
    args = p.parse_args()
    ok = True
    print(rule("设备环境（本解释器）"))
    print(f"   {sys.executable}  Python {sys.version.split()[0]}")
    for mod in ("numpy", "yaml", "cv2", "portal", "pyrealsense2", "i2rt"):
        try:
            m = importlib.import_module(mod)
            ok &= line(True, mod, getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            ok &= line(False, mod, str(exc))
    print(rule("模型环境"))
    if not args.model_python or not os.path.exists(args.model_python):
        ok &= line(False, "model python", f"不存在: {args.model_python}")
    else:
        r = subprocess.run([args.model_python, "-c", MODEL_SNIPPET], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            import json
            info = json.loads(r.stdout.strip().splitlines()[-1])
            valid = (info.get('cuda') and info.get('matmul_ok') and info.get('bf16')
                     and info.get('lerobot') == '0.6.1')
            if tuple(info.get('capability', (0, 0))) >= (12, 0):
                valid = valid and tuple(map(int, str(info.get('cuda_build', '0.0')).split('.')[:2])) >= (12, 8)
            ok &= line(bool(valid), 'Pi / CUDA / bf16', str(info))
        else:
            ok &= line(False, "torch / lerobot", (r.stderr or r.stdout).strip().splitlines()[-1:])
    print(rule("配置与权重"))
    try:
        from pi_deploy.config import HARDWARE_YAML, load_deploy, load_tasks
        from pi_deploy.hw.follower.config import load_teleoperation_config
        from pi_deploy.model.spec import resolve_spec
        from pi_deploy.registry import scan_checkpoints

        deploy = load_deploy()
        tasks = load_tasks(deploy)
        from pi_deploy.config import resolve_path
        tok = resolve_path(deploy['inference']['tokenizer'])
        ok &= line((tok/'tokenizer_config.json').is_file(), '本地 tokenizer', str(tok))
        if args.task:
            if args.task not in tasks:
                raise ValueError(f'未知任务 {args.task}')
            tasks = {args.task: tasks[args.task]}
        load_teleoperation_config(HARDWARE_YAML)
        ok &= line(True, "configs/*.yaml", "deploy / tasks / hardware 校验通过")
        total = 0
        for t in tasks.values():
            ck = scan_checkpoints(t)
            if not t.enabled:
                line(True, f"任务 {t.name}", f"未启用，{len(ck)} 个权重")
                continue
            total += len(ck)
            if not ck:
                print(f'   任务 {t.name}: 尚未放权重（不影响其他任务）')
            else:
                line(True, f'任务 {t.name}', f'{len(ck)} 个权重')
            for info in ck:
                try:
                    s = resolve_spec(info.path, t.model_overrides)
                    line(True, f"  {info.name}", f"state{s.state_dim} -> {s.horizon}x{s.action_dim}")
                except Exception as exc:  # noqa: BLE001
                    ok &= line(False, f"  {info.name}", str(exc))
        ok &= line(total > 0, '存在可部署权重', f'{total} 个')
    except Exception as exc:  # noqa: BLE001
        ok &= line(False, "configs", str(exc))
        return 1
    print(rule("硬件（只读）"))
    if os.name == "posix" and not args.skip_hardware:
        from pi_deploy.hw.arms import leftover_workers, load_can_channels, load_ports, port_open
        from pi_deploy.hw.preflight import check_can, list_realsense

        problems = check_can(load_can_channels(HARDWARE_YAML))
        ok &= line(not problems, "CAN 从臂链路", "; ".join(problems) or str(load_can_channels(HARDWARE_YAML)))
        busy = [f"{s}:{p}" for s, p in load_ports(HARDWARE_YAML).items() if port_open(p)]
        ok &= line(not busy, "worker 端口空闲", ", ".join(busy) or "1234/1235 空闲")
        left = leftover_workers(PROJECT_ROOT / "logs" / ".workers")
        ok &= line(not left, "无遗留 worker", str(left) if left else "")
        serials = list_realsense()
        want = set(map(str, deploy["cameras"]["serials"].values()))
        if serials is None:
            ok &= line(False, "RealSense", "pyrealsense2 不可用")
        else:
            ok &= line(want <= set(serials), "RealSense 相机", f"需要 {sorted(want)}，检测到 {serials}")
    else:
        line(True, "硬件检查", "已跳过；未验收硬件")
    print(rule())
    print(c(" ALL OK", "green", "bold") if ok else c(" 有检查未通过", "red", "bold"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
