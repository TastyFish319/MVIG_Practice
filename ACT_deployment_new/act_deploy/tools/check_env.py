"""Read-only environment check (never powers anything).

    python -m act_deploy.tools.check_env --model-python <py312>
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

from act_deploy import PROJECT_ROOT
from act_deploy.app.console import c, rule

MODEL_SNIPPET = r"""
import json, importlib.metadata as m
import torch
out = {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
       "cuda": torch.cuda.is_available(), "lerobot": m.version("lerobot"), "numpy": m.version("numpy")}
if out["cuda"]:
    out["gpu"] = torch.cuda.get_device_name(0)
    x = torch.randn(256, 256, device="cuda"); out["matmul_ok"] = bool(torch.isfinite(x @ x).all())
print(json.dumps(out))
"""


def line(ok, name, detail="") -> bool:
    mark = c("  OK ", "green") if ok else c(" FAIL", "red")
    print(f" {mark} {name:<28} {c(detail, 'grey') if ok else c(detail, 'red')}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-python", default=os.environ.get("MODEL_PY"))
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
            info = r.stdout.strip().splitlines()[-1]
            ok &= line('"cuda": true' in info, "torch / lerobot / cuda", info)
        else:
            ok &= line(False, "torch / lerobot", (r.stderr or r.stdout).strip().splitlines()[-1:])
    print(rule("配置与权重"))
    try:
        from act_deploy.config import HARDWARE_YAML, load_deploy, load_tasks
        from act_deploy.hw.follower.config import load_teleoperation_config
        from act_deploy.model.spec import resolve_spec
        from act_deploy.registry import scan_checkpoints

        deploy = load_deploy()
        tasks = load_tasks(deploy)
        load_teleoperation_config(HARDWARE_YAML)
        ok &= line(True, "configs/*.yaml", "deploy / tasks / hardware 校验通过")
        for t in tasks.values():
            ck = scan_checkpoints(t)
            if not t.enabled:
                line(True, f"任务 {t.name}", f"未启用，{len(ck)} 个权重")
                continue
            ok &= line(bool(ck), f"任务 {t.name}", f"{len(ck)} 个权重" if ck else "没有权重")
            for info in ck:
                try:
                    s = resolve_spec(info.path, t.model_overrides)
                    line(True, f"  {info.name}", f"state{s.state_dim} -> {s.horizon}x{s.action_dim}")
                except Exception as exc:  # noqa: BLE001
                    ok &= line(False, f"  {info.name}", str(exc))
    except Exception as exc:  # noqa: BLE001
        ok &= line(False, "configs", str(exc))
        return 1
    print(rule("硬件（只读）"))
    if os.name == "posix":
        from act_deploy.hw.arms import leftover_workers, load_can_channels, load_ports, port_open
        from act_deploy.hw.preflight import check_can, list_realsense

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
        line(True, "硬件检查", "非 Linux，跳过")
    print(rule())
    print(c(" ALL OK", "green", "bold") if ok else c(" 有检查未通过", "red", "bold"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
