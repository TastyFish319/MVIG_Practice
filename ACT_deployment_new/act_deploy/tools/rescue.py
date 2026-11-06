"""Rescue: bring arms held by leftover workers smoothly to zero, then release the motors.

Used when the console died while the arms were powered (the workers keep holding the pose).
start.sh runs it automatically after the console exits if workers are still alive.

    python -m act_deploy.tools.rescue                 ask, then current -> zero -> motors off
    python -m act_deploy.tools.rescue --check-only    exit 1 if leftover workers exist
    python -m act_deploy.tools.rescue --release-now   motors off immediately (arms fall!)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from act_deploy import JOINT_IDX
from act_deploy.app.main import PID_DIR
from act_deploy.config import HARDWARE_YAML, load_deploy
from act_deploy.control.trajectory import Route
from act_deploy.hw.arms import FollowerArms, leftover_workers, load_ports, port_open


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--release-now", action="store_true")
    args = p.parse_args()

    ours = leftover_workers(PID_DIR)
    ports = load_ports(HARDWARE_YAML)
    busy = [s for s, port in ports.items() if port_open(port)]
    if args.check_only:
        return 1 if ours else 0
    if not ours:
        if busy:
            print(f"端口 {busy} 被占用，但不是本工程的 worker（可能是 SI 采集 WebUI），不接管。")
        else:
            print("没有遗留的机械臂 worker，无需救援。")
        return 0
    print(f"发现遗留 worker: {[(w['side'], w['pid']) for w in ours]}（手臂保持通电）")
    arms = FollowerArms(HARDWARE_YAML, log_dir=None, pid_dir=PID_DIR, attach=True)
    if args.release_now:
        print("立即释放电机（手臂会掉落）")
        arms.release()
        return 0
    cfg = load_deploy()
    zero = np.asarray(cfg["poses"]["zero"])
    state = arms.read(allow_fault=True)
    print("当前:", np.round(state, 2), "  故障:", arms.fault)
    print("零位:", np.round(zero, 2))
    if not args.yes:
        ans = input("平滑移动到零位后释放电机？[Y/n/x=立即释放] ").strip().lower()
        if ans == "x":
            arms.release()
            return 0
        if ans == "n":
            print("未操作（worker 继续保持）。")
            return 1
    if arms.fault:
        arms.reset_fault()
        state = arms.read()
    arms.begin(state)
    m = cfg["motion"]
    limits = {k: m[k] for k in ("max_joint_velocity", "max_joint_acceleration", "max_gripper_velocity", "min_duration_s")}
    route = Route(state, [zero], limits, time.monotonic())
    print(f"回零中，约 {route.duration:.1f}s …（Ctrl+C 中止 = 原地保持）")
    dt = 1.0 / cfg["control"]["rate_hz"]
    nxt = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            target = route.at(now)
            cur = arms.read()
            err = np.abs(cur - target)[list(JOINT_IDX)].max()
            if err > m["max_tracking_error"]:
                print(f"跟随误差 {err:.2f} rad 超限 -> 停止（worker 保持）")
                arms.end()
                return 1
            arms.submit(target)
            if route.done(now) and now - route.t0 > route.duration + 1.0:
                break
            nxt += dt
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        arms.end()
        print("已中止，worker 原地保持。")
        return 1
    print("到达零位，释放电机。")
    arms.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
