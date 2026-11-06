"""Explicit rescue of this project's surviving follower workers. No force-release bypass."""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
from pi_deploy import JOINT_IDX, GRIPPER_IDX
from pi_deploy.config import HARDWARE_YAML, load_deploy
from pi_deploy.control.trajectory import Route
from pi_deploy.hw.arms import FollowerArms, leftover_workers, load_ports, port_open, SIDES
from pi_deploy.app.main import PID_DIR


def rescue_arm(arms, cfg):
    """Each arms instance owns exactly one verified side; unknown slots are never sent."""
    side = arms.sides[0]
    slot = SIDES.index(side) * 7
    joints = list(range(slot, slot+6)); grip = [slot+6]
    # Re-read AFTER operator confirmation; never align to a state captured before the prompt.
    state = arms.read(allow_fault=True)
    if arms.fault:
        arms.reset_fault()
        state = arms.read()
    if not np.isfinite(state).all():
        raise RuntimeError('救援反馈非法')
    arms.begin(state)
    m = cfg['motion']
    goal = state.copy(); goal[slot:slot+7] = np.asarray(cfg['poses']['zero'])[slot:slot+7]
    limits = {k:m[k] for k in ('max_joint_velocity', 'max_joint_acceleration', 'max_gripper_velocity', 'min_duration_s')}
    route = Route(state, [goal], limits, time.monotonic())
    print(f'{side} 回零，约 {route.duration:.1f}s；Ctrl+C中止保持，不断电')
    dt = 1 / cfg['control']['rate_hz']; nxt = time.monotonic(); stable = 0
    try:
        while True:
            now = time.monotonic(); target = route.at(now); current = arms.read()
            if np.abs(current-target)[joints].max() > m['max_tracking_error']:
                raise RuntimeError('救援轨迹跟随误差超限')
            arms.submit(target)
            if route.done(now):
                delta = np.abs(current-goal)
                ok = delta[joints].max() <= m['settle_tolerance'] and delta[grip].max() <= m['gripper_tolerance']
                stable = stable + 1 if ok else 0
                if stable >= 3:
                    break
                if now > route.t0 + route.duration + m['settle_timeout_s']:
                    raise RuntimeError('回零到位超时，禁止释放')
            nxt += dt
            time.sleep(max(0, nxt-time.monotonic()))
        # release() revalidates the retained worker identity before sending a process signal.
        arms.release()
        print(f'{side} 已确认零位并释放')
        return True
    except BaseException:
        arms.end()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--yes', action='store_true', help='显式授权救援回零，仍强制检查到位')
    args = ap.parse_args()
    ours = leftover_workers(PID_DIR)
    if args.check_only:
        return 1 if ours else 0
    if not ours:
        print('没有本工程遗留 worker；不接管其他工程端口。')
        return 0
    ports = load_ports(HARDWARE_YAML)
    active = sorted({w['side'] for w in ours if port_open(ports[w['side']])})
    unreachable = sorted({w['side'] for w in ours} - set(active))
    if unreachable:
        print(f'这些worker不可连接，不能软件接管：{unreachable}；请现场检查，其他健康侧仍可单独救援。')
    if not active:
        return 1
    print(f'将只救援身份匹配的本工程从臂：{active}，不会给缺失侧发送指令。')
    if not args.yes:
        if input('逐侧平滑回零、确认到位后释放？输入 y 确认（其他输入保持）: ').strip().lower() != 'y':
            print('未操作，worker继续保持。')
            return 1
    cfg = load_deploy(); failed = bool(unreachable)
    # One failed side must not prevent the other healthy arm from being rescued.
    for side in active:
        arm = None
        try:
            arm = FollowerArms(HARDWARE_YAML, log_dir=None, pid_dir=PID_DIR, attach=True, sides=(side,))
            rescue_arm(arm, cfg)
        except KeyboardInterrupt:
            print('救援已中止；未释放的worker保持。')
            return 1
        except Exception as exc:
            failed = True
            print(f'{side} 救援失败，不释放：{exc}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
