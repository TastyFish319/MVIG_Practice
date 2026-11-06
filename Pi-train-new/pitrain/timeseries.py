"""时间对齐、速度计算、首尾静止段裁剪。

原始数据各传感器频率不同（关节 90Hz、夹爪/相机 30Hz），而且时间戳不严格同步。
这里把所有数据流"重采样"到统一的 fps 时间网格上：

    关节 90Hz:  | | | | | | | | | | | | | | | |
    相机 30Hz:  |     |     |     |     |     |
    网格 30Hz:  ^     ^     ^     ^     ^     ^   每个网格点取"该时刻之前最新的一帧"

"取之前最新的一帧"叫因果采样（causal / zero-order hold）：只用过去的信息，
与真机部署时的情况一致——机器人永远看不到未来的传感器读数。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nedf3 import EpisodeRaw


@dataclass
class Layout:
    """state / action 各维度的来源与顺序。"""

    action_arms: tuple[str, ...]          # ("right",) 或 ("right", "left")
    observe_arms: tuple[str, ...] = ("right", "left")

    def __post_init__(self):
        # 动作臂必须排在 state 最前面：LeRobot 的相对动作按 state[:action_dim] 对齐相减
        rest = tuple(a for a in self.observe_arms if a not in self.action_arms)
        self.state_arms = tuple(self.action_arms) + rest

    @staticmethod
    def arm_names(side: str) -> list[str]:
        return [f"{side}_joint{i}" for i in range(6)] + [f"{side}_gripper"]

    @property
    def state_names(self) -> list[str]:
        return [n for arm in self.state_arms for n in self.arm_names(arm)]

    @property
    def action_names(self) -> list[str]:
        return [n for arm in self.action_arms for n in self.arm_names(arm)]

    @property
    def velocity_names(self) -> list[str]:
        return [f"{n}_vel" for n in self.state_names]


def causal_index(times: np.ndarray, grid: np.ndarray, max_age_ns: int, label: str) -> np.ndarray:
    idx = np.searchsorted(times, grid, side="right") - 1
    if np.any(idx < 0):
        raise ValueError(f"{label}: 网格起点早于第一帧")
    age = grid - times[idx]
    if np.any(age > max_age_ns):
        worst = int(np.argmax(age))
        raise ValueError(f"{label}: 数据断流 {age[worst] / 1e6:.0f}ms（>{max_age_ns / 1e6:.0f}ms）于第 {worst} 帧")
    return idx


def causal_velocity(times: np.ndarray, values: np.ndarray, grid: np.ndarray, window_s: float) -> np.ndarray:
    """在每个网格时刻 t，用 (t-window, t] 内的原始采样做最小二乘直线拟合，斜率即速度。

    比简单差分 (q[t]-q[t-1])/dt 更抗噪，且只用过去数据（因果）。
    """
    out = np.zeros((len(grid), values.shape[1]), np.float64)
    hi_all = np.searchsorted(times, grid, side="right")
    lo_all = np.searchsorted(times, grid - int(window_s * 1e9), side="right")
    for i, (lo, hi) in enumerate(zip(lo_all, hi_all, strict=True)):
        if hi - lo < 2:
            lo = max(0, hi - 2)  # 窗口内不足两点：退化为最近两点差分
            if hi - lo < 2:
                continue
        t = (times[lo:hi] - times[hi - 1]) / 1e9
        x = values[lo:hi]
        tc = t - t.mean()
        denom = float((tc * tc).sum())
        if denom > 0:
            out[i] = (tc[:, None] * (x - x.mean(0))).sum(0) / denom
    return out


def idle_bounds(action: np.ndarray, fps: int, joint_tol: float, grip_tol: float, margin_s: float) -> tuple[int, int]:
    """找出首尾"主臂还没动/已经停了"的静止段，返回保留区间 [start, end)。

    为什么要裁剪：开头操作员常常要愣一两秒才开始动。如果保留，模型会学到
    "看到这个画面就原地不动"，部署时可能出现开局卡住不动的现象。
    """
    n = len(action)
    grip_cols = [i for i in range(action.shape[1]) if i % 7 == 6]
    joint_cols = [i for i in range(action.shape[1]) if i % 7 != 6]

    def moved(ref: np.ndarray) -> np.ndarray:
        d = np.abs(action - ref)
        return np.asarray((d[:, joint_cols] > joint_tol).any(1) | (d[:, grip_cols] > grip_tol).any(1))

    m_start, m_end = moved(action[0]), moved(action[-1])
    first = int(np.argmax(m_start)) if m_start.any() else 0
    last = int(n - 1 - np.argmax(m_end[::-1])) if m_end.any() else n - 1
    margin = round(margin_s * fps)
    start, end = max(0, first - margin), min(n, last + 1 + margin)
    if end - start < n * 0.3:  # 异常保护：裁掉太多就不裁
        return 0, n
    return start, end


@dataclass
class Aligned:
    state: np.ndarray          # [T, S] follower
    velocity: np.ndarray       # [T, S] follower 速度
    action: np.ndarray         # [T, A] leader（绝对值；相对化在训练时由 LeRobot 完成）
    leader_full: np.ndarray    # [T, S] leader 按 state 顺序（画图/分析用）
    frame_index: dict[str, np.ndarray]  # 相机 -> 每个网格点用哪一帧
    timestamps: np.ndarray     # [T] 相对起点秒
    trim: tuple[int, int]      # 裁剪前的保留区间
    raw_len: int
    max_age_ms: dict[str, float]


def align(raw: EpisodeRaw, layout: Layout, fps: int, max_age_ms: float = 100.0,
          velocity_window_s: float = 0.1, trim_idle: bool = True,
          idle_joint_tol: float = 0.02, idle_grip_tol: float = 0.05, idle_margin_s: float = 0.5) -> Aligned:
    streams = {**{f"{k[0]}/{k[1]}/{k[2]}": v[0] for k, v in raw.numeric.items()},
               **{f"cam/{k}": v[0] for k, v in raw.cameras.items()}}
    start = max(t[0] for t in streams.values())
    end = min(t[-1] for t in streams.values())
    if end - start < 2e9:
        raise ValueError(f"{raw.path.name}: 公共时间段不足 2 秒")
    step = 1e9 / fps
    # 注意：ns 时间戳 ~1.8e18 超出 float64 精度，必须先算整数偏移再加到 int64 起点上
    grid = start + np.round(np.arange(int((end - start) // step) + 1) * step).astype(np.int64)
    max_age = int(max_age_ms * 1e6)

    def sample(role: str, arms: tuple[str, ...]) -> np.ndarray:
        cols = []
        for arm in arms:
            for kind in ("joint", "gripper"):
                times, values = raw.numeric[(role, arm, kind)]
                cols.append(values[causal_index(times, grid, max_age, f"{raw.path.name} {role}/{arm}/{kind}")])
        return np.concatenate(cols, axis=1)

    def velocity(role: str, arms: tuple[str, ...]) -> np.ndarray:
        cols = []
        for arm in arms:
            for kind in ("joint", "gripper"):
                times, values = raw.numeric[(role, arm, kind)]
                cols.append(causal_velocity(times, values, grid, velocity_window_s))
        return np.concatenate(cols, axis=1)

    state = sample("follower", layout.state_arms)
    vel = velocity("follower", layout.state_arms)
    leader = sample("leader", layout.state_arms)
    action = leader[:, : len(layout.action_names)]
    frames = {cam: causal_index(t, grid, max_age, f"{raw.path.name} 相机 {cam}") for cam, (t, _) in raw.cameras.items()}
    ages = {name: float((grid - t[causal_index(t, grid, max_age, name)]).max() / 1e6) for name, t in streams.items()}

    for arr, who in ((state, "follower"), (leader, "leader")):
        g = arr[:, [i for i in range(arr.shape[1]) if i % 7 == 6]]
        if g.min() < -0.05 or g.max() > 1.05:
            raise ValueError(f"{raw.path.name}: {who} 夹爪超出 [0,1]: [{g.min():.3f}, {g.max():.3f}]")
    state[:, 6::7] = np.clip(state[:, 6::7], 0, 1)
    leader[:, 6::7] = np.clip(leader[:, 6::7], 0, 1)
    action = leader[:, : len(layout.action_names)]

    n = len(grid)
    lo, hi = idle_bounds(action, fps, idle_joint_tol, idle_grip_tol, idle_margin_s) if trim_idle else (0, n)
    sl = slice(lo, hi)
    return Aligned(
        state=state[sl].astype(np.float32), velocity=vel[sl].astype(np.float32),
        action=action[sl].astype(np.float32), leader_full=leader[sl].astype(np.float32),
        frame_index={k: v[sl] for k, v in frames.items()},
        timestamps=((grid[sl] - grid[lo]) / 1e9).astype(np.float64),
        trim=(lo, hi), raw_len=n, max_age_ms=ages,
    )
