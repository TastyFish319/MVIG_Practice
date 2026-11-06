"""夹爪"软下限"滤波 + 从数据中自动估计参数。

问题：
    采集时操作员把主臂夹爪捏到 0.0~0.2，而从臂被积木挡住停在 ~0.30。
    模型学的是主臂值 → 部署时会命令从臂去 0.0~0.2 → 位置误差大 → 夹持力过猛。

方案（只作用在推理输出，训练照常学主臂值）：

    输出 ▲
     1.0 |                    /
         |                  /   ← 高于 knee：原样输出（张开/松开动作不受影响）
    knee |- - - - - - - - +
         |            .-'
   floor |- - - -.--''        ← 低于 knee：平滑地"趴"到 floor 附近，永远不低于 floor
         |___.--'
         +-----------------------▶ 模型输出
         0   floor   knee    1.0

    g >= knee : out = g
    g <  knee : out = floor + (knee - floor) * exp((g - knee) / (knee - floor))

    - 在 knee 处数值与斜率都连续（不会突跳）
    - 单调递增：模型"越想夹紧"，输出越接近 floor，但不会更低
    - floor 设在"积木宽度 - 挤压余量"，保证还有适度夹持力
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class GripperFilter:
    knee: float = 0.35
    floor: float = 0.22
    enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.floor < self.knee <= 1:
            raise ValueError(f"需要 0 <= floor < knee <= 1，当前 floor={self.floor}, knee={self.knee}")

    def __call__(self, g):
        g = np.asarray(g, dtype=np.float64)
        if not self.enabled:
            return np.clip(g, 0.0, 1.0)
        width = self.knee - self.floor
        soft = self.floor + width * np.exp(np.minimum(g - self.knee, 0.0) / width)
        return np.clip(np.where(g >= self.knee, g, soft), 0.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_profile(leader_grip: list[np.ndarray], follower_grip: list[np.ndarray],
                     close_threshold: float = 0.5, stall_gap: float = 0.05,
                     squeeze_margin: float = 0.08, knee_margin: float = 0.05) -> dict:
    """从训练集估计"抓住物体时"从臂停在哪里，并给出滤波参数建议。

    抓持帧定义：主臂夹爪 < close_threshold 且 从臂比主臂大 stall_gap 以上（被物体挡住）。
    """
    lead = np.concatenate(leader_grip)
    foll = np.concatenate(follower_grip)
    hold = (lead < close_threshold) & (foll - lead > stall_gap)
    if hold.sum() < 10:
        return {"hold_frames": int(hold.sum()), "suggested": GripperFilter().to_dict(),
                "note": "抓持帧太少，使用默认参数"}
    hold_width = float(np.median(foll[hold]))
    floor = float(np.clip(hold_width - squeeze_margin, 0.02, 0.9))
    knee = float(np.clip(hold_width + knee_margin, floor + 0.05, 1.0))
    return {
        "hold_frames": int(hold.sum()),
        "follower_hold_median": hold_width,
        "follower_hold_p05_p95": [float(np.percentile(foll[hold], 5)), float(np.percentile(foll[hold], 95))],
        "leader_hold_median": float(np.median(lead[hold])),
        "leader_hold_p05_p95": [float(np.percentile(lead[hold], 5)), float(np.percentile(lead[hold], 95))],
        "squeeze_margin": squeeze_margin,
        "suggested": GripperFilter(knee=round(knee, 3), floor=round(floor, 3)).to_dict(),
    }
