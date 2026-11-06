"""训练/评测时的数据入口：打开 LeRobot 数据集、组装 state（是否带速度）、计算归一化统计。

数据集里存的是::

    observation.state      follower 位置（14 维，动作臂在前）
    observation.velocity   follower 速度（14 维）
    observation.leader_state  主臂（只用于画图）
    action                 leader 绝对位置（7 或 14 维）

模型实际看到的 state:
    不带速度：observation.state                     (14)
    带速度  ：concat(observation.state, velocity)    (28)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .nedf3 import read_json
from .preprocess import META_FILE

STATE, VELOCITY, ACTION, LEADER = "observation.state", "observation.velocity", "action", "observation.leader_state"
EXTRA_KEYS = (VELOCITY, LEADER)
QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")


def load_meta(dataset_root: Path) -> dict:
    return read_json(Path(dataset_root) / META_FILE)


def open_split(dataset_root: Path, split: str, fps: int, chunk_size: int, image_transforms=None):
    """打开 train/ 或 val/。delta_timestamps 让每个样本带上未来 chunk_size 步的动作。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        f"pitrain/{Path(dataset_root).name}_{split}", root=Path(dataset_root) / split,
        delta_timestamps={ACTION: [i / fps for i in range(chunk_size)]},
        image_transforms=image_transforms,
    )


def image_transforms(enable: bool):
    if not enable:
        return None
    from lerobot.transforms import ImageTransforms, ImageTransformsConfig

    cfg = ImageTransformsConfig(enable=True, max_num_transforms=3, random_order=True)
    # 腕部相机视角随手臂运动本来就在变；仿射只给很小的幅度
    cfg.tfs["affine"].kwargs = {"degrees": (-3.0, 3.0), "translate": (0.03, 0.03)}
    return ImageTransforms(cfg)


def compose_batch(batch: dict, use_velocity: bool) -> dict:
    """把速度拼进 state，并移除模型不需要的键。"""
    batch = dict(batch)
    if use_velocity:
        batch[STATE] = torch.cat([batch[STATE], batch[VELOCITY]], dim=-1)
    for k in EXTRA_KEYS:
        batch.pop(k, None)
    return batch


def _as_np(stats: dict) -> dict:
    return {k: np.asarray(v, dtype=np.float64) for k, v in stats.items()}


def widen_quantiles(stats: dict, min_span: float) -> dict:
    """保证每一维 q99 - q01 >= min_span（以 q50 为中心向两侧扩）。

    为什么：pi0.5 用分位数把数据映射到 [-1,1]：x' = 2(x-q01)/(q99-q01) - 1。
    左臂几乎静止时 q99-q01 ≈ 0.001，传感器 0.001 的抖动会被放大成 ±1 甚至 ±100，
    模型看到的就全是噪声。设一个最小跨度可以避免这个问题。
    """
    s = _as_np(stats)
    span = s["q99"] - s["q01"]
    need = np.maximum(0.0, min_span - span) / 2
    s["q01"] = s["q01"] - need
    s["q99"] = s["q99"] + need
    if "q10" in s:
        s["q10"] = np.minimum(s["q10"], s["q01"] + 0.1 * (s["q99"] - s["q01"]))
        s["q90"] = np.maximum(s["q90"], s["q99"] - 0.1 * (s["q99"] - s["q01"]))
    return s


def concat_stats(a: dict, b: dict) -> dict:
    out = {}
    for k, va in a.items():
        if k == "count":
            out[k] = np.asarray(va)
        elif k in b:
            out[k] = np.concatenate([np.asarray(va).reshape(-1), np.asarray(b[k]).reshape(-1)])
    return out


def build_policy_stats(train_ds, meta: dict, cfg) -> dict:
    """组装给 pi0.5 归一化器用的统计量（只来自训练集）。"""
    from lerobot.datasets.compute_stats import compute_relative_action_stats

    raw = train_ds.meta.stats
    state = widen_quantiles(raw[STATE], cfg.min_state_span)
    if cfg.use_velocity:
        state = concat_stats(state, widen_quantiles(raw[VELOCITY], cfg.min_velocity_span))
    # 相对动作统计：遍历训练集所有完整 chunk，计算 (leader[t+k] - follower[t])，夹爪保持绝对
    rel = compute_relative_action_stats(train_ds.hf_dataset, train_ds.meta.features, cfg.chunk_size, ["gripper"])
    action = widen_quantiles(rel, cfg.min_action_span)
    stats = {STATE: state, ACTION: action}
    for key in meta["image_keys"]:
        if key in raw:
            stats[key] = _as_np(raw[key])
    return stats


def policy_features(meta: dict, use_velocity: bool, image_size: int) -> tuple[dict, dict]:
    """返回 (input_features, output_features)，供 PI05Config 使用。"""
    from lerobot.configs import FeatureType, PolicyFeature

    state_dim = len(meta["state_names"]) * (2 if use_velocity else 1)
    inputs = {}
    # 预训练时相机顺序是 base / left_wrist / right_wrist，这里保持"左腕在前、右腕在后"
    for key in sorted(meta["image_keys"], key=lambda k: (0 if "left" in k else 1, k)):
        inputs[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
    inputs[STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(meta["action_names"]),))}
    return inputs, outputs


def evenly_spaced(n_total: int, n_pick: int) -> list[int]:
    if n_pick >= n_total:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_pick).round().astype(int).tolist()
