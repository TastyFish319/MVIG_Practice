"""读取 I2RT NEDF3 格式的原始 episode（MCAP 容器 + 内嵌 protobuf schema）。

一条 episode 目录::

    episode_0000/
      metadata.json          # 相机角色映射、契约版本等
      episode_stats.json     # 录制是否丢包/出错
      data/mcap_index.json   # 分片顺序
      data/*.mcap            # 真正的数据

本模块只负责"把原始消息按 topic 读成时间序列"，不做任何对齐/重采样。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_CONTRACTS = {"i2rt-nedf3-v2"}  # v1 的 JPEG 颜色通道是反的，不支持
ROLES = ("leader", "follower")
SIDES = ("left", "right")
JOINT_NAMES = tuple(f"joint{i}" for i in range(6))


def joint_topic(role: str, side: str) -> str:
    return f"nmx/hal/i2rt/{role}/{side}/joint"


def hand_topic(role: str, side: str) -> str:
    return f"nmx/hal/i2rt/{role}/{side}/hand_state"


@dataclass
class EpisodeRaw:
    """一条 episode 的原始时间序列（未对齐）。"""

    path: Path
    metadata: dict
    # key 例如 ("follower", "right", "joint") -> (times[N] int64 ns, values[N, 6])
    numeric: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # key 例如 "right_wrist" -> (times[N] int64 ns, [jpeg bytes, ...])
    cameras: dict[str, tuple[np.ndarray, list[bytes]]] = field(default_factory=dict)


class _ProtoDecoder:
    """MCAP 里每个 channel 自带 protobuf 描述符，这里按描述符动态生成消息类。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, bytes], Any] = {}

    def decode(self, schema: Any, payload: bytes) -> Any:
        key = (str(schema.name), bytes(schema.data))
        cls = self._cache.get(key)
        if cls is None:
            cls = self._build(schema.name, schema.data)
            self._cache[key] = cls
        return cls.FromString(payload)

    @staticmethod
    def _build(name: str, data: bytes) -> Any:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

        fds = descriptor_pb2.FileDescriptorSet.FromString(data)
        pool = descriptor_pool.DescriptorPool()
        pending = list(fds.file)
        while pending:  # 依赖顺序未知，反复尝试直到全部加入
            remaining = []
            for desc in pending:
                try:
                    pool.Add(desc)
                except Exception:  # noqa: BLE001 - 依赖还没加入
                    remaining.append(desc)
            if len(remaining) == len(pending):
                raise ValueError(f"无法解析 protobuf 描述符: {[d.name for d in remaining]}")
            pending = remaining
        return message_factory.GetMessageClass(pool.FindMessageTypeByName(str(name)))


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 文件 {path}: {exc}") from exc


def mcap_shards(episode: Path) -> list[Path]:
    data_dir = episode / "data"
    index = data_dir / "mcap_index.json"
    if index.is_file():
        names = [entry["filename"] for entry in read_json(index)]
        paths = [(data_dir / n).resolve() for n in names]
    else:
        paths = sorted(data_dir.glob("*.mcap"))
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"MCAP 分片缺失: {data_dir}")
    return paths


def camera_roles(metadata: dict) -> dict[str, str]:
    """metadata.camera_info: {"camera_0": "right_wrist", ...} -> {topic: role}。"""
    info = metadata.get("camera_info") or {}
    return {f"nmx/hal/camera/{cam}/rgbd": role for cam, role in info.items()}


def check_episode(episode: Path) -> dict:
    """基础健康检查，返回 metadata。"""
    metadata = read_json(episode / "metadata.json")
    version = metadata.get("extra", {}).get("contract_version", metadata.get("contract_version"))
    if version not in SUPPORTED_CONTRACTS:
        raise ValueError(f"{episode.name}: 不支持的数据契约版本 {version!r}")
    stats_path = episode / "episode_stats.json"
    if stats_path.is_file():
        stats = read_json(stats_path)
        if stats.get("writer_error") or stats.get("dropped_batches", 0):
            raise ValueError(f"{episode.name}: 录制时出错或丢包 {stats.get('writer_error')}")
    return metadata


def read_episode(episode: Path, cameras: tuple[str, ...], roles=ROLES, sides=SIDES) -> EpisodeRaw:
    """读取一条 episode。

    cameras: 需要的相机角色，例如 ("right_wrist", "left_wrist")。
    """
    from mcap.reader import make_reader

    episode = Path(episode)
    metadata = check_episode(episode)
    cam_topics = {t: r for t, r in camera_roles(metadata).items() if r in cameras}
    missing = set(cameras) - set(cam_topics.values())
    if missing:
        raise ValueError(f"{episode.name}: metadata 中找不到相机 {missing}")
    numeric_topics = {}
    for role in roles:
        for side in sides:
            numeric_topics[joint_topic(role, side)] = (role, side, "joint")
            numeric_topics[hand_topic(role, side)] = (role, side, "gripper")

    decoder = _ProtoDecoder()
    num_buf: dict[tuple, list] = {k: [] for k in numeric_topics.values()}
    cam_buf: dict[str, list] = {r: [] for r in cam_topics.values()}
    wanted = list(numeric_topics) + list(cam_topics)
    for shard in mcap_shards(episode):
        with shard.open("rb") as stream:
            for schema, channel, message in make_reader(stream).iter_messages(topics=wanted):
                topic = channel.topic
                msg = decoder.decode(schema, message.data)
                t = int(message.log_time)
                if topic in cam_topics:
                    data = bytes(msg.rgb.data)
                    if data:
                        cam_buf[cam_topics[topic]].append((t, data))
                    continue
                key = numeric_topics[topic]
                values = {str(j.name): float(j.pos) for j in msg.joints}
                if key[2] == "joint":
                    row = [values[n] for n in JOINT_NAMES]
                else:
                    row = [values["gripper"]]
                if not all(math.isfinite(v) for v in row):
                    raise ValueError(f"{episode.name}: {topic} 出现非有限数值")
                num_buf[key].append((t, row))

    raw = EpisodeRaw(path=episode, metadata=metadata)
    for key, rows in num_buf.items():
        if not rows:
            raise ValueError(f"{episode.name}: 缺少数据流 {key}")
        rows.sort(key=lambda r: r[0])
        raw.numeric[key] = (np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.float64))
    for role, rows in cam_buf.items():
        if not rows:
            raise ValueError(f"{episode.name}: 缺少相机 {role}")
        rows.sort(key=lambda r: r[0])
        raw.cameras[role] = (np.array([r[0] for r in rows], np.int64), [r[1] for r in rows])
    return raw
