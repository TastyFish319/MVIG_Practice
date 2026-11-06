"""I2RT NEDF3-v2读取契约；无需原仓库、设备SDK或Rerun。"""
from pathlib import Path
import json, math
from typing import Any
NEDF3_METADATA_TOPIC = "nmx/nedf/metadata"
NEDF3_CAMERA_TOPICS = {"right":"nmx/hal/camera/camera_0/rgbd", "left":"nmx/hal/camera/camera_1/rgbd", "head":"nmx/hal/camera/camera_2/rgbd"}
CAMERA_TOPIC_INFO = {topic:role for role,topic in NEDF3_CAMERA_TOPICS.items()}
JOINT_TOPIC_INFO = {f"nmx/hal/i2rt/{role}/{side}/joint":(role,side) for role in ("leader","follower") for side in ("left","right")}
HAND_TOPIC_INFO = {f"nmx/hal/i2rt/{role}/{side}/hand_state":(role,side) for role in ("leader","follower") for side in ("left","right")}
NEDF3_RECORD_TOPICS = (NEDF3_METADATA_TOPIC,*CAMERA_TOPIC_INFO,*JOINT_TOPIC_INFO,*HAND_TOPIC_INFO)
JOINT_NAMES = tuple(f"joint{i}" for i in range(6))
class _EmbeddedProtobufDecoder:
    def __init__(self) -> None:
        self._message_classes: dict[tuple[str, bytes], Any] = {}

    def decode(self, schema: Any, payload: bytes) -> Any:
        if schema.encoding != "protobuf":
            raise ValueError(
                f"Unsupported schema encoding {schema.encoding!r} for {schema.name!r}"
            )
        key = (str(schema.name), bytes(schema.data))
        message_class = self._message_classes.get(key)
        if message_class is None:
            message_class = self._build_message_class(schema.name, schema.data)
            self._message_classes[key] = message_class
        return message_class.FromString(payload)

    @staticmethod
    def _build_message_class(message_name: str, descriptor_data: bytes) -> Any:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_data)
        pool = descriptor_pool.DescriptorPool()
        pending = list(descriptor_set.file)
        while pending:
            remaining = []
            for descriptor in pending:
                try:
                    pool.Add(descriptor)
                except Exception:
                    remaining.append(descriptor)
            if len(remaining) == len(pending):
                names = ", ".join(descriptor.name for descriptor in remaining)
                raise ValueError(f"Could not resolve protobuf descriptors: {names}")
            pending = remaining

        descriptor = pool.FindMessageTypeByName(str(message_name))
        factory = message_factory.MessageFactory(pool)
        if hasattr(factory, "GetPrototype"):
            return factory.GetPrototype(descriptor)
        return message_factory.GetMessageClass(descriptor)


def episode_mcap_paths(episode_dir: Path) -> tuple[Path, ...]:
    """Resolve ordered MCAP shards from a finalized NEDF3 episode."""
    episode_dir = episode_dir.expanduser().resolve()
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NEDF3 metadata not found: {metadata_path}")

    data_dir = (episode_dir / "data").resolve()
    if not data_dir.is_relative_to(episode_dir) or not data_dir.is_dir():
        raise ValueError("MCAP数据目录缺失或越出episode边界")
    index_path = data_dir / "mcap_index.json"
    paths: list[Path] = []
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid MCAP index: {index_path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"MCAP index must contain a list: {index_path}")
        for entry in entries:
            filename = entry.get("filename") if isinstance(entry, dict) else None
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"MCAP index contains an invalid filename: {index_path}")
            path = (data_dir / filename).resolve()
            if path.parent != data_dir.resolve():
                raise ValueError(f"MCAP index filename escapes the data directory: {filename}")
            paths.append(path)
    else:
        paths = sorted(data_dir.glob("*.mcap"))

    if len(set(paths)) != len(paths) or set(paths) != {p.resolve() for p in data_dir.glob("*.mcap")}:
        raise ValueError("MCAP索引必须完整且不重复地包含全部mcap分片")
    if any(p.suffix != ".mcap" or p.parent != data_dir for p in paths):
        raise ValueError("MCAP分片路径越界或扩展名错误")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MCAP shard not found: {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"No MCAP shards found under: {data_dir}")
    return tuple(paths)



def _joint_values(decoded: Any, topic: str) -> tuple[float, ...]:
    values = {str(joint.name): float(joint.pos) for joint in decoded.joints}
    missing = [name for name in JOINT_NAMES if name not in values]
    if missing:
        raise ValueError(f"{topic} is missing joints: {missing}")
    result = tuple(values[name] for name in JOINT_NAMES)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{topic} contains a non-finite joint value")
    return result


def _gripper_value(decoded: Any, topic: str) -> float:
    values = {
        str(joint.name): float(joint.pos)
        for joint in decoded.joints
        if str(joint.name) == "gripper"
    }
    if "gripper" not in values or not math.isfinite(values["gripper"]):
        raise ValueError(f"{topic} does not contain a finite gripper value")
    return values["gripper"]

