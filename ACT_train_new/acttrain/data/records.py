"""独立读取NEDF3 v2；只需MCAP内嵌schema，不依赖设备或旧工程。"""
from dataclasses import dataclass
from pathlib import Path
import math
from .core import read_json

@dataclass(frozen=True)
class Record:
    timestamp_ns: int
    kind: str
    role: str
    side: str | None = None
    values: tuple = ()
    image: bytes = b""

class Decoder:
    def __init__(self):
        self.cache = {}
    def decode(self, schema, data):
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        if schema is None or schema.encoding != "protobuf":
            raise ValueError("MCAP消息必须包含protobuf schema")
        key = (schema.name, bytes(schema.data))
        if key not in self.cache:
            pool = descriptor_pool.DescriptorPool()
            pending = list(descriptor_pb2.FileDescriptorSet.FromString(schema.data).file)
            while pending:
                remaining = []
                for item in pending:
                    try:
                        pool.Add(item)
                    except Exception:
                        remaining.append(item)
                if len(remaining) == len(pending):
                    raise ValueError("MCAP protobuf schema依赖无法解析")
                pending = remaining
            self.cache[key] = message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))
        return self.cache[key].FromString(data)

def mcap_paths(episode):
    episode = Path(episode).resolve()
    folder = (episode / "data").resolve()
    if not folder.is_relative_to(episode) or not folder.is_dir():
        raise ValueError("MCAP数据目录缺失或越出episode边界")
    index = folder / "mcap_index.json"
    if index.exists():
        entries = read_json(index)
        if not isinstance(entries, list) or not all(isinstance(e, dict) and isinstance(e.get("filename"), str) for e in entries):
            raise ValueError("MCAP分片索引格式错误")
        paths = [(folder / e["filename"]).resolve() for e in entries]
    else:
        paths = sorted(folder.glob("*.mcap"))
    if any(p.suffix != ".mcap" for p in paths) or set(paths) != {p.resolve() for p in folder.glob("*.mcap")}:
        raise ValueError("MCAP索引必须完整覆盖审核目录中的所有mcap分片")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("MCAP分片缺失或重复")
    if any(p.parent != folder.resolve() or not p.is_file() for p in paths):
        raise ValueError("MCAP分片缺失或路径越界")
    return paths

def iter_records(episode):
    from mcap.reader import make_reader
    metadata = read_json(Path(episode) / "metadata.json")
    info = metadata.get("camera_info", {})
    aliases = {"left_wrist": "left", "right_wrist": "right", "left": "left", "right": "right"}
    topics = {f"nmx/hal/camera/{camera}/rgbd": ("camera", aliases[role], None)
              for camera, role in info.items() if isinstance(role, str) and role in aliases}
    if {item[1] for item in topics.values()} != {"left", "right"}:
        raise ValueError("metadata.camera_info必须明确指定左右腕相机")
    for role in ("leader", "follower"):
        for side in ("left", "right"):
            for suffix, kind in (("joint", "joints"), ("hand_state", "gripper")):
                topics[f"nmx/hal/i2rt/{role}/{side}/{suffix}"] = (kind, role, side)
    decoder = Decoder()
    for shard in mcap_paths(episode):
        with shard.open("rb") as stream:
            for schema, channel, message in make_reader(stream).iter_messages(topics=list(topics)):
                kind, role, side = topics[channel.topic]
                obj = decoder.decode(schema, message.data)
                if kind == "camera":
                    image = bytes(obj.rgb.data)
                    if image:
                        yield Record(int(message.log_time), kind, role, image=image)
                else:
                    values = {str(j.name): float(j.pos) for j in obj.joints}
                    names = [f"joint{i}" for i in range(6)] if kind == "joints" else ["gripper"]
                    if any(n not in values or not math.isfinite(values[n]) for n in names):
                        raise ValueError(f"{channel.topic}: 关节缺失或非有限")
                    yield Record(int(message.log_time), kind, role, side, tuple(values[n] for n in names))
