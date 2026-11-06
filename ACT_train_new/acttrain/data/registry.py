"""数据集发现和只读旧数据引用。"""
from pathlib import Path
from .core import read_json, file_hash, write_json

def check_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("lerobot_version") != "0.4.4" or
        manifest.get("action_semantics") != "future_leader_right_absolute" or
        manifest.get("state_source") != "follower" or manifest.get("action_source") != "leader" or
        (manifest.get("fps"), manifest.get("action_horizon"), manifest.get("action_offset_frames"), manifest.get("state_dim"), manifest.get("action_dim")) != (20,32,1,14,7)):
        raise ValueError("数据集不符合ACT leader标签/state14/20Hz/32步契约")
    if read_json(root / "validation.json").get("status") != "passed":
        raise ValueError("数据集尚未通过官方读取验证")
    episodes = manifest.get("episodes", [])
    names = [e["source_relative"] for e in episodes]
    if len(set(names)) != len(names) or not episodes:
        raise ValueError("manifest episode为空或重复")
    for split in ("train", "val"):
        items = [e for e in episodes if e["split"] == split]
        if not items or [e["dataset_episode_index"] for e in items] != list(range(len(items))):
            raise ValueError("manifest划分或episode索引不连续")
        for e in items:
            trace = (root / e["trace"]).resolve()
            if not trace.is_relative_to(root) or not trace.is_file() or e["rows"] < 32 or manifest["split"].get(e["source_relative"]) != split:
                raise ValueError("manifest trace路径、长度或split不一致")
    return manifest

def resolve_dataset(path):
    path = Path(path).expanduser().resolve()
    reference = path / "dataset_reference.json"
    if reference.is_file():
        value = read_json(reference)
        source = Path(value["path"]).expanduser()
        if not source.is_absolute():
            source = path / source
        source = source.resolve()
        if file_hash(source / "manifest.json") != value["manifest_sha256"]:
            raise ValueError("旧数据引用的manifest已变化")
        path = source
    check_manifest(path)
    return path

def discover(folder):
    found = []
    for item in sorted(Path(folder).iterdir()):
        if not item.is_dir() or item.name.startswith('.') or item.name.endswith('.partial'):
            continue
        try:
            target = resolve_dataset(item)
            found.append((item, target, check_manifest(target)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found

def register(source, output):
    source = resolve_dataset(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset_reference.json", {"schema_version":1, "path":str(source),
               "manifest_sha256":file_hash(source / "manifest.json")})
    return output
