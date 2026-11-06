"""Small statistics reader copied from ACT; never reads the model weight file."""

import json

from pathlib import Path

def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_safetensors(path: Path, keys=None) -> dict:
    """Minimal safetensors reader (numpy only; the format is 8-byte header size + JSON + raw)."""
    import numpy as np

    dtypes = {"F32": np.float32, "F64": np.float64, "F16": np.float16, "I64": np.int64, "I32": np.int32}
    if Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError('统计文件过大，拒绝按统计读取模型权重')
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    out = {}
    for k, v in header.items():
        if k == "__metadata__" or (keys is not None and k not in keys) or v["dtype"] not in dtypes:
            continue
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(raw[8 + n + a:8 + n + b], dtype=dtypes[v["dtype"]]).reshape(v["shape"])
    return out

def normalizer_stats(ckpt: Path) -> dict:
    """Training statistics saved with the preprocessor (observation.state.min/max/mean..., action.*)."""
    pre = _read(ckpt / "policy_preprocessor.json")
    for step in pre.get("steps", []):
        if step.get("registry_name") == "normalizer_processor" and step.get("state_file"):
            path = ckpt / step["state_file"]
            if path.is_file():
                return read_safetensors(path)
    return {}
