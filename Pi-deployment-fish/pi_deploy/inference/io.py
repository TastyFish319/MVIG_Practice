import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取 {path}: {exc}') from exc
