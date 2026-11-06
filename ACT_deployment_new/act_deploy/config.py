"""Strict loaders for configs/deploy.yaml and configs/tasks.yaml.

Unknown keys, missing keys and out-of-range values raise ConfigError with the exact path, so a
typo never silently falls back to a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from act_deploy import PROJECT_ROOT

DEPLOY_YAML = PROJECT_ROOT / "configs" / "deploy.yaml"
TASKS_YAML = PROJECT_ROOT / "configs" / "tasks.yaml"
HARDWARE_YAML = PROJECT_ROOT / "configs" / "hardware.yaml"


class ConfigError(ValueError):
    pass


def num(lo=-math.inf, hi=math.inf, integer=False):
    def check(v, path):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{path} 应为数字，读到 {v!r}")
        if integer and int(v) != v:
            raise ConfigError(f"{path} 应为整数")
        if not (lo <= v <= hi):
            raise ConfigError(f"{path}={v} 超出范围 [{lo}, {hi}]")
        return int(v) if integer else float(v)
    return check


def choice(*options):
    def check(v, path):
        if v not in options:
            raise ConfigError(f"{path}={v!r}，可选 {options}")
        return v
    return check


def text(v, path):
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{path} 应为非空字符串")
    return v


def boolean(v, path):
    if not isinstance(v, bool):
        raise ConfigError(f"{path} 应为 true/false")
    return v


def vec14(v, path):
    if not isinstance(v, list) or len(v) != 14:
        raise ConfigError(f"{path} 应为 14 个数")
    out = [num()(x, f"{path}[{i}]") for i, x in enumerate(v)]
    if not (0 <= out[6] <= 1 and 0 <= out[13] <= 1):
        raise ConfigError(f"{path}: 夹爪(第7、14个数)必须在 0..1")
    return out


def mapping_of(item_schema):
    def check(v, path):
        if not isinstance(v, dict) or not v:
            raise ConfigError(f"{path} 应为非空映射")
        return {k: validate(item_schema, x, f"{path}.{k}") for k, x in v.items()}
    return check


def any_mapping(v, path):
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ConfigError(f"{path} 应为映射")
    return v


def validate(schema, value, path="<root>"):
    if callable(schema):
        return schema(value, path)
    if not isinstance(value, dict):
        raise ConfigError(f"{path} 应为映射")
    optional = schema.get("__optional__", {})
    keys = {k for k in schema if k != "__optional__"}
    unknown = set(value) - keys - set(optional)
    if unknown:
        raise ConfigError(f"{path}: 未知键 {sorted(unknown)}")
    out = {}
    for k in keys:
        if k not in value:
            raise ConfigError(f"{path}: 缺少键 {k!r}")
        out[k] = validate(schema[k], value[k], f"{path}.{k}")
    for k, sub in optional.items():
        if k in value:
            out[k] = validate(sub, value[k], f"{path}.{k}")
    return out


def preset(v, path):
    if not isinstance(v, dict):
        raise ConfigError(f"{path} 应为映射")
    mode = choice("chunk", "ensemble")(v.get("mode"), f"{path}.mode")
    common = {"label": text, "mode": text, "smoothing_tau_s": num(0, 1)}
    if mode == "chunk":
        schema = {**common, "exec_steps": num(1, 200, integer=True), "blend_s": num(0, 2)}
    else:
        schema = {**common, "rate_hz": num(1, 100), "ensemble_m": num(-5, 5)}
    return validate(schema, v, path)


DEPLOY_SCHEMA = {
    "control": {"rate_hz": num(20, 500)},
    "policy_server": {"port": num(1024, 65535, integer=True), "device": text,
                      "warmup_iterations": num(0, 100, integer=True), "keep_warm_s": num(0, 60),
                      "request_timeout_s": num(0.05, 10), "startup_timeout_s": num(5, 3600)},
    "cameras": {"width": num(1, 4096, integer=True), "height": num(1, 4096, integer=True),
                "fps": num(1, 120, integer=True), "max_frame_age_s": num(0.01, 2),
                "serials": mapping_of(lambda v, p: str(v))},
    "motion": {"max_joint_velocity": num(0.01, 2), "max_joint_acceleration": num(0.01, 10),
               "max_gripper_velocity": num(0.01, 5), "min_duration_s": num(0, 60),
               "max_tracking_error": num(0.05, 2), "settle_tolerance": num(0.001, 0.5),
               "gripper_tolerance": num(0.001, 1), "settle_timeout_s": num(0, 30)},
    "poses": {"zero": vec14, "return_route": choice("via_start", "direct"),
              "emergency_route": choice("via_start", "direct")},
    "safety": {"max_joint_velocity": num(0.05, 5), "max_gripper_velocity": num(0.05, 20),
               "max_deviation": num(0.02, 2), "workspace_margin": num(0, 3)},
    "policy": {"max_consecutive_failures": num(1, 100, integer=True),
               "starvation_timeout_s": num(0.05, 5), "jpeg_quality": num(0, 100, integer=True),
               "live_warmup_iterations": num(0, 50, integer=True), "latency_init_s": num(0, 1)},
    "execution": {"default": text, "presets": mapping_of(preset)},
    "recording": {"enabled": boolean, "save_images": boolean},
    "ui": {"refresh_hz": num(0.5, 30)},
}

def text_list(v, path):
    if not isinstance(v, list) or not v:
        raise ConfigError(f"{path} 应为非空列表")
    return [text(x, f"{path}[{i}]") for i, x in enumerate(v)]


TASK_SCHEMA = {
    "label": text, "enabled": boolean, "checkpoint_dirs": text_list,
    "start_pose": vec14, "max_episode_s": num(1, 3600), "execution": text,
    "__optional__": {"model_overrides": any_mapping},
}


def _read_yaml(path: Path):
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取 {path}: {exc}") from exc


def load_deploy(path: Path = DEPLOY_YAML) -> dict:
    cfg = validate(DEPLOY_SCHEMA, _read_yaml(path), Path(path).name)
    if cfg["execution"]["default"] not in cfg["execution"]["presets"]:
        raise ConfigError("execution.default 不在 presets 里")
    return cfg


@dataclass
class Task:
    name: str
    label: str
    enabled: bool
    checkpoint_dirs: list[Path]
    start_pose: list[float]
    max_episode_s: float
    execution: str
    model_overrides: dict


def resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_tasks(deploy: dict, path: Path = TASKS_YAML) -> dict[str, Task]:
    raw = _read_yaml(path)
    if set(raw) != {"tasks"} or not isinstance(raw["tasks"], dict) or not raw["tasks"]:
        raise ConfigError(f"{path.name}: 顶层只能有 tasks:")
    out = {}
    for name, value in raw["tasks"].items():
        t = validate(TASK_SCHEMA, value, f"tasks.{name}")
        if t["execution"] not in deploy["execution"]["presets"]:
            raise ConfigError(f"tasks.{name}.execution={t['execution']!r} 不在 deploy.yaml 的 presets 里")
        out[name] = Task(name=name, label=t["label"], enabled=t["enabled"],
                         checkpoint_dirs=[resolve_path(p) for p in t["checkpoint_dirs"]],
                         start_pose=t["start_pose"], max_episode_s=t["max_episode_s"],
                         execution=t["execution"], model_overrides=t.get("model_overrides", {}))
    return out
