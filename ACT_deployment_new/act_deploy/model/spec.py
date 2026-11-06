"""ModelSpec: what a checkpoint expects as input and what its output means.

Derived ONLY from files inside ``pretrained_model/`` (pure JSON, no torch), so both the
model server (Py3.12) and the robot side (Py3.10) compute the same spec:

    config.json                      input/output feature shapes, chunk_size
    act_observation_contract.json    (optional, v3) extra velocity features in the state
    act_action_contract.json         (optional, v2+) relative joints / absolute gripper, fps, offset

Optional per-task ``model_overrides`` (configs/tasks.yaml) are applied last, so a future model
with a different layout can be described in config instead of code.

All robot-side indices refer to the unified 14-value robot state (see act_deploy/__init__.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

CHECKPOINT_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json",
                    "policy_postprocessor.json")
OBS_CONTRACT = "act_observation_contract.json"
ACTION_CONTRACT = "act_action_contract.json"
IMAGE_PREFIX = "observation.images."


@dataclass(frozen=True)
class StateSegment:
    """One piece of the model's observation.state vector.

    kind="robot":    robot14[indices] at the observation time
    kind="velocity": (robot14(t)[indices] - robot14(t - lag_s)[indices]) / lag_s
                     zero while the episode is younger than lag_s (training: "first_two_zero")
    """

    kind: str
    indices: tuple[int, ...]
    lag_s: float = 0.0

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ImageInput:
    key: str  # model feature name, e.g. observation.images.left_wrist
    camera: str  # camera name in configs/deploy.yaml, e.g. left_wrist
    height: int
    width: int


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    fps: float
    horizon: int
    action_offset_frames: int  # chunk step k is the target for t_obs + (k + offset) / fps
    state_dim: int
    state_segments: tuple[StateSegment, ...]
    images: tuple[ImageInput, ...]
    action_dim: int
    action_targets: tuple[int, ...]  # robot14 index driven by each output dim
    relative_dims: tuple[int, ...]  # output dims that are deltas w.r.t. the observed state
    # demonstration range of the robot state (min/max per robot14 index; None = unknown)
    state_min: tuple[float, ...] | None = None
    state_max: tuple[float, ...] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def controlled(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.action_targets)))

    @property
    def velocity_segments(self) -> tuple[StateSegment, ...]:
        return tuple(s for s in self.state_segments if s.kind == "velocity")

    def summary(self) -> list[str]:
        """Human readable lines (Chinese) for the console."""
        from act_deploy import ROBOT_NAMES

        def names(idx):
            return ",".join(ROBOT_NAMES[i] for i in idx)

        parts = []
        for s in self.state_segments:
            if s.kind == "robot":
                parts.append(f"状态[{names(s.indices)}]")
            else:
                parts.append(f"速度(前{s.lag_s * 1000:.0f}ms)[{names(s.indices)}]")
        rel = [self.action_targets[d] for d in self.relative_dims]
        absd = [self.action_targets[d] for d in range(self.action_dim) if d not in self.relative_dims]
        lines = [
            f"输入 state{self.state_dim} = " + " + ".join(parts),
            "输入图像 = " + ", ".join(f"{i.camera}({i.height}x{i.width})" for i in self.images),
            f"输出 {self.horizon}步 x {self.action_dim}维 @ {self.fps:g}Hz "
            f"(= {self.horizon / self.fps:.2f}s, 第0步对应 +{self.action_offset_frames / self.fps * 1000:.0f}ms)",
            f"  增量(相对观测): {names(rel) or '无'}",
            f"  绝对值: {names(absd) or '无'}",
        ]
        held = [i for i in range(14) if i not in self.action_targets]
        if held:
            lines.append(f"  模型不控制(保持不动): {names(held)}")
        return lines + list(self.notes)


def checkpoint_directory(path) -> Path:
    """Accept either .../<run> or .../<run>/pretrained_model."""
    path = Path(path).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    missing = [n for n in CHECKPOINT_FILES if not (path / n).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint 不完整 {path}: 缺少 {missing}（请复制完整的 pretrained_model 目录）")
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors(path: Path, keys=None) -> dict:
    """Minimal safetensors reader (numpy only; the format is 8-byte header size + JSON + raw)."""
    import numpy as np

    dtypes = {"F32": np.float32, "F64": np.float64, "F16": np.float16, "I64": np.int64, "I32": np.int32}
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


def _default_targets(action_dim: int, arm: str = "right") -> tuple[int, ...]:
    if action_dim == 14:
        return tuple(range(14))
    if action_dim == 7:
        return tuple(range(7, 14)) if arm == "right" else tuple(range(0, 7))
    raise ValueError(f"无法自动推断 {action_dim} 维动作对应的关节，请在 tasks.yaml 的 model_overrides.action_targets 指定")


def resolve_spec(checkpoint, overrides: dict | None = None) -> ModelSpec:
    ckpt = checkpoint_directory(checkpoint)
    cfg = _read(ckpt / "config.json")
    notes: list[str] = []
    if cfg.get("type") != "act":
        raise ValueError(f"只支持 ACT checkpoint，读到 type={cfg.get('type')!r}")
    if int(cfg.get("n_obs_steps", 1)) != 1:
        raise ValueError("只支持 n_obs_steps=1")
    feats = cfg["input_features"]
    state_dim = int(feats["observation.state"]["shape"][0])
    images = []
    for key, value in feats.items():
        if key.startswith(IMAGE_PREFIX):
            c, h, w = (int(v) for v in value["shape"])
            if c != 3:
                raise ValueError(f"{key}: 只支持 RGB 三通道")
            images.append(ImageInput(key, key[len(IMAGE_PREFIX):], h, w))
    action_dim = int(cfg["output_features"]["action"]["shape"][0])
    horizon = int(cfg["chunk_size"])

    # ---- observation layout
    segments: list[StateSegment] = []
    obs_c = _read(ckpt / OBS_CONTRACT) if (ckpt / OBS_CONTRACT).is_file() else None
    state_min = state_max = None
    fps = None
    if obs_c is not None:
        base = int(obs_c["base_state_dim"])
        if base != 14:
            raise ValueError(f"观测契约 base_state_dim={base}，本工程只支持 14 维机器人状态")
        segments.append(StateSegment("robot", tuple(range(14))))
        if "velocity_slice" in obs_c:
            a, b = (int(v) for v in obs_c["velocity_slice"])
            lag = float(obs_c.get("dt_seconds") or int(obs_c["lag_frames"]) / float(obs_c["fps"]))
            if obs_c.get("episode_start", "first_two_zero") != "first_two_zero":
                raise ValueError(f"不支持的速度起始规则 {obs_c.get('episode_start')!r}")
            segments.append(StateSegment("velocity", tuple(range(a, b)), lag))
        fps = float(obs_c.get("fps")) if obs_c.get("fps") else None
        stats = obs_c.get("state_stats") or {}
        if "min" in stats and "max" in stats:
            state_min = tuple(float(v) for v in stats["min"][:14])
            state_max = tuple(float(v) for v in stats["max"][:14])
    else:
        if state_dim != 14:
            raise ValueError(f"state{state_dim} 但没有观测契约文件 {OBS_CONTRACT}，无法知道状态如何拼接")
        segments.append(StateSegment("robot", tuple(range(14))))
        notes.append("说明: 无观测契约（v2 及更早的模型），状态 = state14 左7+右7，不含速度")

    # demo range + dimension cross-check from the saved normalizer (works without any contract)
    stats = normalizer_stats(ckpt)
    for key, dim in (("observation.state", state_dim), ("action", action_dim)):
        for stat in ("mean", "std"):
            arr = stats.get(f"{key}.{stat}")
            if arr is not None and arr.size != dim:
                raise ValueError(f"归一化统计 {key}.{stat} 有 {arr.size} 维，但 config.json 写的是 {dim} 维")
    if state_min is None and "observation.state.min" in stats and "observation.state.max" in stats:
        state_min = tuple(float(v) for v in stats["observation.state.min"].reshape(-1)[:14])
        state_max = tuple(float(v) for v in stats["observation.state.max"].reshape(-1)[:14])

    # ---- action semantics
    act_c = _read(ckpt / ACTION_CONTRACT) if (ckpt / ACTION_CONTRACT).is_file() else None
    offset = 1
    if act_c is not None:
        sem = act_c.get("action_semantics")
        if sem != "right_joints_relative_to_observation":
            raise ValueError(f"不支持的动作语义 {sem!r}")
        targets = _default_targets(action_dim, "right")
        lo, hi = (int(v) for v in act_c["joint_state_slice"])
        relative = tuple(d for d, t in enumerate(targets) if lo <= t < hi)
        if act_c.get("gripper", "absolute") != "absolute":
            raise ValueError("只支持绝对夹爪")
        if int(act_c.get("horizon", horizon)) != horizon:
            raise ValueError("动作契约 horizon 与 config.chunk_size 不一致")
        fps = float(act_c.get("fps", fps or 20))
        offset = int(act_c.get("action_offset_frames", 1))
        if obs_c is not None and act_c.get("dataset_manifest_sha256") != obs_c.get("dataset_manifest_sha256"):
            raise ValueError("观测契约与动作契约来自不同数据集，checkpoint 可能被拼错")
    else:
        targets = _default_targets(action_dim, "right")
        relative = ()
        notes.append("注意: 没有动作契约，按绝对关节角处理（请确认这是绝对动作模型）")
    if fps is None:
        fps = 20.0
        notes.append("注意: 契约里没有 fps，按 20Hz 处理")

    spec = ModelSpec(
        checkpoint=str(ckpt), fps=fps, horizon=horizon, action_offset_frames=offset,
        state_dim=state_dim, state_segments=tuple(segments), images=tuple(images),
        action_dim=action_dim, action_targets=tuple(targets), relative_dims=tuple(relative),
        state_min=state_min, state_max=state_max, notes=tuple(notes),
    )
    spec = apply_overrides(spec, overrides or {})
    validate_spec(spec)
    return spec


OVERRIDE_KEYS = {"fps", "action_offset_frames", "action_targets", "relative_dims", "camera_map"}


def apply_overrides(spec: ModelSpec, ov: dict) -> ModelSpec:
    unknown = set(ov) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"model_overrides 有未知键 {sorted(unknown)}，可用: {sorted(OVERRIDE_KEYS)}")
    changes: dict = {}
    for key in ("fps",):
        if key in ov:
            changes[key] = float(ov[key])
    if "action_offset_frames" in ov:
        changes["action_offset_frames"] = int(ov["action_offset_frames"])
    if "action_targets" in ov:
        changes["action_targets"] = tuple(int(v) for v in ov["action_targets"])
    if "relative_dims" in ov:
        changes["relative_dims"] = tuple(int(v) for v in ov["relative_dims"])
    if "camera_map" in ov:
        cmap = dict(ov["camera_map"])
        changes["images"] = tuple(replace(i, camera=cmap.get(i.key, i.camera)) for i in spec.images)
    if changes:
        changes["notes"] = spec.notes + (f"已应用 tasks.yaml model_overrides: {sorted(ov)}",)
    return replace(spec, **changes)


def validate_spec(spec: ModelSpec) -> None:
    if sum(s.size for s in spec.state_segments) != spec.state_dim:
        raise ValueError(f"状态拼接总长 {sum(s.size for s in spec.state_segments)} != state_dim {spec.state_dim}")
    if len(spec.action_targets) != spec.action_dim or len(set(spec.action_targets)) != spec.action_dim:
        raise ValueError("action_targets 长度必须等于 action_dim 且不重复")
    if any(not 0 <= t < 14 for t in spec.action_targets):
        raise ValueError("action_targets 必须是 0..13")
    if any(not 0 <= d < spec.action_dim for d in spec.relative_dims):
        raise ValueError("relative_dims 越界")
    if any(spec.action_targets[d] in (6, 13) for d in spec.relative_dims):
        raise ValueError("夹爪维度不能设为增量")
    if not spec.images:
        raise ValueError("模型没有图像输入")
    if spec.fps <= 0 or spec.horizon < 2:
        raise ValueError("fps / horizon 非法")
