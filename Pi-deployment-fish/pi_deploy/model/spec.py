"""Pi training contracts -> device-side model description (no torch / LeRobot)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

CANONICAL_NAMES = tuple(f'{a}_{n}' for a in ('left', 'right')
                        for n in [*[f'joint{i}' for i in range(6)], 'gripper'])
CHECKPOINT_FILES = ('config.json', 'model.safetensors', 'pitrain_contract.json',
                    'policy_preprocessor.json', 'policy_postprocessor.json')
OBS_CONTRACT = ACTION_CONTRACT = 'pitrain_contract.json'
IMAGE_PREFIX = 'observation.images.'


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取 {path}: {exc}') from exc


def checkpoint_directory(path) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / 'pretrained_model').is_dir():
        path = path / 'pretrained_model'
    missing = [n for n in CHECKPOINT_FILES if not (path / n).is_file() or not (path / n).stat().st_size]
    if missing:
        raise FileNotFoundError(f'checkpoint 不完整 {path}: 缺少/空文件 {missing}；复制完整 checkpoint 目录')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        data = _read(path / name)
        steps = data.get('steps')
        if not isinstance(steps, list) or not steps:
            raise ValueError(f'{name}: 缺少 steps')
        for step in steps:
            state_file = step.get('state_file')
            if state_file:
                p = (path / state_file).resolve()
                if not p.is_relative_to(path) or not p.is_file() or not p.stat().st_size:
                    raise FileNotFoundError(f'{name}: 统计文件无效 {state_file}')
        required = 'normalizer_processor' if name.startswith('policy_pre') else 'unnormalizer_processor'
        if not any(s.get('registry_name') == required and s.get('state_file') for s in steps):
            raise ValueError(f'{name}: 缺少 {required} 的统计文件')
    return path


@dataclass(frozen=True)
class StateSegment:
    kind: str
    indices: tuple[int, ...]
    lag_s: float = 0.0

    @property
    def size(self):
        return len(self.indices)


@dataclass(frozen=True)
class ImageInput:
    key: str
    camera: str
    height: int
    width: int


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    fps: float
    horizon: int
    state_dim: int
    state_segments: tuple[StateSegment, ...]
    images: tuple[ImageInput, ...]
    action_dim: int
    action_targets: tuple[int, ...]
    n_action_steps: int
    prompt: str
    contract: dict
    fingerprint: str
    action_offset_frames: int = 0
    relative_dims: tuple[int, ...] = ()  # Runner already returns absolute actions!
    state_min: tuple | None = None
    state_max: tuple | None = None

    @property
    def dt(self):
        return 1 / self.fps

    @property
    def controlled(self):
        return tuple(sorted(self.action_targets))

    @property
    def velocity_segments(self):
        return tuple(s for s in self.state_segments if s.kind == 'velocity')

    def summary(self):
        return [f'输入 state{self.state_dim}: ' + ', '.join(self.contract['state_names']),
                '图像: ' + ', '.join(f'{i.camera} RGB {i.width}×{i.height}' for i in self.images),
                f'输出 {self.horizon}步 × {self.action_dim}维 @ {self.fps:g}Hz；默认执行 {self.n_action_steps}步',
                '服务返回绝对目标（含夹爪滤波），设备端不再加状态',
                '动作: ' + ', '.join(self.contract['action_names']),
                f'任务提示: {self.prompt}', '推理无随机图像增强']


def resolve_spec(path, overrides=None) -> ModelSpec:
    ckpt = checkpoint_directory(path)
    cfg, c = _read(ckpt / 'config.json'), _read(ckpt / 'pitrain_contract.json')
    over = dict(overrides or {})
    if set(over) - {'camera_map', 'velocity_window_s'}:
        raise ValueError(f'未知 model_overrides: {set(over) - {"camera_map", "velocity_window_s"}}')
    if cfg.get('type') != 'pi05' or cfg.get('use_relative_actions') is not True:
        raise ValueError('仅支持 Pi-train-new 的 pi05 相对关节/绝对夹爪 checkpoint')
    if cfg.get('relative_exclude_joints') != ['gripper']:
        raise ValueError('relative_exclude_joints 必须为 [gripper]')
    names, actions = c.get('state_names'), c.get('action_names')
    velocity = c.get('use_velocity')
    if not isinstance(velocity, bool) or not isinstance(names, list) or not isinstance(actions, list):
        raise ValueError('契约缺少 state_names/action_names/use_velocity')
    expected = 28 if velocity else 14
    pos = names[:14]
    if len(names) != expected or len(set(pos)) != 14 or set(pos) != set(CANONICAL_NAMES):
        raise ValueError('state_names 必须由左右 follower 的14维位置及可选14维速度组成')
    if velocity and names[14:] != [n + '_vel' for n in pos]:
        raise ValueError('速度名称/顺序必须与位置一致并使用 _vel 后缀')
    if len(actions) not in (7, 14) or actions != pos[:len(actions)]:
        raise ValueError('action_names 必须与 state 的动作臂前缀一致（LeRobot 相对动作对齐规则）')
    if cfg.get('action_feature_names') is not None and cfg['action_feature_names'] != actions:
        raise ValueError('config action_feature_names 与契约不一致，拒绝同维错臂/错关节权重')
    for processor in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in _read(ckpt / processor)['steps']:
            declared = step.get('config', {}).get('action_names')
            if declared is not None and declared != actions:
                raise ValueError(f'{processor} 的 action_names 与契约不一致')
    for arm in ('left', 'right'):
        selected = set(actions) & {n for n in CANONICAL_NAMES if n.startswith(arm + '_')}
        if selected and len(selected) != 7:
            raise ValueError('动作必须控制完整7维臂')
    if cfg['input_features'].get('observation.state', {}).get('shape') != [expected]:
        raise ValueError('模型 state 维数与契约不一致')
    if cfg['output_features'].get('action', {}).get('shape') != [len(actions)]:
        raise ValueError('模型 action 维数与契约不一致')
    h, n, fps = c.get('chunk_size'), c.get('n_action_steps'), c.get('fps')
    if (type(h) is not int or not 2 <= h <= 500 or type(n) is not int or not 1 <= n <= h
            or not isinstance(fps, (float, int)) or isinstance(fps, bool) or not math.isfinite(fps) or not 1 <= fps <= 120):
        raise ValueError('非法 chunk_size/n_action_steps/fps')
    if cfg.get('chunk_size') != h or cfg.get('n_action_steps') != n:
        raise ValueError('模型和契约的动作块长度/执行步数不一致')
    if not isinstance(c.get('prompt'), str) or not c['prompt'].strip():
        raise ValueError('契约缺少非空任务 prompt')
    cam_map = over.get('camera_map', {})
    if not isinstance(cam_map, dict):
        raise ValueError('camera_map 必须为映射')
    images = []
    for key, feat in cfg['input_features'].items():
        if key == 'observation.state':
            continue
        shape = feat.get('shape')
        if not key.startswith(IMAGE_PREFIX) or feat.get('type') != 'VISUAL' or len(shape or []) != 3 or shape[0] != 3:
            raise ValueError(f'不支持的输入特征 {key}')
        if any(type(v) is not int or not 1 <= v <= 4096 for v in shape):
            raise ValueError(f'非法图像尺寸 {key}')
        camera = cam_map.get(key, key[len(IMAGE_PREFIX):])
        if not isinstance(camera, str) or not camera:
            raise ValueError('相机名必须非空')
        images.append(ImageInput(key, camera, shape[1], shape[2]))
    if not images or len({i.camera for i in images}) != len(images) or set(c.get('image_keys', [])) != {i.key for i in images}:
        raise ValueError('图像契约、模型输入或相机映射不一致')
    if set(cam_map) - {i.key for i in images}:
        raise ValueError('camera_map 包含模型没有的键')
    indices = tuple(CANONICAL_NAMES.index(x) for x in pos)
    segments = [StateSegment('robot', indices)]
    if velocity:
        window = float(over.get('velocity_window_s', .1))
        if not math.isfinite(window) or not .01 <= window <= 1:
            raise ValueError('velocity_window_s 应在 .01..1 秒')
        segments.append(StateSegment('velocity', indices, window))
    digest = hashlib.sha256(json.dumps({'config': cfg, 'contract': c, 'overrides': over}, sort_keys=True).encode()).hexdigest()
    from pi_deploy.model.stats import normalizer_stats
    import numpy as np
    stats = normalizer_stats(ckpt)
    lo = hi = None
    if 'observation.state.min' in stats and 'observation.state.max' in stats:
        amin = np.asarray(stats['observation.state.min']).reshape(-1)
        amax = np.asarray(stats['observation.state.max']).reshape(-1)
        if len(amin) != expected or len(amax) != expected or not np.isfinite(amin).all() or not np.isfinite(amax).all() or (amin > amax).any():
            raise ValueError('输入state统计范围非法')
        robot_min, robot_max = np.zeros(14), np.zeros(14)
        robot_min[list(indices)] = amin[:14]; robot_max[list(indices)] = amax[:14]
        lo, hi = tuple(robot_min), tuple(robot_max)
    return ModelSpec(str(ckpt), float(fps), h, expected, tuple(segments), tuple(images), len(actions),
                     tuple(CANONICAL_NAMES.index(x) for x in actions), n, c['prompt'], c, digest, state_min=lo, state_max=hi)
