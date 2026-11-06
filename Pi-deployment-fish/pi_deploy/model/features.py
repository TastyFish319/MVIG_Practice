"""ACT observation/decode interface adapted to Pi; deterministic, numpy-only."""
from __future__ import annotations
import numpy as np
from pi_deploy.model.spec import ModelSpec


def letterbox_rgb(rgb, height, width, jpeg_quality=None):
    """Copied ACT letterbox; Pi size comes from checkpoint. No random augmentation."""
    import cv2
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('图像必须是 uint8 HWC RGB')
    if jpeg_quality:
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise RuntimeError('JPEG 编码失败')
        rgb = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = round(h * scale), round(w * scale)
    out = np.zeros((height, width, 3), np.uint8)
    y, x = (height - nh) // 2, (width - nw) // 2
    out[y:y+nh, x:x+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def build_state(spec: ModelSpec, history, t_obs, episode_start=0):
    robot = history.at(t_obs)
    parts, padded = [], False
    for seg in spec.state_segments:
        if seg.kind == 'robot':
            parts.append(robot[list(seg.indices)])
        else:
            vel, insufficient = history.velocity(t_obs, seg.lag_s)
            parts.append(vel[list(seg.indices)])
            padded |= insufficient
    state = np.concatenate(parts).astype(np.float32)
    if state.shape != (spec.state_dim,) or not np.isfinite(state).all():
        raise ValueError('观测 state 形状/数值非法')
    return state, robot, {'velocity_padded': padded}


def decode_actions(spec, raw, anchor14, hold14):
    """Runner output is already ABSOLUTE, including grippers; no state addition here."""
    raw = np.asarray(raw, np.float64)
    if raw.shape != (spec.horizon, spec.action_dim) or not np.isfinite(raw).all():
        raise ValueError(f'模型输出非法: {raw.shape}')
    out = np.repeat(np.asarray(hold14, float)[None], spec.horizon, axis=0)
    out[:, list(spec.action_targets)] = raw
    out[:, [6, 13]] = np.clip(out[:, [6, 13]], 0, 1)
    return out


def dummy_inputs(spec):
    robot = np.zeros(14, np.float32)
    robot[[6, 13]] = .99
    state = np.concatenate([robot[list(seg.indices)] if seg.kind == 'robot' else np.zeros(seg.size)
                            for seg in spec.state_segments]).astype(np.float32)
    images = {i.key: np.full((i.height, i.width, 3), 128, np.uint8) for i in spec.images}
    return state, images
