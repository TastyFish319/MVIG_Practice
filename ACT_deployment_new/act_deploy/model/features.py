"""Robot data <-> model tensors, exactly as in training (numpy + cv2 only).

Training pipeline being reproduced (incubator/act_training/yam_act):
  * camera: RealSense rgb8 -> recorder JPEG q90 (RGB->BGR encode) -> decode BGR
            -> resize keeping aspect (INTER_AREA) -> zero letterbox 256x256 -> RGB
  * state : follower [left7, right7] at the sample time (+ v3 velocity of right7 over 100 ms,
            zero for the first two 20 Hz frames of an episode)
  * action: right joints = obs right joints + delta; right gripper absolute
"""

from __future__ import annotations

import numpy as np

from act_deploy.model.spec import ModelSpec


def letterbox_rgb(rgb: np.ndarray, height: int, width: int, jpeg_quality: int | None = 90) -> np.ndarray:
    """uint8 HWC RGB (any size) -> uint8 (height, width, 3) RGB, as the training converter did."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"图像必须是 uint8 HWC RGB，收到 {rgb.dtype} {rgb.shape}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if jpeg_quality:
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - nh) // 2, (width - nw) // 2
    canvas[y:y + nh, x:x + nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


def build_state(spec: ModelSpec, history, t_obs: float, episode_start: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """observation.state for time t_obs.

    history: control.history.StateHistory (robot14 readings, monotonic seconds)
    Returns (state vector float32, robot14 at t_obs, info).
    info["velocity_padded"] = True when the velocity was zero-padded (episode younger than the lag
    or not enough history), which reproduces the training rule "first two frames zero".
    """
    robot = history.at(t_obs)
    parts = []
    info = {"velocity_padded": False}
    for seg in spec.state_segments:
        idx = list(seg.indices)
        if seg.kind == "robot":
            parts.append(robot[idx])
        elif seg.kind == "velocity":
            t_prev = t_obs - seg.lag_s
            # training: velocity rows 0 and 1 of every episode are exactly zero
            if t_prev < episode_start - 1e-6 or not history.covers(t_prev):
                parts.append(np.zeros(len(idx)))
                info["velocity_padded"] = True
            else:
                parts.append((robot[idx] - history.at(t_prev)[idx]) / seg.lag_s)
        else:
            raise ValueError(f"unknown state segment {seg.kind}")
    state = np.concatenate(parts).astype(np.float32)
    if state.shape != (spec.state_dim,) or not np.isfinite(state).all():
        raise ValueError("构造出的状态向量非法")
    return state, robot, info


def decode_actions(spec: ModelSpec, raw: np.ndarray, anchor14: np.ndarray, hold14: np.ndarray) -> np.ndarray:
    """Model output (horizon, action_dim) -> absolute robot14 targets (horizon, 14).

    anchor14: robot state the observation was built from (for relative dims)
    hold14:   value for robot dims the model does not drive (e.g. the idle left arm)
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (spec.horizon, spec.action_dim) or not np.isfinite(raw).all():
        raise ValueError(f"模型输出形状/数值非法: {raw.shape}")
    out = np.repeat(np.asarray(hold14, dtype=np.float64)[None], spec.horizon, axis=0)
    for d, target in enumerate(spec.action_targets):
        out[:, target] = raw[:, d] + (anchor14[target] if d in spec.relative_dims else 0.0)
    out[:, [6, 13]] = np.clip(out[:, [6, 13]], 0.0, 1.0)
    return out


def dummy_inputs(spec: ModelSpec) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Plausible inputs for warmup: demo-mean-ish state, grey letterboxed images."""
    state = np.zeros(spec.state_dim, np.float32)
    if spec.state_min is not None and spec.state_max is not None:
        mid = (np.asarray(spec.state_min) + np.asarray(spec.state_max)) / 2
        state[:14] = mid
    state[[6, 13]] = 0.99
    images = {}
    for im in spec.images:
        img = np.zeros((im.height, im.width, 3), np.uint8)
        pad = im.height // 8
        img[pad:im.height - pad] = 128
        images[im.key] = img
    return state, images
