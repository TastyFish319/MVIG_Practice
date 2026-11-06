"""Seven-value follower endpoint (6 joints + gripper) used by the follower worker.

Adapted from SuperInference devices/robots/i2rt_teleop/endpoints.py (see docs/vendored.md):
follower only, integrated i2rt_can gripper only, plus a FakeFollowerEndpoint (sim / tests).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .config import FollowerConfig

ARM_DOF = 6
ENDPOINT_DOF = 7


def _vector(value: Any, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{field} must contain exactly {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must contain only finite values")
    return result


class FollowerEndpoint:
    """A YAM arm with its integrated I2RT gripper behind a seven-value API."""

    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self._closed = False

    def get_info(self) -> np.ndarray:
        obs = self.arm.get_observations()
        joints = _vector(obs["joint_pos"], ARM_DOF, "follower joints")
        gripper = _vector(obs.get("gripper_pos"), 1, "I2RT gripper feedback")[0]
        return np.concatenate([joints, [float(np.clip(gripper, 0.0, 1.0))]])

    def command_joint_pos(self, joint_pos: Any) -> None:
        self.arm.command_joint_pos(_vector(joint_pos, ENDPOINT_DOF, "follower command").copy())

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        self.arm.update_kp_kd(_vector(kp, ENDPOINT_DOF, "kp"), _vector(kd, ENDPOINT_DOF, "kd"))

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        del enabled, kp, kd

    def enter_safe_state(self, state: Any) -> None:
        self.command_joint_pos(_vector(state, ENDPOINT_DOF, "follower state"))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.arm.close()  # i2rt: zero torque


def create_follower_endpoint(config: FollowerConfig) -> FollowerEndpoint:
    """Open the CAN bus and enable the motors of one follower arm."""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    if config.gripper.backend != "i2rt_can":
        raise ValueError(f"本工程只支持一体化 i2rt_can 夹爪，配置为 {config.gripper.backend!r}")
    gripper_type = GripperType.from_string_name(config.gripper.i2rt_type)
    return FollowerEndpoint(get_yam_robot(channel=config.can_channel, gripper_type=gripper_type))


class FakeFollowerEndpoint:
    """Hardware-free arm: the state follows the last command with a ~130 ms first-order lag."""

    def __init__(self, initial: Any | None = None, tau_s: float = 0.13) -> None:
        start = np.zeros(ENDPOINT_DOF) if initial is None else _vector(initial, ENDPOINT_DOF, "initial")
        if initial is None:
            start[-1] = 1.0
        self._state = start.copy()
        self._target = start.copy()
        self._tau = tau_s
        self._t = time.monotonic()
        self._lock = threading.Lock()

    def _advance(self) -> None:
        now = time.monotonic()
        a = 1.0 - np.exp(-(now - self._t) / self._tau)
        self._state = self._state + (self._target - self._state) * a
        self._t = now

    def get_info(self) -> np.ndarray:
        with self._lock:
            self._advance()
            return self._state.copy()

    def command_joint_pos(self, joint_pos: Any) -> None:
        with self._lock:
            self._advance()
            self._target = _vector(joint_pos, ENDPOINT_DOF, "follower command").copy()

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        del kp, kd

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        del enabled, kp, kd

    def enter_safe_state(self, state: Any) -> None:
        self.command_joint_pos(state)

    def close(self) -> None:
        pass
