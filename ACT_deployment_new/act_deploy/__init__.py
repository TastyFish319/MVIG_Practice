"""ACT deployment for the I2RT YAM follower arms (self-contained project)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Unified robot coordinate used everywhere in this project (14 values):
#   0-5  left arm joints [rad]   6  left gripper [0 closed .. 1 open]
#   7-12 right arm joints [rad]  13 right gripper [0 .. 1]
ROBOT_DIM = 14
LEFT = tuple(range(0, 7))
RIGHT = tuple(range(7, 14))
JOINT_IDX = tuple(range(0, 6)) + tuple(range(7, 13))
GRIPPER_IDX = (6, 13)
ROBOT_NAMES = (
    [f"L.j{i}" for i in range(6)] + ["L.grip"] + [f"R.j{i}" for i in range(6)] + ["R.grip"]
)
