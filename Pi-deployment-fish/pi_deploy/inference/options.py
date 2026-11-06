"""Shared floor override (no torch); keeps training-estimated knee/floor."""
from .gripper import GripperFilter


def gripper_options(contract: dict, mode: str) -> dict:
    if mode not in ('checkpoint', 'on', 'off'):
        raise ValueError(f'未知夹爪 floor 选项 {mode}')
    cfg = dict(contract.get('gripper_filter') or GripperFilter().to_dict())
    if mode != 'checkpoint':
        cfg['enabled'] = mode == 'on'
    return GripperFilter(**cfg).to_dict()
