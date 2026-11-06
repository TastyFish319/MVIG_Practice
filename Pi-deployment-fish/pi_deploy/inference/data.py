"""Inference-only compose_batch copied from Pi-train-new/pitrain/data.py."""
import torch

STATE, VELOCITY, LEADER = 'observation.state', 'observation.velocity', 'observation.leader_state'


def compose_batch(batch: dict, use_velocity: bool) -> dict:
    batch = dict(batch)
    if use_velocity:
        batch[STATE] = torch.cat([batch[STATE], batch[VELOCITY]], dim=-1)
    for k in (VELOCITY, LEADER):
        batch.pop(k, None)
    return batch
