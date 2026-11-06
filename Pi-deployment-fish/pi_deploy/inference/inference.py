"""推理封装：加载 checkpoint，输入观测，输出绝对动作 chunk（并对夹爪做软下限滤波）。

评测和将来的真机部署都用这个类，保证两边处理完全一致::

    runner = Pi05Runner("runs/xxx/checkpoints/step_012000", tokenizer="pretrained/paligemma_tokenizer")
    out = runner.predict(batch)          # batch: LeRobot 格式的 dict（state / velocity / images / task）
    out["raw"]       # [B, chunk, A] 模型原始输出（绝对值）
    out["filtered"]  # [B, chunk, A] 夹爪经过软下限滤波后的指令（部署时发这个）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import compose_batch
from .gripper import GripperFilter
from .policy_factory import load_checkpoint, predict_chunk


class Pi05Runner:
    def __init__(self, ckpt_dir: Path, tokenizer: Path, device: str = "cuda", gripper_filter: dict | None = None):
        self.policy, self.pre, self.post, self.contract = load_checkpoint(Path(ckpt_dir), Path(tokenizer), device)
        self.device = device
        gf = gripper_filter or self.contract.get("gripper_filter") or {}
        self.filter = GripperFilter(**gf) if gf else GripperFilter()
        names = self.contract["action_names"]
        self.grip_cols = [i for i, n in enumerate(names) if n.endswith("gripper")]
        self.use_velocity = bool(self.contract["use_velocity"])

    @torch.no_grad()
    def predict(self, batch: dict, seed: int | None = 0) -> dict:
        batch = compose_batch(batch, self.use_velocity)
        if "task" not in batch:
            b = batch["observation.state"].shape[0]
            batch["task"] = [self.contract["prompt"]] * b
        proc = self.pre(batch)
        b = proc["observation.state"].shape[0]
        noise = None
        if seed is not None:  # 固定噪声：同一输入每次输出相同，便于比较不同 checkpoint
            gen = torch.Generator().manual_seed(seed)
            pc = self.policy.config
            noise = torch.randn((b, pc.chunk_size, pc.max_action_dim), generator=gen).to(self.device)
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            pred = predict_chunk(self.policy, proc, noise)
        raw = self.post(pred).float().cpu().numpy()
        filtered = raw.copy()
        for c in self.grip_cols:
            filtered[..., c] = self.filter(raw[..., c])
        return {"raw": raw, "filtered": filtered.astype(np.float32)}
