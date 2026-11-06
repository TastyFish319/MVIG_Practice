"""学习率调度：所有模式都先线性 warmup，再按所选方式变化。

    lr ▲
       |   cosine:  ___
       |          /    `--..__
       |         /            ``--.___  (降到 初始lr × cosine_final_ratio)
       |   steplr:  ____
       |          /    |____
       |         /          |____      (每 step_size 步乘以 gamma)
       |   constant: ____________
       +--------------------------------▶ step
         warmup
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def lr_factor(step: int, cfg) -> float:
    warm = max(0, cfg.warmup_steps)
    if warm and step < warm:
        return (step + 1) / (warm + 1)
    if cfg.scheduler == "constant":
        return 1.0
    if cfg.scheduler == "steplr":
        return cfg.steplr_gamma ** ((step - warm) // max(1, cfg.steplr_step_size))
    progress = min(1.0, (step - warm) / max(1, cfg.steps - warm))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.cosine_final_ratio + (1 - cfg.cosine_final_ratio) * cosine


def build_scheduler(optimizer, cfg) -> LambdaLR:
    return LambdaLR(optimizer, lambda s: lr_factor(s, cfg))
