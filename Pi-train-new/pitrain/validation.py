"""训练过程中的验证（在每个 checkpoint 间隔执行）。

两类指标，都在"固定的验证子集 + 固定随机数"上计算，所以不同 step 之间可以直接比较：

1. val_loss  —— 与训练 loss 同口径的 flow-matching 损失（噪声 / 时间点用固定种子）。
   便宜，但它衡量的是"去噪一步的误差"，不等于真正执行动作的误差。
2. val_nmae 等 —— 真正跑一遍推理（10 步去噪）得到绝对动作，与主臂真值比较：
   - val_joint_mae_deg_first : 前 n_action_steps 步（真正会被执行的部分）的关节平均误差（度）
   - val_joint_mae_deg_full  : 整个 chunk 的关节平均误差（度）
   - val_gripper_mae_*       : 夹爪误差（0~1）
   - val_nmae                : 每维误差 ÷ 该维动作幅度（(q99-q01)/2）后再平均——无量纲，用来挑 best
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import ACTION, compose_batch, evenly_spaced
from .policy_factory import predict_chunk

PAD = "action_is_pad"


class Validator:
    def __init__(self, val_ds, cfg, meta: dict, accelerator, action_stats: dict):
        self.cfg, self.acc = cfg, accelerator
        n = len(val_ds)
        rank, world = accelerator.process_index, accelerator.num_processes
        loss_idx = evenly_spaced(n, cfg.val_loss_samples)[rank::world]
        infer_idx = evenly_spaced(n, cfg.val_infer_samples)[rank::world]
        bs = max(1, cfg.batch_size_per_gpu)
        kw = {"batch_size": bs, "shuffle": False, "num_workers": min(4, cfg.num_workers)}
        self.loss_loader = DataLoader(Subset(val_ds, loss_idx), **kw)
        self.infer_loader = DataLoader(Subset(val_ds, infer_idx), **kw)
        a_dim = len(meta["action_names"])
        self.grip = np.array([n.endswith("gripper") for n in meta["action_names"]])
        self.scale = np.maximum((np.asarray(action_stats["q99"]) - np.asarray(action_stats["q01"])) / 2, 1e-3)[:a_dim]
        self.a_dim = a_dim

    # ---------- 固定随机数的 flow-matching loss ----------
    def _loss_batch(self, policy, batch, gen) -> tuple[float, int]:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        pc = policy.config
        images, img_masks = policy._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = policy.prepare_action(batch)
        b = actions.shape[0]
        noise = torch.randn(actions.shape, generator=gen).to(actions.device, actions.dtype)
        u = torch.rand(b, generator=gen)
        # 训练时 t ~ Beta(alpha, beta)；beta=1 时可用逆 CDF u^(1/alpha) 精确且可复现地采样
        t = u ** (1.0 / pc.time_sampling_beta_alpha) if pc.time_sampling_beta_beta == 1.0 else torch.rand(b, generator=gen)
        t = (t * pc.time_sampling_scale + pc.time_sampling_offset).to(actions.device, torch.float32)
        losses = policy.model.forward(images, img_masks, tokens, masks, actions, noise, t)[:, :, : self.a_dim]
        return float(losses.float().mean().item()) * b, b

    @torch.no_grad()
    def run(self, policy, preprocessor, postprocessor) -> dict:
        policy.eval()
        dev = self.acc.device
        gen = torch.Generator().manual_seed(1234)
        loss_sum, loss_n = 0.0, 0
        with self.acc.autocast():
            for batch in self.loss_loader:
                batch = preprocessor(compose_batch(batch, self.cfg.use_velocity))
                s, n = self._loss_batch(policy, batch, gen)
                loss_sum, loss_n = loss_sum + s, loss_n + n

        k = self.cfg.n_action_steps
        sums = np.zeros((4, self.a_dim))  # [first 误差和, first 计数, full 误差和, full 计数]
        gen = torch.Generator().manual_seed(5678)
        with self.acc.autocast():
            for batch in self.infer_loader:
                gt = batch[ACTION].float().numpy()               # 绝对动作真值 [B, C, A]
                valid = ~batch[PAD].numpy()                      # 超出 episode 末尾的步是填充的
                proc = preprocessor(compose_batch(batch, self.cfg.use_velocity))
                b = gt.shape[0]
                noise = torch.randn((b, policy.config.chunk_size, policy.config.max_action_dim), generator=gen)
                pred = predict_chunk(policy, proc, noise.to(dev))
                pred = postprocessor(pred).float().cpu().numpy()  # 反归一化 + 加回 state = 绝对动作
                err = np.abs(pred - gt)
                m = valid[..., None].astype(np.float64)
                sums[0] += (err[:, :k] * m[:, :k]).sum((0, 1))
                sums[1] += m[:, :k].sum((0, 1))
                sums[2] += (err * m).sum((0, 1))
                sums[3] += m.sum((0, 1))
        packed = torch.tensor(np.concatenate([[loss_sum, loss_n], sums.ravel()]), device=dev, dtype=torch.float64)
        packed = self.acc.reduce(packed, reduction="sum").cpu().numpy()
        loss_sum, loss_n, sums = packed[0], packed[1], packed[2:].reshape(4, self.a_dim)
        mae_first = sums[0] / np.maximum(sums[1], 1)
        mae_full = sums[2] / np.maximum(sums[3], 1)
        j = ~self.grip
        policy.train()
        return {
            "val_loss": float(loss_sum / max(loss_n, 1)),
            "val_nmae": float((mae_full / self.scale).mean()),
            "val_nmae_first": float((mae_first / self.scale).mean()),
            "val_joint_mae_deg_first": float(np.rad2deg(mae_first[j]).mean()),
            "val_joint_mae_deg_full": float(np.rad2deg(mae_full[j]).mean()),
            "val_gripper_mae_first": float(mae_first[self.grip].mean()) if self.grip.any() else 0.0,
            "val_gripper_mae_full": float(mae_full[self.grip].mean()) if self.grip.any() else 0.0,
        }
