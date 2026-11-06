"""pi0.5 微调主循环（Accelerate 多卡）。

启动（由 scripts/train.sh / pitrain.pipeline 自动生成命令）::

    torchrun --standalone --nproc-per-node=N -m pitrain.train --config runs/<名称>/train_config.json

每一步::

    取 batch ─▶ (可选)拼速度 ─▶ 预处理器(相对化/归一化/分词) ─▶ pi0.5 前向得 loss
      ─▶ 反向传播(多卡自动平均梯度) ─▶ 梯度裁剪 ─▶ AdamW 更新 ─▶ 学习率调度

每 val_every 步: 验证 ─▶ 若进入 top-k 则保存权重（淘汰最差的那个）─▶ 刷新曲线图
训练结束: 再验证一次 ─▶ 保存 final
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from .data import build_policy_stats, compose_batch, image_transforms, load_meta, open_split
from .hparams import TrainConfig
from .metrics_log import MetricsLogger, plot_curves
from .policy_factory import build_policy_config, build_processors, load_policy, required_token_length
from .validation import Validator

log = logging.getLogger("pitrain")


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


class TopK:
    """维护"最好的 k 个 checkpoint"。指标越小越好。"""

    def __init__(self, ckpt_root: Path, k: int, metric: str):
        self.root, self.k, self.metric = ckpt_root, k, metric
        self.items: list[dict] = []

    def would_keep(self, value: float) -> bool:
        return len(self.items) < self.k or value < max(i["value"] for i in self.items)

    def add(self, step: int, value: float, metrics: dict) -> Path | None:
        """登记新的 checkpoint，返回需要删除的旧目录。"""
        self.items.append({"step": step, "value": value, "dir": f"step_{step:06d}", "metrics": metrics})
        self.items.sort(key=lambda i: i["value"])
        evicted = self.items[self.k:]
        self.items = self.items[: self.k]
        return self.root / evicted[0]["dir"] if evicted else None

    def dump(self) -> dict:
        return {"metric": self.metric, "lower_is_better": True,
                "ranking": [{"rank": r + 1, **i} for r, i in enumerate(self.items)]}


def _test_hook() -> None:
    """仅供本地冒烟测试：PITRAIN_TEST_PATCH=<模块名> 会在启动时导入该模块（用于替换成微型网络）。"""
    import importlib
    import os

    mod = os.environ.get("PITRAIN_TEST_PATCH")
    if mod:
        importlib.import_module(mod)


def remove_dir(path: Path) -> None:
    """删除目录；失败只告警不中断训练（磁盘上多一个权重好过训练崩溃）。"""
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError as exc:
        log.warning("删除 %s 失败：%s（请稍后手动清理）", path, exc)


def save_checkpoint(out: Path, policy, pre, post, contract: dict) -> None:
    tmp = out.with_name(out.name + ".partial")
    remove_dir(tmp)
    tmp.mkdir(parents=True)
    policy.save_pretrained(tmp)          # config.json + model.safetensors（bf16，约 7GB）
    pre.save_pretrained(tmp)             # 预处理器 JSON + 归一化统计
    post.save_pretrained(tmp)
    write_json(tmp / "pitrain_contract.json", contract)
    remove_dir(out)
    tmp.rename(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    cfg.validate()
    _test_hook()

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.utils.utils import cycle

    acc = Accelerator(mixed_precision="bf16", step_scheduler_with_optimizer=False, cpu=cfg.device == "cpu",
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    main_proc = acc.is_main_process
    logging.basicConfig(level=logging.INFO if main_proc else logging.WARNING,
                        format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S", force=True)
    if acc.num_processes != cfg.gpus:
        raise RuntimeError(f"配置 gpus={cfg.gpus}，但实际启动了 {acc.num_processes} 个进程")
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(cfg.output)
    ckpt_root = run_dir / "checkpoints"
    status_path = run_dir / "status.json"
    status = {"stage": "training", "started_at": datetime.now(UTC).isoformat(), "step": 0, "steps": cfg.steps}
    if main_proc:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        write_json(status_path, status)

    # ---------------- 数据 ----------------
    meta = load_meta(Path(cfg.dataset))
    ds_root = Path(cfg.dataset)
    train_ds = open_split(ds_root, "train", meta["fps"], cfg.chunk_size, image_transforms(cfg.image_augmentation))
    val_ds = open_split(ds_root, "val", meta["fps"], cfg.chunk_size)
    stats = build_policy_stats(train_ds, meta, cfg)  # 每个进程都算一遍（纯数值、确定性、几秒钟）

    # ---------------- 模型 / 处理器 / 优化器 ----------------
    device = acc.device
    if cfg.tokenizer_max_length <= 0:
        state_dim = len(meta["state_names"]) * (2 if cfg.use_velocity else 1)
        cfg.tokenizer_max_length = required_token_length(Path(cfg.tokenizer), meta["prompt"], state_dim)
        if main_proc:
            log.info("文本提示长度自动设为 %d token（state %d 维）", cfg.tokenizer_max_length, state_dim)
    policy_cfg = build_policy_config(cfg, meta, Path(cfg.pretrained), device.type)
    policy = load_policy(policy_cfg, Path(cfg.pretrained))
    pre, post = build_processors(policy_cfg, Path(cfg.pretrained), Path(cfg.tokenizer), str(device), stats)
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in policy.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=1e-8, weight_decay=cfg.weight_decay)
    from .schedule import build_scheduler

    scheduler = build_scheduler(optimizer, cfg)

    episodes = train_ds.meta.episodes
    if episodes is None:
        raise RuntimeError("训练集缺少 episode 元数据")
    sampler = EpisodeAwareSampler(
        episodes["dataset_from_index"], episodes["dataset_to_index"],
        episode_indices_to_use=train_ds.episodes, shuffle=True, seed=cfg.seed,
        absolute_to_relative_idx=train_ds.absolute_to_relative_idx)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size_per_gpu, sampler=sampler, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None)
    policy, optimizer, loader, scheduler = acc.prepare(policy, optimizer, loader, scheduler)
    validator = Validator(val_ds, cfg, meta, acc, stats["action"])
    logger = MetricsLogger(run_dir, enabled=main_proc)
    topk = TopK(ckpt_root, cfg.keep_top_k, cfg.best_metric)
    contract = {
        "dataset": str(cfg.dataset), "dataset_name": Path(cfg.dataset).name, "prompt": meta["prompt"],
        "fps": meta["fps"], "use_velocity": cfg.use_velocity, "image_keys": meta["image_keys"],
        "state_names": meta["state_names"] + (meta["velocity_names"] if cfg.use_velocity else []),
        "action_names": meta["action_names"], "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
        "action_representation": "关节: 相对(leader - 当前follower)；夹爪: 绝对 leader",
        "gripper_filter": meta["gripper_profile"]["suggested"], "train_config": cfg.__dict__,
    }
    epoch_frames = train_ds.num_frames
    if main_proc:
        log.info("训练集 %d 帧 / %d 条，验证集 %d 帧 / %d 条", train_ds.num_frames, train_ds.num_episodes,
                 val_ds.num_frames, val_ds.num_episodes)
        log.info("参数量 %.2fB（可训练 %.2fB），全局 batch = %d × %d = %d", n_total / 1e9, n_train / 1e9,
                 cfg.batch_size_per_gpu, cfg.gpus, cfg.global_batch)

    # ---------------- 训练循环 ----------------
    it = cycle(loader)
    policy.train()
    window = {"loss": 0.0, "grad": 0.0, "n": 0, "t_data": 0.0, "t_step": 0.0}
    t_start = time.time()

    def do_validation(step: int, final: bool) -> None:
        unwrapped = acc.unwrap_model(policy)
        t0 = time.time()
        metrics = validator.run(unwrapped, pre, post)
        metrics["val_seconds"] = round(time.time() - t0, 1)
        value = metrics[cfg.best_metric]
        keep = topk.would_keep(value) if math.isfinite(value) else False
        if main_proc:
            logger.log("val", step, {**metrics, "kept": int(keep)})
            log.info("[验证 step %d] loss=%.4f  nmae=%.4f  关节MAE(前%d步)=%.2f°  夹爪MAE=%.3f  %s",
                     step, metrics["val_loss"], metrics["val_nmae"], cfg.n_action_steps,
                     metrics["val_joint_mae_deg_first"], metrics["val_gripper_mae_first"],
                     "★ 进入 top-k，保存" if keep else "")
            if keep:
                evicted = topk.add(step, value, metrics)
                save_checkpoint(ckpt_root / f"step_{step:06d}", unwrapped, pre, post, {**contract, "step": step,
                                                                                         "val_metrics": metrics})
                if evicted is not None:
                    remove_dir(evicted)
                write_json(ckpt_root / "topk.json", topk.dump())
            if final:
                save_checkpoint(ckpt_root / "final", unwrapped, pre, post, {**contract, "step": step,
                                                                            "val_metrics": metrics})
            plot_curves(run_dir, [i["step"] for i in topk.items])
        acc.wait_for_everyone()

    try:
        for step in range(1, cfg.steps + 1):
            t0 = time.perf_counter()
            batch = pre(compose_batch(next(it), cfg.use_velocity))
            t1 = time.perf_counter()
            with acc.autocast():
                loss, _ = policy(batch)
            acc.backward(loss)
            grad_norm = acc.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window["loss"] += loss.detach().float().item()
            window["grad"] += float(grad_norm) if grad_norm is not None else 0.0
            window["n"] += 1
            window["t_data"] += t1 - t0
            window["t_step"] += time.perf_counter() - t0

            if step % cfg.log_every == 0 or step == cfg.steps:
                vals = torch.tensor([window["loss"], window["grad"], window["n"]], device=device)
                vals = acc.reduce(vals, reduction="sum")
                n = float(vals[2])
                if main_proc:
                    sec = window["t_step"] / window["n"]
                    row = {"loss": float(vals[0]) / n, "grad_norm": float(vals[1]) / n,
                           "lr": scheduler.get_last_lr()[0], "epoch": step * cfg.global_batch / epoch_frames,
                           "sec_per_step": sec, "data_frac": window["t_data"] / max(window["t_step"], 1e-9),
                           "samples_per_s": cfg.global_batch / max(sec, 1e-9),
                           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0}
                    logger.log("train", step, row)
                    eta = (cfg.steps - step) * sec
                    log.info("step %6d/%d | loss %.4f | lr %.2e | grad %.2f | %.2fs/step | epoch %.2f | 显存 %.1fGB | 剩余 %s",
                             step, cfg.steps, row["loss"], row["lr"], row["grad_norm"], sec, row["epoch"],
                             row["gpu_mem_gb"], time.strftime("%H:%M:%S", time.gmtime(eta)))
                    status.update(step=step, loss=row["loss"], eta_s=round(eta))
                    write_json(status_path, status)
                window = dict.fromkeys(window, 0.0)
                window["n"] = 0

            if step % cfg.val_every == 0 or step == cfg.steps:
                do_validation(step, final=step == cfg.steps)
    except BaseException as exc:
        if main_proc:
            status.update(stage="failed", error=repr(exc))
            write_json(status_path, status)
        raise
    finally:
        logger.close()

    if main_proc:
        status.update(stage="trained", finished_at=datetime.now(UTC).isoformat(),
                      hours=round((time.time() - t_start) / 3600, 2), topk=topk.dump())
        write_json(status_path, status)
        log.info("训练完成，耗时 %.2f 小时；top-k: %s", (time.time() - t_start) / 3600,
                 [(i["step"], round(i["value"], 4)) for i in topk.items])
    acc.wait_for_everyone()
    acc.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
