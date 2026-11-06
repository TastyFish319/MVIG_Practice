"""一键训练（交互式）：选数据集 → 选卡数 → 调超参 → 命名 → 后台执行 训练+评测+打包。

python -m pitrain.cli.train
python -m pitrain.cli.train --dataset blue_block_v2_right_30hz_64tr16va --gpus 4 --yes   # 全默认
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .. import console as ui
from ..hparams import TrainConfig, suggest
from ..nedf3 import read_json
from ..pipeline import ROOT, runtime_env
from ..preprocess import META_FILE

GB_PER_CKPT = 7.3  # bf16 权重 + 处理器，单个 checkpoint 约 7.3GB


def discover_datasets(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if (d / META_FILE).is_file():
            try:
                out.append((d, read_json(d / META_FILE)))
            except ValueError:
                continue
    return out


def available_gpus() -> int | None:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - 仅用于提示
        return None


def ask_hparams(cfg: TrainConfig, sug: dict, yes: bool) -> None:
    """逐项询问常用超参（回车=默认）；高级参数可选。"""
    ui.kv_table("自动推荐的默认值是怎么来的", sug["why"])
    if yes:
        return
    cfg.batch_size_per_gpu = ui.ask_int(
        "每卡 batch size", cfg.batch_size_per_gpu, 1, 256, hint=f"全局 = 每卡 × {cfg.gpus}；H20 96GB 每卡 32 左右较稳"
    )
    if cfg.global_batch != sug["batch_size_per_gpu"] * cfg.gpus:
        new_lr = 2.5e-5 * (cfg.global_batch / 32) ** 0.5
        ui.info(f"全局 batch 改为 {cfg.global_batch}，按平方根缩放建议 lr = {new_lr:.2e}")
        cfg.lr = float(f"{new_lr:.3g}")
    cfg.steps = ui.ask_int("训练步数 steps", cfg.steps, 100, 200_000)
    cfg.lr = ui.ask_float("初始学习率 lr", cfg.lr, 1e-7, 1e-3, hint="官方微调 2.5e-5")
    use_step = ui.confirm("启用 StepLR？（否 = 使用官方 cosine 衰减）", False)
    if use_step:
        cfg.scheduler = "steplr"
        cfg.steplr_step_size = ui.ask_int("  StepLR 衰减间隔（步）", sug["steplr_step_size"], 1, cfg.steps)
        cfg.steplr_gamma = ui.ask_float("  StepLR gamma（每次乘以）", cfg.steplr_gamma, 0.01, 1.0)
    else:
        cfg.scheduler = "cosine"
    cfg.warmup_steps = ui.ask_int("warmup 步数", cfg.warmup_steps, 0, cfg.steps)
    cfg.val_every = ui.ask_int("验证 / checkpoint 间隔（步）", cfg.val_every, 10, cfg.steps)
    cfg.keep_top_k = ui.ask_int(
        "保留 best 权重个数 top-k", cfg.keep_top_k, 1, 10, hint=f"每个约 {GB_PER_CKPT}GB，另有 final"
    )
    cfg.image_augmentation = ui.confirm("启用图像增强（亮度/对比度/色调/轻微平移）", cfg.image_augmentation)
    ui.info("注：pi0.5（Gemma）本身没有 dropout 超参；防过拟合靠图像增强、步数控制、冻结视觉编码器。")
    if ui.confirm("调整高级参数？", False):
        cfg.weight_decay = ui.ask_float("weight decay", cfg.weight_decay, 0, 1)
        cfg.grad_clip = ui.ask_float("梯度裁剪 grad_clip", cfg.grad_clip, 0.01, 100)
        cfg.n_action_steps = ui.ask_int(
            "执行步数 n_action_steps（评测/部署每次执行几步）", cfg.n_action_steps, 1, cfg.chunk_size
        )
        cfg.freeze_vision_encoder = ui.confirm(
            "冻结视觉编码器 SigLIP（省显存、防过拟合，略降上限）", cfg.freeze_vision_encoder
        )
        cfg.train_expert_only = ui.confirm("只训练动作专家（冻结整个 VLM，最省资源）", cfg.train_expert_only)
        cfg.num_workers = ui.ask_int("每卡 DataLoader 进程数", cfg.num_workers, 0, 32)
        cfg.best_metric = ["val_nmae", "val_loss"][
            ui.choose("best 依据哪个指标？", ["val_nmae（真实推理误差，推荐）", "val_loss（flow matching 损失）"], 1)
        ]
        cfg.seed = ui.ask_int("随机种子", cfg.seed)


def main(argv=None, *, launch: Callable[[Path, int, bool], int] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset")
    ap.add_argument("--gpus", type=int, choices=[1, 2, 4])
    ap.add_argument("--name")
    ap.add_argument("--velocity", choices=["yes", "no"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--yes", action="store_true", help="全部采用默认值，不询问")
    ap.add_argument("--foreground", action="store_true", help="前台运行（默认后台，SSH 断开不影响）")
    ap.add_argument("--dry-run", action="store_true", help="只生成配置，不启动")
    args = ap.parse_args(argv)
    yes = args.yes
    ui.banner("π0.5 一键训练", "选择数据集 → 卡数 → 超参数 → 命名 → 训练 + 评测 + 打包报告")

    # 1. 数据集
    ui.step(1, 5, "选择数据集")
    sets = discover_datasets(ROOT / "dataset")
    if not sets:
        ui.error("dataset/ 下没有预处理好的数据集，请先运行 bash scripts/preprocess.sh")
        return 1
    if args.dataset:
        match = [s for s in sets if s[0].name == args.dataset]
        if not match:
            ui.error(f"找不到数据集 {args.dataset}")
            return 1
        ds_dir, meta = match[0]
    else:
        ui.table(
            "可用数据集",
            ["#", "名称", "训练集", "验证集", "动作臂", "fps", "创建时间"],
            [
                [
                    i,
                    d.name,
                    f"{m['counts']['train']['episodes']}条/{m['counts']['train']['frames']}帧",
                    f"{m['counts']['val']['episodes']}条",
                    "+".join(m["action_arms"]),
                    m["fps"],
                    m["created_at"][:16],
                ]
                for i, (d, m) in enumerate(sets, 1)
            ],
        )
        ds_dir, meta = sets[0 if yes else ui.choose("选择编号", [d.name for d, _ in sets])]
    ui.ok(f"数据集：{ds_dir.name}（prompt: {meta['prompt']}）")

    # 2. 卡数
    ui.step(2, 5, "选择 GPU 数量（评测使用相同卡数）")
    n_avail = available_gpus()
    if n_avail is not None:
        ui.info(f"本机检测到 {n_avail} 张 GPU")
    gpus = args.gpus or (
        1 if yes else [1, 2, 4][ui.choose("GPU 数", ["1 张", "2 张", "4 张"], 3 if (n_avail or 0) >= 4 else 1)]
    )
    if n_avail is not None and gpus > n_avail:
        ui.warn(f"选择了 {gpus} 张卡但只检测到 {n_avail} 张；训练会失败，除非在其它机器上运行")

    # 3. 超参数
    ui.step(3, 5, "超参数（回车 = 默认值）")
    train_frames = meta["counts"]["train"]["frames"]
    sug = suggest(train_frames, gpus)
    cfg = TrainConfig(
        dataset=str(ds_dir),
        gpus=gpus,
        pretrained=str(ROOT / "pretrained/pi05_base"),
        tokenizer=str(ROOT / "pretrained/paligemma_tokenizer"),
        **{k: v for k, v in sug.items() if k in TrainConfig.__dataclass_fields__},
    )
    if args.steps:
        cfg.steps = args.steps
    if args.velocity:
        cfg.use_velocity = args.velocity == "yes"
    elif not yes:
        ui.info('速度版：把从臂关节速度也作为输入。可能让动作更连贯，也可能让模型"照着当前速度外推"而忽略画面。')
        cfg.use_velocity = ui.choose("输入 state 是否包含速度？", ["不含速度（推荐，与官方一致）", "含速度"], 1) == 1
    ask_hparams(cfg, sug, yes)

    # 4. 名称
    ui.step(4, 5, "本次训练名称")
    default_name = f"pi05_{'vel_' if cfg.use_velocity else ''}{gpus}gpu_{datetime.now():%m%d_%H%M}"
    while True:
        name = args.name or (default_name if yes else ui.ask_value("训练名称（runs/ 下的文件夹名）", default_name))
        if not re.fullmatch(r"[\w.-]{1,80}", name):
            ui.error("只能用字母/数字/下划线/点/短横线")
        elif (ROOT / "runs" / name).exists():
            ui.error("该名称已存在")
        else:
            break
        if args.name or yes:
            return 1
    run_dir = ROOT / "runs" / name
    cfg.output, cfg.run_name = str(run_dir), name
    cfg.validate()

    # 5. 确认并启动
    ui.step(5, 5, "确认")
    ui.kv_table(
        "本次训练",
        {
            "数据集": ds_dir.name,
            "GPU": gpus,
            "state 含速度": cfg.use_velocity,
            "batch": f"{cfg.batch_size_per_gpu}/卡 × {gpus} = {cfg.global_batch}",
            "steps": f"{cfg.steps}（≈ {cfg.steps * cfg.global_batch / train_frames:.1f} epoch）",
            "lr / 调度": f"{cfg.lr:.2e} / {cfg.scheduler}"
            + (f"(每{cfg.steplr_step_size}步×{cfg.steplr_gamma})" if cfg.scheduler == "steplr" else ""),
            "warmup": cfg.warmup_steps,
            "验证间隔": cfg.val_every,
            "保留": f"top-{cfg.keep_top_k} + final",
            "预计权重占用": f"≈ {(cfg.keep_top_k + 1) * GB_PER_CKPT:.0f} GB",
            "输出": run_dir,
        },
    )
    missing = [
        p
        for p in (Path(cfg.pretrained) / "model.safetensors", Path(cfg.tokenizer) / "tokenizer.model")
        if not p.exists()
    ]
    if missing:
        ui.error(f"缺少预训练文件：{[str(m) for m in missing]}；请先运行 bash scripts/download_models.sh")
        if not args.dry_run:
            return 1
    if not yes and not ui.confirm("开始？", True):
        ui.warn("已取消")
        return 130
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    cfg.save(run_dir / "train_config.json")
    if args.dry_run:
        ui.ok(f"已生成配置 {run_dir / 'train_config.json'}（dry-run，未启动）")
        return 0

    if launch is not None:
        foreground = args.foreground or (
            not yes and ui.choose("运行方式", ["后台运行（推荐，SSH 断开不影响）", "前台运行"], 1) == 1
        )
        return launch(run_dir, gpus, foreground)

    cmd = [sys.executable, "-m", "pitrain.pipeline", "--run", str(run_dir)]
    log = run_dir / "logs/pipeline.log"
    if args.foreground or (not yes and ui.choose("运行方式", ["后台运行（推荐，SSH 断开不影响）", "前台运行"], 1) == 1):
        return subprocess.call(cmd, cwd=str(ROOT), env=runtime_env())
    with log.open("ab") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=runtime_env(),
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (run_dir / "logs/pipeline.pid").write_text(str(p.pid))
    ui.ok(f"已在后台启动（PID {p.pid}）")
    ui.kv_table(
        "如何查看",
        {
            "实时日志": f"tail -f {log}",
            "进度状态": f"cat {run_dir / 'status.json'}",
            "TensorBoard": f"tensorboard --logdir {run_dir / 'tensorboard'} --port 6006",
            "中止": f"kill -- -{p.pid}" if os.name != "nt" else f"taskkill /PID {p.pid} /T /F",
            "最终产物": run_dir / f"{name}_statistics.zip",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
