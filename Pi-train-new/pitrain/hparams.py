"""训练超参数：定义 + 根据数据集大小 / 卡数自动给出默认值。

自动默认值的思路（详见 docs/tuning_guide.md）：

    全局 batch  = 32（OpenPI 官方微调值），平均分到每张卡
    学习率      = 2.5e-5 × sqrt(全局batch / 32)        平方根缩放
    训练轮数    = 小数据(≤5 万帧)约 15 个 epoch，数据越多 epoch 越少
    steps       = 轮数 × 训练帧数 / 全局batch，限制在 [3000, 30000]
    验证间隔    = steps / 10
    warmup      = min(1000, 5% steps)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_GLOBAL_BATCH = 32
BASE_LR = 2.5e-5


@dataclass
class TrainConfig:
    # --- 数据 / 输出 ---
    dataset: str = ""
    output: str = ""
    run_name: str = ""
    pretrained: str = "pretrained/pi05_base"
    tokenizer: str = "pretrained/paligemma_tokenizer"
    use_velocity: bool = False          # state 是否拼接 follower 速度
    # --- 规模 ---
    gpus: int = 1
    batch_size_per_gpu: int = 32
    steps: int = 15000
    num_workers: int = 6
    seed: int = 42
    # --- 优化器（AdamW，与 OpenPI / LeRobot pi05 预设一致）---
    lr: float = BASE_LR
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 750
    # --- 学习率调度 ---
    scheduler: str = "cosine"           # cosine | steplr | constant
    cosine_final_ratio: float = 0.1     # cosine 结束时 lr = 初始 lr × 该比例（官方 2.5e-6 / 2.5e-5）
    steplr_step_size: int = 5000
    steplr_gamma: float = 0.5
    # --- 模型 ---
    chunk_size: int = 50                # 一次预测未来多少步（30Hz 下 50 步 ≈ 1.67s）
    n_action_steps: int = 15            # 部署/评测时每次执行多少步再重新推理（0.5s）
    num_inference_steps: int = 10       # flow matching 去噪步数
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    gradient_checkpointing: bool = True
    tokenizer_max_length: int = 0       # 0 = 自动（按 state 维度计算，避免截断/浪费）
    image_augmentation: bool = True     # 颜色/锐度/轻微仿射扰动，小数据集防过拟合
    # --- 验证与保存 ---
    val_every: int = 1500               # 每隔多少 step 验证一次（同时也是 checkpoint 间距）
    val_loss_samples: int = 512         # 验证 loss 用多少帧（固定子集）
    val_infer_samples: int = 128        # 验证推理 MAE 用多少帧（固定子集）
    best_metric: str = "val_nmae"       # val_nmae | val_loss
    keep_top_k: int = 2                 # 保留最好的 k 个权重（~7GB/个）+ final
    log_every: int = 20
    # --- 归一化保护 ---
    min_state_span: float = 0.2         # state 的 q99-q01 至少这么大（rad），防止静止关节的噪声被放大
    min_velocity_span: float = 0.5      # 速度的 q99-q01 最小跨度（rad/s）
    min_action_span: float = 0.05       # 相对动作的 q99-q01 最小跨度
    device: str = "cuda"                # 仅用于本地冒烟测试时设为 cpu
    extra: dict = field(default_factory=dict)

    @property
    def global_batch(self) -> int:
        return self.batch_size_per_gpu * self.gpus

    def validate(self) -> None:
        if self.gpus not in (1, 2, 4):
            raise ValueError("gpus 只能是 1/2/4")
        if self.scheduler not in ("cosine", "steplr", "constant"):
            raise ValueError("scheduler 只能是 cosine/steplr/constant")
        if not 0 < self.steplr_gamma <= 1:
            raise ValueError("steplr_gamma 需在 (0,1]")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps 不能大于 chunk_size")
        if self.best_metric not in ("val_nmae", "val_loss"):
            raise ValueError("best_metric 只能是 val_nmae / val_loss")
        if self.keep_top_k < 1 or self.steps < 1 or self.val_every < 1:
            raise ValueError("keep_top_k / steps / val_every 必须为正")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取训练配置 {path}: {exc}") from exc
        data["betas"] = tuple(data.get("betas", (0.9, 0.95)))
        return cls(**data)


def round_to(x: float, base: int) -> int:
    return int(max(base, base * round(x / base)))


def suggest(train_frames: int, gpus: int, global_batch: int = BASE_GLOBAL_BATCH) -> dict:
    """根据训练帧数和卡数给出默认超参数，并附上推导说明。"""
    per_gpu = max(1, global_batch // gpus)
    global_batch = per_gpu * gpus
    epochs = 15.0 if train_frames <= 50_000 else max(4.0, 15.0 * 50_000 / train_frames)
    steps = int(min(30_000, max(3_000, round_to(epochs * train_frames / global_batch, 500))))
    lr = BASE_LR * math.sqrt(global_batch / BASE_GLOBAL_BATCH)
    val_every = max(200, round_to(steps / 10, 100))
    return {
        "batch_size_per_gpu": per_gpu, "steps": steps, "lr": float(f"{lr:.3g}"),
        "warmup_steps": int(min(1000, round_to(0.05 * steps, 50))), "val_every": val_every,
        "steplr_step_size": round_to(steps / 3, 100),
        "why": {
            "全局batch": f"{global_batch} = 每卡 {per_gpu} × {gpus} 卡",
            "目标epoch": f"{epochs:.1f}（训练集 {train_frames} 帧）",
            "steps": f"{epochs:.1f} × {train_frames} / {global_batch} ≈ {steps}",
            "lr": f"2.5e-5 × sqrt({global_batch}/32) = {lr:.3g}",
        },
    }
