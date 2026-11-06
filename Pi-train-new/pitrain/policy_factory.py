"""构建 pi0.5 策略与前/后处理器（复用 LeRobot 官方实现）。

预处理器（preprocessor）做的事，按顺序::

    原始 batch ──▶ 加 batch 维 ──▶ 相对动作化(action -= state, 夹爪除外)
               ──▶ 分位数归一化到 [-1,1] ──▶ state 离散成 256 档写进文字提示
               ──▶ tokenizer 把 "Task: ..., State: 12 200 ...;\\nAction: " 变成 token
               ──▶ 搬到 GPU

后处理器（postprocessor）::

    模型输出 ──▶ 反归一化 ──▶ 加回 state 得到绝对动作 ──▶ 搬回 CPU
"""
from __future__ import annotations

from pathlib import Path

import torch

from .data import policy_features


def build_policy_config(cfg, meta: dict, pretrained: Path, device: str):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    inputs, outputs = policy_features(meta, cfg.use_velocity, meta["image_size"])
    pc = PI05Config(
        dtype="bfloat16", chunk_size=cfg.chunk_size, n_action_steps=cfg.n_action_steps,
        num_inference_steps=cfg.num_inference_steps, use_relative_actions=True,
        relative_exclude_joints=["gripper"], gradient_checkpointing=cfg.gradient_checkpointing,
        freeze_vision_encoder=cfg.freeze_vision_encoder, train_expert_only=cfg.train_expert_only,
        tokenizer_max_length=cfg.tokenizer_max_length, device=device,
        optimizer_lr=cfg.lr, optimizer_weight_decay=cfg.weight_decay, optimizer_grad_clip_norm=cfg.grad_clip,
    )
    pc.input_features, pc.output_features = inputs, outputs
    pc.action_feature_names = list(meta["action_names"])
    pc.pretrained_path = Path(pretrained)
    return pc


def _verify_weights_loaded(policy, weights_file: Path) -> None:
    """LeRobot 的 from_pretrained 在加载失败时只打印警告并返回随机初始化模型——这里强制检查。"""
    from safetensors import safe_open

    sd = policy.state_dict()
    probes = ("action_in_proj.weight", "action_out_proj.weight", "time_mlp_in.weight")
    with safe_open(str(weights_file), framework="pt") as f:
        checked = 0
        for key in f.keys():  # noqa: SIM118 - safe_open 对象不支持直接迭代
            if not key.endswith(probes):
                continue
            target = key if key.startswith("model.") else f"model.{key}"
            got = sd.get(target)
            if got is None or tuple(got.shape) != tuple(f.get_slice(key).get_shape()):
                continue
            ref = f.get_tensor(key).float()
            if not torch.allclose(got.float().cpu(), ref, atol=1e-2, rtol=1e-2):
                raise RuntimeError(f"预训练权重未正确加载：{target} 与文件不一致")
            checked += 1
        if not checked:
            raise RuntimeError("无法核对预训练权重（找不到对照参数），请检查 pretrained 目录")


def required_token_length(tokenizer: Path, prompt: str, state_dim: int) -> int:
    """估算文本提示最多需要多少 token。

    pi0.5 把 state 离散成 0~255 的整数写进提示里，Gemma 分词器按单个数字切分，
    所以 14 维 state 约 120 token，28 维（含速度）约 210 token——超过默认上限 200 会被截断，
    把末尾的 "Action:" 切掉。这里用最坏情况（每维都是 3 位数）算出所需长度，向上取整到 16 的倍数。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    text = prompt.strip().replace("_", " ").replace("\n", " ")
    worst = f"Task: {text}, State: {' '.join(['255'] * state_dim)};\nAction: "
    n = len(tok(worst)["input_ids"]) + 4
    return int(16 * -(-n // 16))


@torch.no_grad()
def predict_chunk(policy, batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
    """与 PI05Policy.predict_action_chunk 相同，但可以显式传入初始噪声（用于可复现的评测）。

    flow matching 推理：从高斯噪声出发，沿模型预测的"速度场"走 num_inference_steps 步，得到动作序列。
    """
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy.eval()
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    return actions[:, :, : policy.config.output_features[ACTION].shape[0]]


def load_policy(policy_cfg, weights_dir: Path):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    weights_dir = Path(weights_dir)
    if not (weights_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"{weights_dir}/model.safetensors 不存在；先运行 bash scripts/download_models.sh")
    policy = PI05Policy.from_pretrained(str(weights_dir), config=policy_cfg, local_files_only=True)
    _verify_weights_loaded(policy, weights_dir / "model.safetensors")
    return policy


def build_processors(policy_cfg, processor_dir: Path, tokenizer: Path, device: str, stats: dict | None):
    """从 processor_dir（pi05_base 或我们自己的 checkpoint）加载处理器 JSON，并覆盖关键参数。"""
    from lerobot.policies.factory import make_pre_post_processors

    pre_over = {
        "device_processor": {"device": device},
        "tokenizer_processor": {"tokenizer_name": str(tokenizer), "max_length": policy_cfg.tokenizer_max_length},
        "normalizer_processor": {"features": {**policy_cfg.input_features, **policy_cfg.output_features},
                                 "norm_map": policy_cfg.normalization_mapping},
        "relative_actions_processor": {"enabled": True, "exclude_joints": ["gripper"],
                                       "action_names": policy_cfg.action_feature_names},
    }
    post_over = {
        "unnormalizer_processor": {"features": policy_cfg.output_features, "norm_map": policy_cfg.normalization_mapping},
        "absolute_actions_processor": {"enabled": True},
    }
    if stats is not None:  # 训练开始：用本数据集训练集统计；评测：沿用 checkpoint 里保存的统计
        pre_over["normalizer_processor"]["stats"] = stats
        post_over["unnormalizer_processor"]["stats"] = stats
    return make_pre_post_processors(policy_cfg, pretrained_path=str(processor_dir),
                                    preprocessor_overrides=pre_over, postprocessor_overrides=post_over)


def load_checkpoint(ckpt_dir: Path, tokenizer: Path, device: str):
    """评测/部署：从我们保存的 checkpoint 恢复 (policy, preprocessor, postprocessor, contract)。"""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    from .nedf3 import read_json

    ckpt_dir = Path(ckpt_dir)
    contract = read_json(ckpt_dir / "pitrain_contract.json")  # 内部已把读取错误转为 ValueError
    pc = PreTrainedConfig.from_pretrained(str(ckpt_dir), local_files_only=True)
    if not isinstance(pc, PI05Config):
        raise TypeError(f"{ckpt_dir} 不是 pi05 checkpoint（type={pc.type}）")
    pc.device = device
    pc.pretrained_path = ckpt_dir
    pc.gradient_checkpointing = False
    pc.action_feature_names = contract["action_names"]
    policy = PI05Policy.from_pretrained(str(ckpt_dir), config=pc, local_files_only=True)
    _verify_weights_loaded(policy, ckpt_dir / "model.safetensors")
    policy.eval()
    pre, post = build_processors(pc, ckpt_dir, tokenizer, device, stats=None)
    return policy, pre, post, contract
