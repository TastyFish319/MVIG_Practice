"""Copied inference-only functions from Pi-train-new/pitrain/policy_factory.py."""

from __future__ import annotations

from pathlib import Path

import torch

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

    from .io import read_json

    ckpt_dir = Path(ckpt_dir)
    contract = read_json(ckpt_dir / "pitrain_contract.json")  # 内部已把读取错误转为 ValueError
    from pi_deploy.model.spec import resolve_spec
    resolve_spec(ckpt_dir)  # Also enforce names against saved config/processor metadata.
    pc = PreTrainedConfig.from_pretrained(str(ckpt_dir), local_files_only=True)
    if not isinstance(pc, PI05Config):
        raise TypeError(f"{ckpt_dir} 不是 pi05 checkpoint（type={pc.type}）")
    pc.device = device
    pc.pretrained_path = ckpt_dir
    pc.gradient_checkpointing = False
    pc.action_feature_names = contract["action_names"]
    class CheckedPI05Policy(PI05Policy):
        """The upstream loader catches exceptions; require an actual strict load receipt."""
        _deployment_loaded = False

        def load_state_dict(self, *args, **kwargs):
            result = super().load_state_dict(*args, **kwargs)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError('checkpoint 参数缺失或多余，拒绝不完整加载')
            self._deployment_loaded = True
            return result

    policy = CheckedPI05Policy.from_pretrained(str(ckpt_dir), config=pc, local_files_only=True, strict=True)
    if not policy._deployment_loaded:
        raise RuntimeError('Pi05 权重未完成严格加载；上游加载器可能吞掉异常，详见 server.log')
    _verify_weights_loaded(policy, ckpt_dir / "model.safetensors")
    policy.eval()
    pre, post = build_processors(pc, ckpt_dir, tokenizer, device, stats=None)
    return policy, pre, post, contract
