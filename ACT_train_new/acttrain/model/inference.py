"""Restore an ACT checkpoint and its saved processors; no robot/optimizer imports."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from acttrain.data.core import IMAGE_KEYS, prepare_rgb

OBSERVATION_KEYS = ("observation.state", *IMAGE_KEYS.values())
CHECKPOINT_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")


def checkpoint_directory(path):
    path = Path(path).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path /= "pretrained_model"
    for name in CHECKPOINT_FILES:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {path / name}; copy the complete pretrained_model directory")
    return path


def prepare_live_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Live images must be nonempty uint8 HWC RGB")
    return prepare_rgb(rgb[..., ::-1].copy(), 256)


def observation_batch(batch):
    """Whitelist inputs so a validation target can never enter the VAE encoder."""
    return {key: batch[key] for key in OBSERVATION_KEYS}


class ACTInference:
    state_dim = 14

    def __init__(self, checkpoint, device="cuda", expected_stats=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if importlib.metadata.version("lerobot") != "0.4.4":
            raise ValueError("This checkpoint adapter requires lerobot==0.4.4")
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("Use an explicit cpu or cuda device")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
        self.checkpoint = checkpoint_directory(checkpoint)
        from acttrain.model.relative import checkpoint_contract
        self.action_contract = checkpoint_contract(self.checkpoint)
        cfg = PreTrainedConfig.from_pretrained(str(self.checkpoint), local_files_only=True)
        if cfg.type != "act" or cfg.chunk_size != 32 or cfg.n_obs_steps != 1:
            raise ValueError("Expected ACT with one observation and chunk_size=32")
        expected_shapes = {"observation.state": (self.state_dim,), **{key: (3, 256, 256) for key in IMAGE_KEYS.values()}}
        if {key: tuple(value.shape) for key, value in cfg.input_features.items()} != expected_shapes:
            raise ValueError(f"Checkpoint observation features differ from the dual-wrist/state{self.state_dim} contract")
        if set(cfg.output_features) != {"action"} or tuple(cfg.output_features["action"].shape) != (7,):
            raise ValueError("Checkpoint must predict seven right-arm axes")
        # All backbone parameters are restored from the checkpoint. Avoid ImageNet downloads.
        cfg.pretrained_backbone_weights = None
        cfg.device = "cpu"
        self.policy = ACTPolicy.from_pretrained(str(self.checkpoint), config=cfg, strict=True, local_files_only=True)
        self.policy.to(device).eval()
        self.policy.config.device = device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": {"device": device}},
            postprocessor_overrides={"unnormalizer_processor": {"device": device}},
        )
        self.torch, self.device = torch, device
        if expected_stats is not None:
            if self.action_contract is not None:
                expected_stats = {**expected_stats, "action": self.action_contract["action_stats"]}
            self.verify_normalization(expected_stats)

    def verify_normalization(self, expected):
        from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        for pipeline, kind, keys in ((self.preprocessor, NormalizerProcessorStep, ("observation.state", "action")),
                                     (self.postprocessor, UnnormalizerProcessorStep, ("action",))):
            steps = [step for step in pipeline.steps if isinstance(step, kind)]
            if len(steps) != 1:
                raise ValueError("Checkpoint lacks its saved normalization processor")
            for key in keys:
                for stat in ("mean", "std"):
                    np.testing.assert_allclose(steps[0].stats[key][stat], expected[key][stat], atol=1e-6, rtol=1e-5,
                                               err_msg=f"Checkpoint {key}/{stat} differs from train statistics")

    def predict_batch(self, batch):
        inputs = observation_batch(batch)
        state = inputs["observation.state"]
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Expected batched state [B,{self.state_dim}]")
        for key, value in inputs.items():
            if not self.torch.isfinite(value).all():
                raise ValueError(f"Nonfinite input: {key}")
            if key in IMAGE_KEYS.values() and (tuple(value.shape) != (len(state), 3, 256, 256)
                                               or value.min() < 0 or value.max() > 1):
                raise ValueError("Images must be batched CHW float RGB in [0,1]")
        with self.torch.inference_mode():
            prepared = self.preprocessor(inputs)
            # Never call forward(batch-with-action): inference uses the zero/prior latent.
            normalized = self.policy.predict_action_chunk(prepared)
            actions = self.postprocessor(normalized.float()).detach().cpu().numpy()
            if self.action_contract is not None:
                from acttrain.model.relative import absolute_actions
                actions = absolute_actions(actions, state.detach().cpu().numpy()[:, :14])
        if actions.shape != (len(state), 32, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid ACT action output")
        return actions

    def predict_rgb(self, state, left_rgb, right_rgb):
        state = np.asarray(state, np.float32)
        if state.shape != (14,) or not np.isfinite(state).all() or np.any((state[[6, 13]] < 0) | (state[[6, 13]] > 1)):
            raise ValueError("Expected finite state14, radians and normalized grippers")
        batch = {"observation.state": self.torch.from_numpy(state.copy())[None]}
        for side, rgb in (("left", left_rgb), ("right", right_rgb)):
            chw = prepare_live_rgb(rgb).transpose(2, 0, 1).copy().astype(np.float32) / 255
            batch[IMAGE_KEYS[side]] = self.torch.from_numpy(chw)[None]
        return self.predict_batch(batch)[0]
