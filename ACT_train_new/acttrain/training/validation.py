"""Comparable checkpoint-time train/val prior probes; no optimizer updates."""
from pathlib import Path
import random
import time

import numpy as np

from acttrain.data.core import read_json
from acttrain.evaluation.runner import action_metrics, evaluation_starts
from acttrain.model.inference import OBSERVATION_KEYS


def _prior_split(root, manifest, split, policy, preprocessor, postprocessor, stride, batch_size,
                 action_decoder, std, state_transform=None):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    entries = [entry for entry in manifest["episodes"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"No {split} Episodes for checkpoint prior probe")
    print(f"Checkpoint prior probe: split={split}, episodes={len(entries)}, stride={stride}", flush=True)
    started, cursor, metrics = time.monotonic(), 0, []
    dataset = LeRobotDataset(f"yam_local/{manifest['task']}_{split}", root=root / split, video_backend="pyav")
    for entry in entries:
        starts = evaluation_starts(entry["rows"], horizon=32, stride=stride)
        predictions, targets = [], []
        with np.load(root / entry["trace"], allow_pickle=False) as trace:
            for offset in range(0, len(starts), batch_size):
                chosen = starts[offset:offset + batch_size]
                samples = [dataset[cursor + index] for index in chosen]
                batch = {key: torch.stack([sample[key] for sample in samples]) for key in OBSERVATION_KEYS}
                if state_transform is not None:
                    batch = state_transform(batch, trace["state"], chosen)
                actions = policy.predict_action_chunk(preprocessor(batch))
                # Prior bypasses Accelerate's wrapped forward FP32 conversion.
                # Unnormalize in float32: NumPy has no bfloat16 dtype.
                decoded = postprocessor(actions.float()).detach().cpu().numpy()
                if action_decoder is not None:
                    decoded = action_decoder(decoded, batch["observation.state"].detach().cpu().numpy())
                predictions.append(decoded)
                targets.extend(trace["action"][index:index + 32] for index in chosen)
        metrics.append(action_metrics(np.concatenate(predictions), np.asarray(targets), std))
        cursor += entry["rows"]
    if cursor != len(dataset):
        raise ValueError(f"{split} Episode lengths differ from the dataset")
    return {"loss": float(np.mean([row["full_chunk_normalized_mae"] for row in metrics])),
            "acc": float(np.mean([row["gripper_threshold_agreement"] for row in metrics])),
            "windows": sum(row["windows"] for row in metrics), "seconds": time.monotonic() - started}


def prior_validation(dataset_root, policy, preprocessor, postprocessor, stride=16, batch_size=16,
                     action_decoder=None, action_std=None, include_train=True, state_transform=None):
    """Probe both splits at the same weights, preserving training mode and RNG state.

    Both losses use full legal 32-step windows, the same stride and training std,
    and equal episode weights. ``include_train=False`` supports val-only callers.
    No target actions reach the policy, including when probing the training set.
    """
    import torch

    if batch_size < 1 or stride < 1:
        raise ValueError("Positive batch size and stride required")
    root = Path(dataset_root)
    manifest = read_json(root / "manifest.json")
    std = np.asarray(action_std if action_std is not None else read_json(root / "normalization_train.json")["action"]["std"])
    was_training, python_rng, numpy_rng = policy.training, random.getstate(), np.random.get_state()
    # Each ACT process owns one GPU; don't initialize contexts on other ranks' GPUs.
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            policy.eval()
            validation = _prior_split(root, manifest, "val", policy, preprocessor, postprocessor,
                                      stride, batch_size, action_decoder, std, state_transform)
            result = {"val_loss": validation["loss"], "val_acc": validation["acc"],
                      "windows": validation["windows"], "validation_s": validation["seconds"]}
            if include_train:
                training = _prior_split(root, manifest, "train", policy, preprocessor, postprocessor,
                                       stride, batch_size, action_decoder, std, state_transform)
                result.update(train_prior_loss=training["loss"], train_prior_acc=training["acc"],
                              train_prior_windows=training["windows"], train_prior_s=training["seconds"])
    finally:
        policy.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if include_train:
        print(f"Checkpoint prior train: train_prior_loss={result['train_prior_loss']:.5f}, "
              f"windows={result['train_prior_windows']}, seconds={result['train_prior_s']:.1f}", flush=True)
    print(f"Checkpoint validation: val_loss={result['val_loss']:.5f}, gripper_agreement={result['val_acc']:.4f}, "
          f"windows={result['windows']}; not task success rate", flush=True)
    return result
