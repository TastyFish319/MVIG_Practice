"""Causal right-follower velocity observations for ACT v3; source data stays state14."""
from contextlib import contextmanager
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
from acttrain.data.core import read_json, write_json, file_hash, numeric_statistics
from acttrain.model.relative import relative_training, absolute_actions

CONTRACT_FILE = "act_observation_contract.json"
CONTRACT = {"schema_version": 3, "state_dim": 21, "base_state_dim": 14,
            "velocity_source": "follower_right", "velocity_slice": [7, 14],
            "lag_frames": 2, "fps": 20, "dt_seconds": 0.1,
            "episode_start": "first_two_zero", "units": "six_rad_per_s_and_normalized_gripper_per_s"}


def augment_states(state, timestamps=None):
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 14 or not np.isfinite(state).all():
        raise ValueError("Expected finite episode states [N,14]")
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.shape != (len(state),) or not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), 50_000_000, rtol=0, atol=1):
            raise ValueError("Velocity requires a regular 20 Hz timestamp_ns grid")
    result = np.zeros((len(state), 21), np.float32)
    result[:, :14] = state
    result[2:, 14:] = (state[2:, 7:14] - state[:-2, 7:14]) / .1
    return result


def episode_states(root, entry):
    with np.load(Path(root) / entry["trace"], allow_pickle=False) as trace:
        result = augment_states(trace["state"], trace["timestamp_ns"])
    if len(result) != entry["rows"]:
        raise ValueError("Trace and manifest lengths differ")
    return result


def velocity_state_statistics(root, entries, original):
    arrays = [episode_states(root, e)[:, 14:] for e in entries if e["split"] == "train"]
    if not arrays:
        raise ValueError("No training episodes")
    values = np.concatenate(arrays)
    extra = numeric_statistics(values)
    result = deepcopy(original)
    for key in ("mean", "std", "min", "max"):
        result[key] = np.concatenate([np.asarray(original[key]), extra[key]]).tolist()
    for key in original:
        if key.startswith("q") and key[1:].isdigit():
            result[key] = np.concatenate([np.asarray(original[key]), np.quantile(values, int(key[1:]) / 100, axis=0)]).tolist()
    return result


class VelocityDataset:
    def __init__(self, dataset, root, entries, split, state_stats):
        self.dataset = dataset
        selected = sorted((e for e in entries if e["split"] == split), key=lambda e: e["dataset_episode_index"])
        if [e["dataset_episode_index"] for e in selected] != list(range(len(selected))):
            raise ValueError("Expected contiguous dataset episode indices")
        self.states = np.concatenate([episode_states(root, e) for e in selected])
        if len(self.states) != len(dataset):
            raise ValueError("Manifest and dataset lengths differ")
        self.meta = copy(dataset.meta)
        if hasattr(dataset.meta, "info"):
            self.meta.info = deepcopy(dataset.meta.info)
        else:
            self.meta.features = deepcopy(dataset.meta.features)
        feature = self.meta.features["observation.state"]
        feature["shape"] = (21,)
        if isinstance(feature.get("names"), list):
            feature["names"] = feature["names"] + [f"right_velocity_{i}" for i in range(7)]
        self.meta.stats = deepcopy(dataset.meta.stats)
        self.meta.stats["observation.state"] = {k: np.asarray(v) for k, v in state_stats.items()}

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        state = sample["observation.state"]
        raw = state if isinstance(state, np.ndarray) else state.detach().cpu().numpy()
        if raw.shape != (14,) or not np.allclose(raw, self.states[index, :14], atol=1e-6, rtol=1e-5):
            raise ValueError("Dataset state does not match its episode trace")
        values = self.states[index].copy()
        sample["observation.state"] = values if isinstance(state, np.ndarray) else state.new_tensor(values)
        return sample

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def prepare_batch(batch, states, indices):
    state = batch["observation.state"]
    values = augment_states(states)[np.asarray(indices, dtype=int)]
    raw = state.detach().cpu().numpy()
    if raw.shape != values[:, :14].shape or not np.allclose(raw, values[:, :14], atol=1e-6, rtol=1e-5):
        raise ValueError("Evaluation observations differ from their episode history")
    return {**batch, "observation.state": state.new_tensor(values)}


def decode_actions(actions, state):
    return absolute_actions(actions, np.asarray(state)[..., :14])


@contextmanager
def velocity_training(trainer, root, action_stats, state_stats):
    manifest = read_json(Path(root) / "manifest.json")
    contract = {**CONTRACT, "state_stats": state_stats,
                "dataset_manifest_sha256": file_hash(Path(root) / "manifest.json")}
    with relative_training(trainer, root, action_stats):
        original_dataset, original_save = trainer.make_dataset, trainer.save_checkpoint
        def dataset(*args, **kwargs):
            return VelocityDataset(original_dataset(*args, **kwargs), root, manifest["episodes"], "train", state_stats)
        def save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            write_json(Path(kwargs["checkpoint_dir"]) / "pretrained_model" / CONTRACT_FILE, contract)
            return result
        trainer.make_dataset, trainer.save_checkpoint = dataset, save
        try:
            yield
        finally:
            trainer.make_dataset, trainer.save_checkpoint = original_dataset, original_save
