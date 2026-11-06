"""ACT v2: one observation anchor for all future joints; absolute gripper."""
from contextlib import contextmanager
from copy import copy, deepcopy
import os
from pathlib import Path
import numpy as np
from acttrain.data.core import read_json, write_json, file_hash

CONTRACT_FILE = "act_action_contract.json"
CONTRACT = {"schema_version": 2, "action_semantics": "right_joints_relative_to_observation",
            "joint_state_slice": [7, 13], "gripper": "absolute", "horizon": 32,
            "fps": 20, "action_offset_frames": 1}


def _convert(actions, state, sign):
    actions, state = np.asarray(actions), np.asarray(state)
    if (actions.ndim not in (2, 3) or actions.shape[-1] != 7 or
            state.shape != (*actions.shape[:-2], 14) or
            not np.isfinite(actions).all() or not np.isfinite(state).all()):
        raise ValueError("Expected finite [H,7]/[14] or [B,H,7]/[B,14]")
    result = actions.astype(np.result_type(actions.dtype, np.float32), copy=True)
    result[..., :6] += sign * state[..., None, 7:13]
    return result


def relative_actions(actions, state):
    return _convert(actions, state, -1)


def absolute_actions(actions, state):
    return _convert(actions, state, 1)


def relative_statistics(root, entries, horizon=32):
    """Float64 population moments over every target in every legal train chunk."""
    count, mean, m2 = 0, np.zeros(7), np.zeros(7)
    low, high = np.full(7, np.inf), np.full(7, -np.inf)
    for entry in entries:
        if entry["split"] != "train":
            continue
        with np.load(Path(root) / entry["trace"], allow_pickle=False) as trace:
            state, action = trace["state"], trace["action"]
            if len(state) != entry["rows"] or len(action) != len(state):
                raise ValueError("Trace and manifest lengths differ")
            # Bound temporary memory, independent of episode length.
            for start in range(0, len(action) - horizon + 1, 256):
                indices = np.arange(start, min(start + 256, len(action) - horizon + 1))
                chunks = action[indices[:, None] + np.arange(horizon)]
                values = relative_actions(chunks.astype(np.float64), state[indices]).reshape(-1, 7)
                n, average = len(values), values.mean(axis=0)
                delta = average - mean
                m2 += ((values - average) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
                low, high = np.minimum(low, values.min(axis=0)), np.maximum(high, values.max(axis=0))
    if not count:
        raise ValueError("No legal training chunks")
    return {"mean": mean.tolist(), "std": np.maximum(np.sqrt(m2 / count), 1e-3).tolist(),
            "min": low.tolist(), "max": high.tolist(), "count": [count]}


class RelativeDataset:
    """Map-style adapter; retain official metadata/sampler and leave source untouched."""
    def __init__(self, dataset, action_stats):
        self.dataset = dataset
        self.meta = copy(dataset.meta)
        self.meta.stats = deepcopy(dataset.meta.stats)
        self.meta.stats["action"] = {key: np.asarray(value) for key, value in action_stats.items()}

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        actions, state = sample["action"], sample["observation.state"]
        if actions.shape != (32, 7) or state.shape != (14,):
            raise ValueError("ACT v2 requires a full [32,7] chunk and state14")
        if isinstance(actions, np.ndarray):
            sample["action"] = relative_actions(actions, state)
        else:
            relative = actions.clone()
            relative[:, :6] -= state[None, 7:13]
            sample["action"] = relative
        return sample


def launch_coordinates(environ=None, world_size=1):
    env = os.environ if environ is None else environ
    if world_size not in (1, 2, 4):
        raise ValueError("Supported GPU counts are 1, 2, 4")
    if int(env.get("WORLD_SIZE", "1")) != world_size or int(env.get("LOCAL_WORLD_SIZE", "1")) != world_size:
        raise ValueError(f"Use one node and --nproc-per-node={world_size}")
    rank, local = int(env.get("RANK", "0")), int(env.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size or local != rank:
        raise ValueError("Expected matching global/local ranks on one node")
    return rank, local


@contextmanager
def relative_training(trainer, dataset_root, stats):
    original_dataset, original_save = trainer.make_dataset, trainer.save_checkpoint
    contract = {**CONTRACT, "dataset_manifest_sha256": file_hash(Path(dataset_root) / "manifest.json"),
                "normalization": "train_legal_chunks_all_32_targets", "action_stats": stats}
    manifest = read_json(Path(dataset_root) / "manifest.json")
    contract.update(state_source=manifest.get("state_source", "follower"),
                    action_source=manifest.get("action_source", "follower"))

    def dataset(*args, **kwargs):
        return RelativeDataset(original_dataset(*args, **kwargs), stats)

    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        directory = Path(kwargs["checkpoint_dir"]) / "pretrained_model"
        write_json(directory / CONTRACT_FILE, contract)
        return result

    trainer.make_dataset, trainer.save_checkpoint = dataset, save
    try:
        yield
    finally:
        trainer.make_dataset, trainer.save_checkpoint = original_dataset, original_save


def checkpoint_contract(checkpoint):
    path = Path(checkpoint) / CONTRACT_FILE
    if not path.exists():
        train_config = Path(checkpoint) / "train_config.json"
        if train_config.exists() and read_json(train_config).get("job_name") == "yam_right_arm_act_relative_v2":
            raise ValueError("ACT v2 checkpoint is missing its action contract; copy the complete pretrained_model")
        return None
    value = read_json(path)
    if any(value.get(key) != expected for key, expected in CONTRACT.items()):
        raise ValueError("Unsupported ACT relative action contract")
    return value
