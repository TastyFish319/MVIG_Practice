"""Read the exported dataset using LeRobot; never create or train a model."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from acttrain.data.core import IMAGE_KEYS, legal_window_starts, read_json, write_json


def validate(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if not (set(manifest["split"].values()) == {"train", "val"}):
        raise ValueError("数据集一致性检查失败: set(manifest[\"split\"].values()) == {\"train\", \"val\"}")
    if not (len(set(manifest["split"])) == len(manifest["episodes"])):
        raise ValueError("数据集一致性检查失败: len(set(manifest[\"split\"])) == len(manifest[\"episodes\"])")
    checked_rows, legal_chunks, checked_chunk_samples, checked_images = 0, 0, 0, 0
    result = {"status": "passed", "local_training_executed": False, "splits": {}}
    for partition in ("train", "val"):
        entries = [e for e in manifest["episodes"] if e["split"] == partition]
        dataset = LeRobotDataset(f"yam_local/{manifest['task']}_{partition}", root=root / partition, video_backend="pyav")
        if not (dataset.num_episodes == len(entries)):
            raise ValueError("数据集一致性检查失败: dataset.num_episodes == len(entries)")
        if not (dataset.features["observation.state"]["shape"] == (14,)):
            raise ValueError("数据集一致性检查失败: dataset.features[\"observation.state\"][\"shape\"] == (14,)")
        if not (dataset.features["action"]["shape"] == (7,)):
            raise ValueError("数据集一致性检查失败: dataset.features[\"action\"][\"shape\"] == (7,)")
        numeric = dataset.hf_dataset.select_columns(["observation.state", "action", "episode_index", "frame_index", "timestamp"])
        cursor = 0
        chunk_dataset = LeRobotDataset(
            f"yam_local/{manifest['task']}_{partition}", root=root / partition,
            video_backend="pyav", delta_timestamps={"action": [j / 20 for j in range(manifest["action_horizon"])]},
        )
        for entry in entries:
            with np.load(root / entry["trace"], allow_pickle=False) as trace:
                if not (np.all(trace["action_timestamp_ns"] - trace["timestamp_ns"] == 50_000_000)):
                    raise ValueError("数据集一致性检查失败: np.all(trace[\"action_timestamp_ns\"] - trace[\"timestamp_ns\"] == 50_000_000)")
                if manifest.get("action_source", "follower") == "leader":
                    if not (manifest["action_semantics"] == "future_leader_right_absolute"):
                        raise ValueError("数据集一致性检查失败: manifest[\"action_semantics\"] == \"future_leader_right_absolute\"")
                    source = trace["full_action_source"]
                else:
                    source = trace["full_state"][:, 7:14]
                if not (source.shape == (entry["rows"] + 1, 7)):
                    raise ValueError("数据集一致性检查失败: source.shape == (entry[\"rows\"] + 1, 7)")
                np.testing.assert_array_equal(trace["action"], source[1:])
                for key in trace.files:
                    if key.startswith("action_source_time_"):
                        ages = trace["action_timestamp_ns"] - trace[key]
                        if not (np.all((ages >= 0) & (ages <= manifest["max_source_age_ms"] * 1e6))):
                            raise ValueError("数据集一致性检查失败: np.all((ages >= 0) & (ages <= manifest[\"max_source_age_ms\"] * 1e6))")
                    if key.startswith("source_time_"):
                        ages = trace["timestamp_ns"] - trace[key]
                        if not (np.all((ages >= 0) & (ages <= manifest["max_source_age_ms"] * 1e6))):
                            raise ValueError("数据集一致性检查失败: np.all((ages >= 0) & (ages <= manifest[\"max_source_age_ms\"] * 1e6))")
                batch = numeric[cursor:cursor + entry["rows"]]
                state = np.asarray(batch["observation.state"], dtype=np.float32)
                action = np.asarray(batch["action"], dtype=np.float32)
                np.testing.assert_array_equal(state, trace["state"])
                np.testing.assert_array_equal(action, trace["action"])
                if not (np.all(np.asarray(batch["episode_index"]).reshape(-1) == entry["dataset_episode_index"])):
                    raise ValueError("数据集一致性检查失败: np.all(np.asarray(batch[\"episode_index\"]).reshape(-1) == entry[\"dataset_episode_index\"])")
                np.testing.assert_allclose(np.asarray(batch["timestamp"]).reshape(-1), np.arange(entry["rows"]) / 20, atol=1e-4)
                for key, expected in entry["image_rgb_sha256_checks"].items():
                    side, index = key.split(":")
                    image = dataset[cursor + int(index)][IMAGE_KEYS[side]]
                    recovered = np.rint(image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    if not (hashlib.sha256(recovered.tobytes()).hexdigest() == expected):
                        raise ValueError("数据集一致性检查失败: hashlib.sha256(recovered.tobytes()).hexdigest() == expected")
                    checked_images += 1
                valid = list(legal_window_starts(entry["rows"], manifest["action_horizon"]))
                if not (len(valid) == entry["legal_action_windows"]):
                    raise ValueError("数据集一致性检查失败: len(valid) == entry[\"legal_action_windows\"]")
                for index in (valid[0], valid[-1]):
                    sample = chunk_dataset[cursor + index]
                    if not (sample["action"].shape == (manifest["action_horizon"], 7)):
                        raise ValueError("数据集一致性检查失败: sample[\"action\"].shape == (manifest[\"action_horizon\"], 7)")
                    if not (not sample["action_is_pad"].any()):
                        raise ValueError("数据集一致性检查失败: not sample[\"action_is_pad\"].any()")
                    np.testing.assert_array_equal(sample["action"].numpy(), trace["action"][index:index + manifest["action_horizon"]])
                    checked_chunk_samples += 1
                cursor += entry["rows"]
                checked_rows += entry["rows"]
                legal_chunks += len(valid)
            print(f"Validated {partition}/{entry['dataset_episode_index']:02}: {entry['source_relative']}", flush=True)
        if not (cursor == len(dataset)):
            raise ValueError("数据集一致性检查失败: cursor == len(dataset)")
        result["splits"][partition] = {"episodes": len(entries), "rows": cursor}
    train_stats = read_json(root / "train/meta/stats.json")
    if not (read_json(root / "normalization_train.json") == train_stats):
        raise ValueError("数据集一致性检查失败: read_json(root / \"normalization_train.json\") == train_stats")
    original_train_stats = read_json(root / "provenance/train_stats_unclamped.json")
    for key in IMAGE_KEYS.values():
        if not (train_stats[key] == original_train_stats[key]):
            raise ValueError("数据集一致性检查失败: train_stats[key] == original_train_stats[key]")
    training = [entry for entry in manifest["episodes"] if entry["split"] == "train"]
    for key, trace_key in (("observation.state", "state"), ("action", "action")):
        values = []
        for entry in training:
            with np.load(root / entry["trace"], allow_pickle=False) as trace:
                values.append(trace[trace_key])
        values = np.concatenate(values).astype(np.float64)
        np.testing.assert_allclose(train_stats[key]["mean"], values.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(train_stats[key]["min"], values.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(train_stats[key]["max"], values.max(axis=0), atol=1e-6)
        np.testing.assert_allclose(train_stats[key]["std"], np.maximum(values.std(axis=0), manifest["normalization_std_floor"]), atol=1e-5)
        if not (train_stats[key]["count"] == [len(values)]):
            raise ValueError("数据集一致性检查失败: train_stats[key][\"count\"] == [len(values)]")
    result.update(checked_numeric_rows=checked_rows, legal_action_windows=legal_chunks, checked_chunk_samples=checked_chunk_samples, checked_rgb_roundtrips=checked_images,
                  normalization="train_only", action_target=f"next_50ms_{manifest.get('action_source', 'follower')}_right_absolute", source_timestamp_causal=True)
    write_json(root / "validation.json", result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    validate(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
