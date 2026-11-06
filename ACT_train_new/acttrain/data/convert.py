"""Convert final human-approved NEDF3 episodes with the official LeRobot writer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from acttrain.data.core import ACTION_NAMES, IMAGE_KEYS, STATE_NAMES, file_hash, finalize_normalization, legal_window_starts, load_selection, prepare_rgb, read_episode, read_json, split_episodes, write_json


def features(size):
    result = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }
    for key in IMAGE_KEYS.values():
        result[key] = {"dtype": "image", "shape": (size, size, 3), "names": ["height", "width", "channels"]}
    return result


def _convert(raw_root, config_path, output_override=None, attempt=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    repo = Path(raw_root).resolve()
    cfg = read_json(config_path)
    from acttrain.cli.preprocess import validate_config
    validate_config(cfg)
    action_source = cfg.get("action_source", "leader")
    if action_source not in ("follower", "leader"):
        raise ValueError("action_source must be follower or leader")
    if importlib.metadata.version("lerobot") != cfg["lerobot_version"]:
        raise ValueError("Use the pinned LeRobot version to keep the export contract stable")
    if cfg["fps"] != 20 or cfg["action_offset_frames"] != 1:
        raise ValueError("This converter implements exactly 20Hz and a one-frame action offset")
    approved_path, review_path = repo / cfg["approved_file"], repo / cfg["review_file"]
    task, selected = load_selection(repo, approved_path, review_path)
    if task != cfg["task_id"]:
        raise ValueError("Config task and final review task differ")
    split = split_episodes([entry["relative"] for entry in selected], cfg["seed"], cfg["val_fraction"],
                           validation_episodes=cfg.get("validation_episodes"))
    output = Path(output_override).resolve() if output_override else (repo / cfg["output"]).resolve()
    stage = output.with_name(output.name + ".partial")
    if output.exists() or stage.exists():
        raise FileExistsError("Destination or .partial already exists; inspect it and choose a new output, never overwrite")
    stage.mkdir(parents=True)
    write_json(stage / "status.json", {"status":"running","stage":"preprocessing","attempt":attempt})
    (stage / "provenance").mkdir()
    (stage / "traces").mkdir()
    shutil.copy2(approved_path, stage / "provenance/approved_episodes.json")
    shutil.copy2(review_path, stage / "provenance/human_review.json")
    shutil.copy2(config_path, stage / "provenance/conversion_config.json")
    from acttrain.data.records import iter_records as iter_i2rt_records
    manifest = {
        "schema_version": 1, "task": task, "description": cfg["task_description"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lerobot_version": cfg["lerobot_version"], "lerobot_format": "v3.0", "storage": "image",
        "fps": 20, "state_dim": 14, "action_dim": 7, "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES, "action_semantics": f"future_{action_source}_right_absolute",
        "state_source": "follower", "action_source": action_source,
        "source_fields": {"joints": "joints[name=joint0..joint5].pos", "gripper": "joints[name=gripper].pos"},
        "state_topics": [f"nmx/hal/i2rt/follower/{side}/{kind}" for side in ("left", "right") for kind in ("joint", "hand_state")],
        "action_topics": [f"nmx/hal/i2rt/{action_source}/right/{kind}" for kind in ("joint", "hand_state")],
        "action_offset_frames": 1, "action_horizon": cfg["action_horizon"],
        "max_source_age_ms": cfg["max_source_age_ms"], "image_size": cfg["image_size"],
        "image_transform": "BGR JPEG decode -> RGB resize preserving aspect + zero letterbox; no augmentation",
        "seed": cfg["seed"], "split": split, "source_root": "raw_data",
        "split_method": "explicit_validation_episodes" if cfg.get("validation_episodes") is not None else "stratified_episode_order",
        "review_sha256": file_hash(review_path), "approved_sha256": file_hash(approved_path),
        "notes": f"Follower observations; right-only future {action_source} absolute position labels.",
        "episodes": [],
    }
    writers = {}
    for partition in ("train", "val"):
        writers[partition] = LeRobotDataset.create(
            repo_id=f"yam_local/{task}_{partition}", root=stage / partition, fps=20,
            robot_type="i2rt_yam_right_arm_with_left_observer", features=features(cfg["image_size"]),
            use_videos=False, image_writer_threads=4, video_backend="pyav", metadata_buffer_size=1,
        )
    episode_numbers = Counter()
    try:
        for number, entry in enumerate(selected, 1):
            relative, path = entry["relative"], entry["path"]
            data = read_episode(path, iter_i2rt_records, cfg["max_source_age_ms"], action_source=action_source)
            rows = len(data["state"])
            window_count = len(legal_window_starts(rows, cfg["action_horizon"]))
            if not window_count:
                raise ValueError(f"Episode too short for the action horizon: {relative}")
            partition = split[relative]
            writer = writers[partition]
            cache = {}
            source_image_checks = {}
            for index in range(rows):
                frame = {"observation.state": data["state"][index], "action": data["action"][index],
                         "task": cfg["task_description"]}
                for side, key in IMAGE_KEYS.items():
                    source_index = int(data["image_indices"][side][index])
                    if cache.get(side, (-1, None))[0] != source_index:
                        jpeg = data["images"][side][source_index][1]
                        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        cache[side] = (source_index, prepare_rgb(image, cfg["image_size"]))
                    frame[key] = cache[side][1]
                    if index in (0, rows // 2, rows - 1):
                        source_image_checks[f"{side}:{index}"] = __import__("hashlib").sha256(frame[key].tobytes()).hexdigest()
                writer.add_frame(frame)
            writer.save_episode(parallel_encoding=False)
            trace_name = f"{partition}_{episode_numbers[partition]:04d}_{path.name}.npz"
            traces = {"state": data["state"], "action": data["action"], "full_state": data["full_state"],
                      "full_action_source": data["full_action_source"],
                      "timestamp_ns": data["timestamp_ns"], "action_timestamp_ns": data["action_timestamp_ns"]}
            traces.update({f"source_time_{key}": value for key, value in data["source_times"].items()})
            traces.update({f"action_source_time_{key}": value for key, value in data["action_source_times"].items()})
            traces.update({f"image_index_{key}": value for key, value in data["image_indices"].items()})
            from acttrain.model.velocity import augment_states
            traces["velocity_right"] = augment_states(data["state"])[:, 14:]
            np.savez_compressed(stage / "traces" / trace_name, **traces)
            manifest["episodes"].append({
                "source_relative": relative, "split": partition, "dataset_episode_index": episode_numbers[partition],
                "rows": rows, "legal_action_windows": window_count, "trace": f"traces/{trace_name}",
                "max_source_age_ms": data["max_source_age_ms"], "image_rgb_sha256_checks": source_image_checks,
                "source_files": entry["review"]["source_files"], "review_notes": entry["review"].get("notes", ""),
                "metadata_sha256": file_hash(path / "metadata.json"), "stats_sha256": file_hash(path / "episode_stats.json"),
                "initial_state": data["state"][0].tolist(),
                "left_max_deviation_rad": np.max(np.abs(data["state"][:, :6] - data["state"][0, :6]), axis=0).tolist(),
            })
            episode_numbers[partition] += 1
            write_json(stage / "manifest.json", manifest)
            print(f"{number:02}/{len(selected)} {path.name} -> {partition}: {rows} rows, {window_count} valid chunks", flush=True)
    finally:
        for writer in writers.values():
            writer.finalize()
            writer.stop_image_writer()
    # Policies must use the train statistics, including when evaluating val.
    finalize_normalization(stage, manifest)
    manifest["counts"] = {partition: {
        "episodes": sum(e["split"] == partition for e in manifest["episodes"]),
        "rows": sum(e["rows"] for e in manifest["episodes"] if e["split"] == partition),
        "valid_chunks": sum(e["legal_action_windows"] for e in manifest["episodes"] if e["split"] == partition),
    } for partition in ("train", "val")}
    write_json(stage / "manifest.json", manifest)
    from acttrain.data.validate import validate
    validate(stage)
    write_json(stage / "status.json", {"status":"completed","stage":"validated"})
    stage.rename(output)
    print(f"Exported: {output}\n{manifest['counts']}", flush=True)
    return output



def convert(raw_root, config_path, output_override=None):
    """Only a failed attempt that created a staging directory may update its status."""
    import uuid
    cfg = read_json(config_path)
    from acttrain.cli.preprocess import validate_config
    validate_config(cfg)
    output = Path(output_override).resolve() if output_override else (Path(raw_root) / cfg["output"]).resolve()
    stage = output.with_name(output.name + ".partial")
    if output.exists() or stage.exists():
        raise FileExistsError("已有数据集或partial目录，拒绝覆盖")
    attempt = uuid.uuid4().hex
    try:
        return _convert(raw_root, config_path, output, attempt)
    except BaseException as error:
        status = stage / "status.json"
        if status.is_file() and read_json(status).get("attempt") == attempt:
            write_json(status, {"status":"failed", "attempt":attempt, "error":str(error)})
        raise
