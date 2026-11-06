"""Data contracts; no model or hardware imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STATE_NAMES = [f"left_joint{i}" for i in range(6)] + ["left_gripper"] + [f"right_joint{i}" for i in range(6)] + ["right_gripper"]
ACTION_NAMES = STATE_NAMES[7:14]
IMAGE_KEYS = {side: f"observation.images.{side}_wrist" for side in ("left", "right")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_statistics(values, std_floor=1e-3):
    """Stable population statistics; E[x^2]-E[x]^2 in float32 loses static axes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all() or std_floor <= 0:
        raise ValueError("Expected finite nonempty numerical rows and a positive std floor")
    return {"mean": values.mean(axis=0).tolist(),
            "std": np.maximum(values.std(axis=0), std_floor).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "count": [len(values)]}


def finalize_normalization(root, manifest):
    """Replace only train numerical moments; preserve official image/quantile stats."""
    import shutil
    root = Path(root)
    stats_path = root / "train/meta/stats.json"
    original_path = root / "provenance/train_stats_unclamped.json"
    if not original_path.exists():
        shutil.copy2(stats_path, original_path)
    stats = read_json(original_path)
    training = [entry for entry in manifest["episodes"] if entry["split"] == "train"]
    for key, trace_key in (("observation.state", "state"), ("action", "action")):
        arrays = []
        for entry in training:
            with np.load(root / entry["trace"], allow_pickle=False) as trace:
                arrays.append(trace[trace_key])
        stats[key].update(numeric_statistics(np.concatenate(arrays), std_floor=1e-3))
    write_json(stats_path, stats)
    write_json(root / "normalization_train.json", stats)
    manifest["normalization_std_floor"] = 1e-3
    manifest["normalization_numeric_computation"] = "float64 population moments of train rows only"


def causal_indices(times, targets, max_age_ns):
    times, targets = np.asarray(times, np.int64), np.asarray(targets, np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must be strictly increasing and nonempty")
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0) or np.any(targets - times[np.maximum(indices, 0)] > max_age_ns):
        raise ValueError("Missing or stale observation; do not bridge gaps")
    return indices


def shift_right_actions(state):
    state = np.asarray(state, np.float32)
    if state.ndim != 2 or state.shape[1] != 14 or len(state) < 2 or not np.isfinite(state).all():
        raise ValueError("Expected finite (N,14) follower states with N>=2")
    return state[:-1].copy(), state[1:, 7:14].copy()


def legal_window_starts(rows, horizon):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    return range(max(0, rows - horizon + 1))


def split_episodes(names, seed=42, val_fraction=.2, validation_episodes=None):
    if len(names) < 2 or len(set(names)) != len(names) or not 0 < val_fraction < 1:
        raise ValueError("Need unique episodes, >=2, and a valid validation fraction")
    if validation_episodes is not None:
        if not isinstance(validation_episodes, list) or not all(isinstance(name, str) for name in validation_episodes):
            raise ValueError("validation_episodes must be a list of approved episode paths")
        val = set(validation_episodes)
        if (len(val) != len(validation_episodes) or not val or len(val) >= len(names)
                or not val.issubset(names)):
            raise ValueError("Explicit validation episodes must be unique approved episodes and leave both splits nonempty")
    else:
        import random
        ranked = sorted(names)
        count = min(len(names) - 1, max(1, round(len(names) * val_fraction)))
        rng = random.Random(seed)
        val = {ranked[rng.randrange(round(i*len(names)/count), round((i+1)*len(names)/count))] for i in range(count)}
    return {name: "val" if name in val else "train" for name in sorted(names)}


def prepare_rgb(bgr, size=256):
    import cv2
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Invalid three-channel source image")
    h, w = bgr.shape[:2]
    scale = min(size / h, size / w)
    resized = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


def load_selection(data_root, approved_file, review_file):
    data_root = Path(data_root).resolve()
    approved, review = read_json(approved_file), read_json(review_file)
    if not isinstance(approved.get("source_fingerprint"),str) or not approved["source_fingerprint"] or not approved.get("task"):
        raise ValueError("审核记录缺少task或source_fingerprint")
    if approved.get("task") != review.get("task") or approved.get("source_fingerprint") != review.get("source_fingerprint"):
        raise ValueError("Approved list and review identify different datasets")
    if approved.get("pending_count", 0):
        raise ValueError("Use a final review with no pending decisions")
    selected = approved.get("approved_episodes", [])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Approved Episode list is empty or duplicated")
    records = review.get("episodes", [])
    by_path = {entry.get("source_relative"): entry for entry in records}
    if len(by_path) != len(records):
        raise ValueError("Duplicate source paths in review")
    actual_approved = {p for p, entry in by_path.items() if entry.get("decision") == "approved"}
    if set(selected) != actual_approved:
        raise ValueError("Approved list does not exactly match human decisions")
    if any(entry.get("decision") not in ("approved", "rejected") for entry in records):
        raise ValueError("Review still has pending/unsure entries")
    result = []
    for relative in sorted(selected):
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or not path.is_dir() or not path.name.startswith("episode_"):
            raise ValueError(f"Unsafe or missing Episode path: {relative}")
        entry = by_path[relative]
        if entry.get("technical_status") == "fail":
            raise ValueError(f"Human approved a technical failure: {relative}")
        if not (path / "data").resolve().is_relative_to(path):
            raise ValueError(f"MCAP数据目录越出episode边界: {relative}")
        expected = entry.get("source_files", [])
        actual_names = sorted(p.name for p in (path / "data").glob("*.mcap"))
        if not expected or sorted(item["name"] for item in expected) != actual_names:
            raise ValueError(f"MCAP inventory differs from reviewed files: {relative}")
        for item in expected:
            candidate = (path / "data" / item["name"]).resolve()
            if candidate.parent != (path / "data").resolve() or file_hash(candidate) != item["sha256"]:
                raise ValueError(f"Source fingerprint changed: {relative}")
        result.append({"relative": relative, "path": path, "review": entry})
    return approved["task"], result


def read_episode(path, records, max_age_ms=100, action_source="leader"):
    """Return unnormalized causal 20Hz observations and raw compressed wrist frames."""
    if action_source not in ("follower", "leader"):
        raise ValueError("action_source must be follower or leader")
    path = Path(path)
    metadata, stats = read_json(path / "metadata.json"), read_json(path / "episode_stats.json")
    if metadata.get("extra", {}).get("contract_version", metadata.get("contract_version")) != "i2rt-nedf3-v2":
        raise ValueError("Only verified RGB I2RT NEDF3 v2 is supported")
    if stats.get("writer_error") or stats.get("dropped_batches", 0):
        raise ValueError("Source episode reports recording errors")
    numeric = {f"{side}_{kind}": [] for side in ("left", "right") for kind in ("joints", "gripper")}
    if action_source == "leader":
        numeric.update({f"leader_right_{kind}": [] for kind in ("joints", "gripper")})
    images = {side: [] for side in ("left", "right")}
    for record in records(path):
        if record.kind == "camera" and record.role in images:
            images[record.role].append((record.timestamp_ns, record.image))
        elif record.role == "follower" and record.kind in ("joints", "gripper"):
            numeric[f"{record.side}_{record.kind}"].append((record.timestamp_ns, record.values))
        elif action_source == "leader" and record.role == "leader" and record.side == "right" and record.kind in ("joints", "gripper"):
            numeric[f"leader_right_{record.kind}"].append((record.timestamp_ns, record.values))
    streams = {**numeric, **images}
    if any(not entries for entries in streams.values()):
        raise ValueError("Required follower/wrist/action-source stream is missing")
    times = {key: np.array([row[0] for row in entries], np.int64) for key, entries in streams.items()}
    start, end = max(t[0] for t in times.values()), min(t[-1] for t in times.values())
    if end <= start:
        raise ValueError("No common source time interval")
    grid = np.arange(start, end + 1, 50_000_000, dtype=np.int64)
    indices = {key: causal_indices(t, grid, round(max_age_ms * 1e6)) for key, t in times.items()}
    order = ("left_joints", "left_gripper", "right_joints", "right_gripper")
    state = np.concatenate([np.asarray([numeric[key][i][1] for i in indices[key]], np.float32) for key in order], axis=1)
    if np.any((state[:, [6, 13]] < 0) | (state[:, [6, 13]] > 1)):
        raise ValueError("Gripper values outside [0,1]")
    observations, actions = shift_right_actions(state)
    full_action_source = state[:, 7:14].copy()
    if action_source == "leader":
        keys = ("leader_right_joints", "leader_right_gripper")
        full_action_source = np.concatenate([
            np.asarray([numeric[key][i][1] for i in indices[key]], np.float32) for key in keys], axis=1)
        if (full_action_source.shape != (len(grid), 7) or not np.isfinite(full_action_source).all()
                or np.any((full_action_source[:, 6] < 0) | (full_action_source[:, 6] > 1))):
            raise ValueError("Invalid leader joint/gripper values")
        actions = full_action_source[1:].copy()
    ages = {key: (grid - times[key][indices[key]]) / 1e6 for key in streams}
    return {"state": observations, "action": actions, "full_state": state,
            "full_action_source": full_action_source,
            "timestamp_ns": grid[:-1], "action_timestamp_ns": grid[1:],
            "source_times": {key: times[key][indices[key]][:-1] for key in streams},
            "action_source_times": {kind: times[f"leader_right_{kind}" if action_source == "leader" else f"right_{kind}"][
                indices[f"leader_right_{kind}" if action_source == "leader" else f"right_{kind}"]][1:]
                for kind in ("joints", "gripper")},
            "image_indices": {side: indices[side][:-1] for side in images}, "images": images,
            "max_source_age_ms": {key: float(value.max()) for key, value in ages.items()}}
