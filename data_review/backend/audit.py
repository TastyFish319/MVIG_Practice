"""Read-only NEDF3 audit. Outputs are derivatives; no source files are changed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2
import imageio_ffmpeg
import numpy as np
from mcap.reader import make_reader


def causal_index(times, target):
    """Return an observed frame, never a future frame; -1 means unavailable."""
    return int(np.searchsorted(times, target, side="right")) - 1


def stream_stats(times):
    values = np.asarray(times, dtype=np.int64)
    diff = np.diff(values)
    duration = (int(values[-1]) - int(values[0])) / 1e9 if len(values) > 1 else 0
    return {
        "count": len(values), "duration_s": duration,
        "duplicate_timestamps": int(np.sum(diff == 0)),
        "backward_timestamps": int(np.sum(diff < 0)),
        "max_gap_ms": float(np.max(diff) / 1e6) if len(diff) else None,
        "gaps_over_100ms": int(np.sum(diff > 100_000_000)),
        "effective_hz": (len(values) - 1) / duration if duration > 0 else None,
    }


def fit(image, width=480, height=360):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def video_preview(frames, destination, episode, fps=15):
    if not all(frames.get(role) for role in ("left", "right")):
        raise ValueError("Cannot preview without both wrist streams")
    ordered = {role: sorted(frames[role], key=lambda x: x[0]) for role in ("left", "right")}
    times = {role: np.asarray([item[0] for item in rows], np.int64) for role, rows in ordered.items()}
    start = max(int(t[0]) for t in times.values())
    end = min(int(t[-1]) for t in times.values())
    if end <= start:
        raise ValueError("No overlapping camera time range")
    count = int(np.floor((end - start) / 1e9 * fps)) + 1
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    log_path = destination.with_suffix(".ffmpeg.log")
    with log_path.open("wb") as errors:
        proc = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", "960x416", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        selected = {}
        max_age = 0.0
        poster_at = int(count * 0.45)
        try:
            for k in range(count):
                target = start + round(k * 1e9 / fps)
                canvas = np.full((416, 960, 3), (20, 25, 30), np.uint8)
                for column, role in enumerate(("left", "right")):
                    index = causal_index(times[role], target)
                    if index < 0:
                        raise ValueError("Preview attempted to use a future image")
                    if selected.get(role, (-1, None))[0] != index:
                        decoded = cv2.imdecode(np.frombuffer(ordered[role][index][1], np.uint8), cv2.IMREAD_COLOR)
                        selected[role] = (index, fit(decoded))
                    age_ms = (target - int(times[role][index])) / 1e6
                    max_age = max(max_age, age_ms)
                    canvas[40:400, column * 480:(column + 1) * 480] = selected[role][1]
                    label = f"{role.upper()} WRIST | frame age {age_ms:.0f} ms"
                    cv2.putText(canvas, label, (column * 480 + 10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                                0.59, (90, 210, 245) if age_ms <= 100 else (70, 70, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{episode}  t={k/fps:.2f}s  |  review only - not training data",
                            (10, 413), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                if k == poster_at:
                    if not cv2.imwrite(str(destination.with_suffix(".jpg")), canvas):
                        raise OSError("Could not write thumbnail")
                proc.stdin.write(canvas.tobytes())
        finally:
            proc.stdin.close()
            exit_code = proc.wait(timeout=120)
        if exit_code:
            raise RuntimeError(f"FFmpeg failed ({exit_code}); see {log_path.name}")
    return {"fps": fps, "frames": count, "duration_s": count / fps,
            "camera_common_start_ns": start, "camera_common_end_ns": end,
            "max_frame_age_ms": max_age, "causal_alignment": True,
            "path": f"media/{destination.name}", "poster": f"media/{destination.with_suffix('.jpg').name}"}


def inspect(episode, output, reader, *, task_id, source_relative):
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    recorded = json.loads((episode / "episode_stats.json").read_text(encoding="utf-8"))
    extra = metadata.get("extra", {})
    version = extra.get("contract_version", metadata.get("contract_version"))
    if version != "i2rt-nedf3-v2":
        raise ValueError(f"Expected RGB v2 contract, got {version!r}")
    decoder = reader._EmbeddedProtobufDecoder()
    timestamps = defaultdict(list)
    numeric = defaultdict(list)
    frames = defaultdict(list)
    shapes = defaultdict(Counter)
    counts = Counter()
    camera_errors = Counter()
    schemas = set()
    source_hashes = []
    for shard in reader.episode_mcap_paths(episode):
        digest = hashlib.sha256()
        with shard.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        source_hashes.append({"name": shard.name, "bytes": shard.stat().st_size, "sha256": digest.hexdigest()})
        with shard.open("rb") as source:
            for schema, channel, message in make_reader(source, validate_crcs=True).iter_messages(log_time_order=False):
                topic, time_ns = str(channel.topic), int(message.log_time)
                counts[topic] += 1
                timestamps[topic].append(time_ns)
                if topic == reader.NEDF3_METADATA_TOPIC:
                    continue
                if topic not in reader.CAMERA_TOPIC_INFO and topic not in reader.JOINT_TOPIC_INFO and topic not in reader.HAND_TOPIC_INFO:
                    continue
                decoded = decoder.decode(schema, message.data)
                schemas.add((schema.name, schema.encoding))
                if topic in reader.CAMERA_TOPIC_INFO:
                    role = reader.CAMERA_TOPIC_INFO[topic]
                    image_bytes = bytes(decoded.rgb.data)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
                    if image is None:
                        camera_errors[role] += 1
                        continue
                    shapes[role][str(tuple(image.shape))] += 1
                    if image.shape[:2] != (int(decoded.rgb.rows), int(decoded.rgb.cols)):
                        camera_errors[f"{role}_size_mismatch"] += 1
                    if role in ("left", "right"):
                        frames[role].append((time_ns, image_bytes))
                elif topic in reader.JOINT_TOPIC_INFO:
                    role, side = reader.JOINT_TOPIC_INFO[topic]
                    values = reader._joint_values(decoded, topic)
                    numeric[f"{role}_{side}_joints"].append((time_ns, values))
                else:
                    role, side = reader.HAND_TOPIC_INFO[topic]
                    value = reader._gripper_value(decoded, topic)
                    numeric[f"{role}_{side}_gripper"].append((time_ns, (value,)))
    streams = {topic: stream_stats(times) for topic, times in timestamps.items()}
    failures, warnings = [], []
    for topic in reader.NEDF3_RECORD_TOPICS:
        if not counts[topic]:
            failures.append(f"Missing canonical topic: {topic}")
    for topic, info in streams.items():
        if info["duplicate_timestamps"] or info["backward_timestamps"]:
            failures.append(f"Non-increasing source timestamps: {topic}")
        if info["gaps_over_100ms"]:
            warnings.append(f"Source gaps >100ms ({info['gaps_over_100ms']}): {topic}")
    if recorded.get("writer_error") or recorded.get("dropped_batches", 0):
        failures.append("Recorder reported writer errors or dropped batches")
    if camera_errors:
        failures.append(f"Image decoding or declared size errors: {dict(camera_errors)}")
    mismatches = {topic: {"stats": expected, "mcap": counts[topic]}
                  for topic, expected in recorded.get("topic_counts", {}).items() if expected != counts[topic]}
    if mismatches:
        failures.append("MCAP message counts differ from episode_stats.json")
    motion = {}
    arrays = {}
    for key, entries in numeric.items():
        values = np.asarray([row[1] for row in entries], dtype=np.float64)
        times = np.asarray([row[0] for row in entries], dtype=np.int64)
        if not np.isfinite(values).all():
            failures.append(f"Non-finite state: {key}")
        if key.endswith("gripper") and np.any((values < -1e-6) | (values > 1 + 1e-6)):
            failures.append(f"Gripper outside [0,1]: {key}")
        arrays[key] = values
        arrays[key + "_time_ns"] = times
        motion[key] = {
            "start": values[0].tolist(), "end": values[-1].tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "range": np.ptp(values, axis=0).tolist(),
            "max_abs_delta_from_start": np.max(np.abs(values - values[0]), axis=0).tolist(),
        }
    left_deviation = max(motion.get("follower_left_joints", {}).get("max_abs_delta_from_start", [0]))
    if left_deviation > 0.1:
        warnings.append("Left follower deviation >0.1 rad on at least one joint; review viewpoint (heuristic, not safety limit)")
    required = [timestamps[reader.NEDF3_CAMERA_TOPICS[side]] for side in ("left", "right")]
    required += [arrays[f"follower_{side}_{kind}_time_ns"] for side in ("left", "right") for kind in ("joints", "gripper")]
    alignment = {"grid_hz": 20, "max_age_ms": 100}
    if all(len(t) for t in required):
        sorted_times = [np.sort(np.asarray(t, dtype=np.int64)) for t in required]
        begin, end = max(int(t[0]) for t in sorted_times), min(int(t[-1]) for t in sorted_times)
        if end > begin:
            grid = np.arange(begin, end + 1, 50_000_000, dtype=np.int64)
            ages = np.stack([(grid - t[np.searchsorted(t, grid, side="right") - 1]) / 1e6 for t in sorted_times])
            bad_frames = np.any(ages > 100, axis=0)
            alignment.update({"frames": len(grid), "stale_grid_frames": int(bad_frames.sum()),
                              "max_age_observed_ms": float(ages.max()), "duration_s": (end - begin) / 1e9})
            if bad_frames.any():
                failures.append(f"20Hz causal alignment has {int(bad_frames.sum())} stale observations >100ms")
        else:
            failures.append("Required streams have no common interval")
    else:
        failures.append("Required follower/wrist stream is empty")
    np.savez_compressed(output / "numeric" / f"{episode.name}.npz", **arrays)
    preview = video_preview(frames, output / "media" / f"{episode.name}.mp4", episode.name)
    result = {
        "episode": episode.name, "source_relative": source_relative,
        "canonical_task": task_id, "recording_duration_s": recorded.get("duration_s"),
        "contract_version": version, "mcap_crc_validation": True, "source_files": source_hashes,
        "streams": streams, "camera_shapes": {key: dict(value) for key, value in shapes.items()},
        "decoded_image_errors": dict(camera_errors), "count_mismatches": mismatches,
        "motion": motion, "alignment": alignment, "preview": preview,
        "technical_status": "fail" if failures else ("warning" if warnings else "pass"),
        "failures": failures, "warnings": warnings,
        "human_decision": "pending", "notes": "", "task_success": "not_assessed",
    }
    return result


if __name__ == "__main__":
    raise SystemExit("Use review.py --task <directory> to generate a reusable review.")
