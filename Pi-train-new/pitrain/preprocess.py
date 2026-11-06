"""预处理核心：approved episode -> 对齐/裁剪/缩放 -> LeRobot v3 数据集（train/ 与 val/）。

数据流::

    raw_data/<任务>/episode_xxxx/*.mcap
        │  nedf3.read_episode       读出各 topic 原始时间序列
        │  timeseries.align         30Hz 因果对齐 + 速度 + 首尾静止裁剪
        │  images.letterbox         640x480 JPEG -> 224x224 RGB
        ▼
    dataset/<名称>/
        train/   LeRobot 数据集（parquet + 内嵌 PNG），归一化统计只来自这里
        val/     LeRobot 数据集（统计不使用）
        pitrain_meta.json   契约：state/action 各维含义、fps、相机、提示词、划分、夹爪画像
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .gripper import estimate_profile
from .images import decode_jpeg_rgb, letterbox
from .nedf3 import read_episode
from .review import ApprovedEpisode
from .timeseries import Layout, align

META_FILE = "pitrain_meta.json"
SCHEMA_VERSION = 1


@dataclass
class PreprocessConfig:
    raw_folder: str
    output: str
    prompt: str
    val_episodes: list[str]
    fps: int = 30
    image_size: int = 224
    cameras: list[str] = field(default_factory=lambda: ["right_wrist", "left_wrist"])
    action_arms: list[str] = field(default_factory=lambda: ["right"])
    max_age_ms: float = 100.0
    velocity_window_s: float = 0.1
    trim_idle: bool = True
    idle_joint_tol_rad: float = 0.02
    idle_grip_tol: float = 0.05
    idle_margin_s: float = 0.5
    workers: int = 8
    seed: int = 42
    val_ratio: float = 0.2


def image_key(cam: str) -> str:
    return f"observation.images.{cam}"


def features(cfg: PreprocessConfig, layout: Layout) -> dict:
    s = len(layout.state_names)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (s,), "names": layout.state_names},
        "observation.velocity": {"dtype": "float32", "shape": (s,), "names": layout.velocity_names},
        "action": {"dtype": "float32", "shape": (len(layout.action_names),), "names": layout.action_names},
        # 主臂完整状态（按 state 顺序），只用于评测画图，不参与训练
        "observation.leader_state": {"dtype": "float32", "shape": (s,), "names": [f"leader_{n}" for n in layout.state_names]},
    }
    for cam in cfg.cameras:
        feats[image_key(cam)] = {"dtype": "image", "shape": (cfg.image_size, cfg.image_size, 3),
                                 "names": ["height", "width", "channels"]}
    return feats


def _process_episode(args: tuple) -> dict:
    """子进程：读取 + 对齐 + 解码缩放图像。返回可直接写入的数组。"""
    path, cfg_dict = args
    cfg = PreprocessConfig(**cfg_dict)
    layout = Layout(action_arms=tuple(cfg.action_arms))
    raw = read_episode(Path(path), cameras=tuple(cfg.cameras))
    al = align(raw, layout, cfg.fps, cfg.max_age_ms, cfg.velocity_window_s, cfg.trim_idle,
               cfg.idle_joint_tol_rad, cfg.idle_grip_tol, cfg.idle_margin_s)
    images = {}
    for cam in cfg.cameras:
        jpegs = raw.cameras[cam][1]
        cache: dict[int, np.ndarray] = {}
        frames = []
        for idx in al.frame_index[cam]:
            idx = int(idx)
            if idx not in cache:
                cache = {idx: letterbox(decode_jpeg_rgb(jpegs[idx]), cfg.image_size)}
            frames.append(cache[idx])
        images[cam] = np.stack(frames)
    repeated = {cam: float(1 - len(np.unique(al.frame_index[cam])) / len(al.frame_index[cam])) for cam in cfg.cameras}
    return {"name": Path(path).name, "state": al.state, "velocity": al.velocity, "action": al.action,
            "leader_full": al.leader_full, "images": images, "trim": al.trim, "raw_len": al.raw_len,
            "max_age_ms": al.max_age_ms, "repeated_frame_ratio": repeated}


def _bounded_imap(fn, items, workers: int):
    """有界并行：最多 workers*2 个任务在途，避免所有 episode 的图像同时堆在内存里。"""
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending, it = [], iter(items)
        for _ in range(workers * 2):
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(fn, nxt))
        while pending:
            result = pending.pop(0).result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(fn, nxt))
            yield result


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_preprocess(cfg: PreprocessConfig, episodes: list[ApprovedEpisode], task: str, review_files: tuple[Path, Path],
                   on_episode: Callable[[dict, str], None] | None = None) -> Path:
    import datasets as hf_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hf_datasets.disable_progress_bars()  # 避免刷屏，由我们自己的进度条显示
    out = Path(cfg.output)
    stage = out.with_name(out.name + ".partial")
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}（换个名字，或手动删除）")
    try:
        if stage.exists():
            shutil.rmtree(stage)  # 上次中断留下的半成品
        (stage / "review").mkdir(parents=True)
        for f in review_files:
            shutil.copy2(f, stage / "review" / Path(f).name)
    except OSError as exc:
        raise RuntimeError(f"无法准备输出目录 {stage}: {exc}") from exc

    layout = Layout(action_arms=tuple(cfg.action_arms))
    feats = features(cfg, layout)
    val_set = set(cfg.val_episodes)
    writers = {split: LeRobotDataset.create(
        repo_id=f"pitrain/{out.name}_{split}", root=stage / split, fps=cfg.fps, features=feats,
        robot_type="i2rt_yam_bimanual", use_videos=False, image_writer_threads=8)
        for split in ("train", "val")}

    records, grip_lead, grip_foll = [], [], []
    counters = {"train": 0, "val": 0}
    t0 = time.time()
    items = [(str(e.path), asdict(cfg)) for e in episodes]
    try:
        for res in _bounded_imap(_process_episode, items, cfg.workers):
            split = "val" if res["name"] in val_set else "train"
            w = writers[split]
            n = len(res["state"])
            for i in range(n):
                frame = {"observation.state": res["state"][i], "observation.velocity": res["velocity"][i],
                         "action": res["action"][i], "observation.leader_state": res["leader_full"][i],
                         "task": cfg.prompt}
                for cam in cfg.cameras:
                    frame[image_key(cam)] = res["images"][cam][i]
                w.add_frame(frame)
            w.save_episode()
            if split == "train":
                for k in range(len(cfg.action_arms)):
                    grip_lead.append(res["action"][:, 7 * k + 6])
                    grip_foll.append(res["state"][:, 7 * k + 6])
            rec = {"episode": res["name"], "split": split, "index_in_split": counters[split], "frames": n,
                   "seconds": round(n / cfg.fps, 2), "trim": list(res["trim"]), "raw_frames": res["raw_len"],
                   "max_source_age_ms": round(max(res["max_age_ms"].values()), 1),
                   "repeated_frame_ratio": {k: round(v, 4) for k, v in res["repeated_frame_ratio"].items()}}
            counters[split] += 1
            records.append(rec)
            if on_episode:
                on_episode(rec, split)
    finally:
        for w in writers.values():
            w.finalize()

    by_split = {s: [r for r in records if r["split"] == s] for s in ("train", "val")}
    profile = estimate_profile(grip_lead, grip_foll)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "task": task, "prompt": cfg.prompt, "fps": cfg.fps, "image_size": cfg.image_size,
        "cameras": cfg.cameras, "image_keys": [image_key(c) for c in cfg.cameras],
        "action_arms": cfg.action_arms, "state_arms": list(layout.state_arms),
        "state_names": layout.state_names, "action_names": layout.action_names,
        "velocity_names": layout.velocity_names,
        "contract": {
            "state": "follower（从臂）位置，动作臂在前",
            "velocity": f"follower 速度，因果最小二乘窗口 {cfg.velocity_window_s}s（rad/s；夹爪 1/s）",
            "action": "leader（主臂）绝对位置；训练时由 LeRobot 转为相对动作（关节相对 / 夹爪绝对）",
            "leader_state": "leader 全部臂位置，只用于评测画图",
            "time_alignment": f"{cfg.fps}Hz 网格，因果最近邻（zero-order hold），最大允许延迟 {cfg.max_age_ms}ms",
            "images": f"JPEG -> RGB -> 等比缩放 + 黑边 -> {cfg.image_size}x{cfg.image_size}",
        },
        "config": asdict(cfg),
        "split": {s: [r["episode"] for r in by_split[s]] for s in by_split},
        "counts": {s: {"episodes": len(by_split[s]), "frames": sum(r["frames"] for r in by_split[s])} for s in by_split},
        "gripper_profile": profile,
        "episodes": records,
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_json(stage / META_FILE, meta)
    _verify(stage, meta)
    stage.rename(out)
    return out


def _verify(root: Path, meta: dict) -> None:
    """用 LeRobot 官方读取器回读，确认数据集可用、帧数一致。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for split in ("train", "val"):
        ds = LeRobotDataset(f"pitrain/{split}", root=root / split)
        if ds.num_frames != meta["counts"][split]["frames"] or ds.num_episodes != meta["counts"][split]["episodes"]:
            raise RuntimeError(f"{split}: 回读帧数/episode 数不一致")
        item = ds[0]
        for key in meta["image_keys"]:
            if tuple(item[key].shape) != (3, meta["image_size"], meta["image_size"]):
                raise RuntimeError(f"{split}: 图像尺寸异常 {key} {tuple(item[key].shape)}")
        stats = ds.meta.stats or {}
        if "q01" not in stats.get("observation.state", {}):
            raise RuntimeError("统计中缺少分位数（pi0.5 的 QUANTILES 归一化需要 q01/q99）")
