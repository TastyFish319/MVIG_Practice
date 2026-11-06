"""Find the checkpoints of a task (folders containing a complete pretrained_model)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from act_deploy.config import Task
from act_deploy.model.spec import CHECKPOINT_FILES, OBS_CONTRACT, ACTION_CONTRACT


@dataclass
class CheckpointInfo:
    name: str  # path relative to the task folder, e.g. v3_long2_017000
    path: Path  # the pretrained_model directory
    step: int | None
    mtime: float
    size_mb: float
    contracts: str  # e.g. "obs+act"

    def describe(self) -> str:
        step = f"step {self.step}" if self.step is not None else "step ?"
        date = time.strftime("%m-%d %H:%M", time.localtime(self.mtime))
        return f"{self.name:<28} {step:<12} {date}  {self.size_mb:5.0f}MB  契约:{self.contracts}"


def _is_ckpt(d: Path) -> bool:
    return all((d / f).is_file() for f in CHECKPOINT_FILES)


def _step(run_dir: Path, ckpt: Path) -> int | None:
    for f in (run_dir / "training_step.json", run_dir / "training_state" / "training_step.json"):
        try:
            return int(json.loads(f.read_text())["step"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    digits = "".join(ch for ch in run_dir.name.split("_")[-1] if ch.isdigit())
    return int(digits) if digits else None


def scan_checkpoints(task: Task, max_depth: int = 4) -> list[CheckpointInfo]:
    found: list[CheckpointInfo] = []
    seen = set()
    for root in task.checkpoint_dirs:
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            depth = len(d.relative_to(root).parts)
            if depth > max_depth:
                dirnames.clear()
                continue
            if not _is_ckpt(d):
                continue
            dirnames.clear()
            key = d.resolve()
            if key in seen:
                continue
            seen.add(key)
            run_dir = d.parent if d.name == "pretrained_model" else d
            rel = run_dir.relative_to(root).as_posix() if run_dir != root else root.name
            contracts = "+".join(n for n, f in (("obs", OBS_CONTRACT), ("act", ACTION_CONTRACT))
                                 if (d / f).is_file()) or "无"
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e6
            found.append(CheckpointInfo(rel if len(task.checkpoint_dirs) == 1 else f"{root.name}/{rel}",
                                        d, _step(run_dir, d), (d / "model.safetensors").stat().st_mtime,
                                        size, contracts))
    found.sort(key=lambda c: (c.name.rsplit("_", 1)[0], c.step or 0))
    return found
