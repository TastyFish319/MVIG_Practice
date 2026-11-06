"""Per-episode recording: 100 Hz command stream + every inference (npz + json)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: Path, index: int, meta: dict, save_images: bool):
        self.out_dir, self.index, self.meta = Path(out_dir), index, dict(meta)
        self.save_images = save_images
        self.ctrl: dict[str, list] = {"t": [], "state": [], "target": [], "cmd": []}
        self.inf: dict[str, list] = {"t_req": [], "t_obs": [], "t_done": [], "state_vec": [], "raw": [],
                                     "points": [], "infer_ms": [], "velocity_padded": []}
        self.images: dict[str, list] = {}
        self.t_start = time.monotonic()
        self.wall_start = time.time()

    def control(self, t, state, target, cmd) -> None:
        self.ctrl["t"].append(t)
        self.ctrl["state"].append(np.asarray(state, np.float32))
        self.ctrl["target"].append(np.full(14, np.nan, np.float32) if target is None else np.asarray(target, np.float32))
        self.ctrl["cmd"].append(np.asarray(cmd, np.float32))

    def inference(self, t_req, t_obs, t_done, state_vec, raw, points, infer_ms, velocity_padded, images=None) -> None:
        for k, v in (("t_req", t_req), ("t_obs", t_obs), ("t_done", t_done), ("state_vec", state_vec),
                     ("raw", raw), ("points", points), ("infer_ms", infer_ms),
                     ("velocity_padded", velocity_padded)):
            self.inf[k].append(v)
        if self.save_images and images:
            for k, img in images.items():
                self.images.setdefault(k, []).append(img)

    def close(self, reason: str) -> Path | None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"episode_{self.index:03d}"
        arrays = {}
        for prefix, d in (("ctrl_", self.ctrl), ("inf_", self.inf)):
            for k, v in d.items():
                if v:
                    arrays[prefix + k] = np.asarray(v)
        for k, v in self.images.items():
            arrays["img_" + k.replace(".", "_")] = np.asarray(v, np.uint8)
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        lat = np.asarray(self.inf["t_done"]) - np.asarray(self.inf["t_req"]) if self.inf["t_done"] else np.zeros(0)
        summary = {**self.meta, "end_reason": reason, "duration_s": round(time.monotonic() - self.t_start, 2),
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall_start)),
                   "n_control": len(self.ctrl["t"]), "n_inference": len(self.inf["t_req"]),
                   "latency_ms_median": float(np.median(lat) * 1000) if lat.size else None,
                   "latency_ms_max": float(lat.max() * 1000) if lat.size else None}
        base.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return base.with_suffix(".npz")
