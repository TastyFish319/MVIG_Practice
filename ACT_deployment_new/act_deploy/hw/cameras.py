"""Wrist cameras.  RealSense D405 colour stream rgb8 640x480@30 (same as the SI recorder).

Each camera has a background thread keeping only the newest frame + its arrival time
(time.monotonic()).  grab() refuses frames older than max_age_s.
FakeCameraRig produces synthetic frames for sim mode.
"""

from __future__ import annotations

import importlib
import threading
import time

import numpy as np


class _Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.count = 0
        self.fps = 0.0
        self._fps_t = time.monotonic()
        self._fps_n = 0

    def put(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self.lock:
            self.frame, self.t = frame, now
            self.count += 1
            self._fps_n += 1
            if now - self._fps_t >= 1.0:
                self.fps = self._fps_n / (now - self._fps_t)
                self._fps_t, self._fps_n = now, 0

    def get(self):
        with self.lock:
            return self.t, self.frame, self.count


class RealSenseCamera:
    def __init__(self, name: str, serial: str, width: int, height: int, fps: int):
        rs = importlib.import_module("pyrealsense2")
        self.name, self.serial = name, serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(cfg)
        for _ in range(10):  # let auto exposure settle
            self.pipeline.wait_for_frames(5000)
        self.latest = _Latest()
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{name}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError as exc:
                self.error = str(exc)
                continue
            color = frames.get_color_frame()
            if color:
                self.latest.put(np.asanyarray(color.get_data()).copy())
                self.error = None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


class FakeCamera:
    """Synthetic 30 fps frames: grey table + a coloured square drifting slowly."""

    def __init__(self, name: str, width: int, height: int, fps: int):
        self.name, self.serial, self.error = name, "fake", None
        self.w, self.h, self.period = width, height, 1.0 / fps
        self.latest = _Latest()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{name}")
        self._thread.start()

    def _loop(self) -> None:
        rng = np.random.default_rng(0)
        base = np.full((self.h, self.w, 3), 110, np.uint8)
        k = 0
        while not self._stop.wait(self.period):
            img = base.copy()
            x = int((np.sin(k / 60) * 0.3 + 0.5) * (self.w - 80))
            img[200:280, x:x + 80] = (40, 60, 200)
            img[:: 8] = rng.integers(100, 120, size=(img[::8].shape), dtype=np.uint8)
            self.latest.put(img)
            k += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class CameraRig:
    def __init__(self, cams: dict):
        self.cams = cams

    @classmethod
    def realsense(cls, serials: dict[str, str], width: int, height: int, fps: int) -> "CameraRig":
        cams = {}
        try:
            for name, serial in serials.items():
                cams[name] = RealSenseCamera(name, str(serial), width, height, fps)
        except Exception:
            for c in cams.values():
                c.close()
            raise
        return cls(cams)

    @classmethod
    def fake(cls, names, width: int, height: int, fps: int) -> "CameraRig":
        return cls({n: FakeCamera(n, width, height, fps) for n in names})

    def wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while any(c.latest.get()[1] is None for c in self.cams.values()):
            if time.monotonic() > deadline:
                bad = [n for n, c in self.cams.items() if c.latest.get()[1] is None]
                raise TimeoutError(f"相机无画面: {bad}")
            time.sleep(0.05)

    def grab(self, names, max_age_s: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        now = time.monotonic()
        frames, times = {}, {}
        for n in names:
            if n not in self.cams:
                raise RuntimeError(f"模型需要相机 {n!r}，但 deploy.yaml 没有配置它")
            t, f, _ = self.cams[n].latest.get()
            if f is None or now - t > max_age_s:
                raise RuntimeError(f"相机 {n} 画面过期 ({now - t:.2f}s)")
            frames[n], times[n] = f, t
        return frames, times

    def status(self) -> dict[str, dict]:
        now = time.monotonic()
        out = {}
        for n, c in self.cams.items():
            t, f, count = c.latest.get()
            out[n] = {"fps": c.latest.fps, "age_s": (now - t) if f is not None else None, "error": c.error}
        return out

    def close(self) -> None:
        for c in self.cams.values():
            c.close()
