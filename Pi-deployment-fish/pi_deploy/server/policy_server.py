"""Model server: the ONLY place that imports torch / lerobot.  Runs in the model environment.

    <model-python> -m pi_deploy.server.policy_server --checkpoint <pretrained_model> --port 8890

Lifecycle (reported by the "health" op, so the console can show progress):
    loading -> warming -> ready          (or "error" with a message)
Ops:  health | predict(state, img:<key>...) -> actions (horizon, action_dim), unnormalized,
      ABSOLUTE joint/gripper targets from Pi05Runner | shutdown
Never opens robot hardware.  Binds to loopback only.  Exits when its parent process dies.
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import sys
import threading
import time
import traceback

import numpy as np

from pi_deploy.ipc import recv_msg, send_msg
from pi_deploy.model.features import dummy_inputs
from pi_deploy.model.spec import ModelSpec, checkpoint_directory, resolve_spec


def log(msg: str) -> None:
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LeRobotBackend:
    """Training Pi05Runner, with independent local tokenizer and floor override."""

    def __init__(self, ckpt: str, device: str, spec: ModelSpec, tokenizer: str,
                 gripper_floor: str = "checkpoint", seed: int = 0):
        import importlib.metadata

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from pathlib import Path
        from pi_deploy.inference.inference import Pi05Runner
        from pi_deploy.inference.options import gripper_options

        version = importlib.metadata.version("lerobot")
        if version != "0.6.1":
            raise RuntimeError(f"需要 lerobot==0.6.1，当前 {version}；使用独立 Pi 环境")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("要求 CUDA 但不可用（不会静默改用 CPU）")
        tok = Path(tokenizer).expanduser().resolve()
        if not tok.is_dir() or not (tok / 'tokenizer_config.json').is_file():
            raise FileNotFoundError(f'本地 tokenizer 不完整: {tok}')
        gf = gripper_options(spec.contract, gripper_floor)
        self.runner = Pi05Runner(Path(ckpt), tok, device, gripper_filter=gf)
        self.seed = seed
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True  # fixed input shapes; tuned during warmup
        self.torch, self.device = torch, device
        self.spec = spec
        self.info = {"backend": "lerobot", "lerobot": version, "torch": torch.__version__,
                     "device": device,
                     "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "cpu",
                     "gripper_filter": gf, "seed": seed, "tokenizer": str(tok)}

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(np.asarray(state[:14], np.float32).copy())[None],
                 "task": [self.spec.prompt]}
        if self.spec.velocity_segments:
            batch['observation.velocity'] = torch.from_numpy(np.asarray(state[14:], np.float32).copy())[None]
        for im in self.spec.images:
            hwc = images[im.key]
            chw = torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).float().div_(255.0)
            batch[im.key] = chw[None]
        with torch.inference_mode():
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
            actions = self.runner.predict(batch, seed=self.seed)['filtered']
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
        return actions[0]


class MockBackend:
    """No torch: small slow sine deltas around the observation (UI / pipeline tests only)."""

    def __init__(self, spec: ModelSpec, latency_s: float = 0.015):
        self.spec, self.latency_s = spec, latency_s
        self.info = {"backend": "mock", "device": "cpu", "gpu": "none"}
        self._t0 = time.monotonic()

    def predict(self, state, images):
        time.sleep(self.latency_s)
        h, a = self.spec.horizon, self.spec.action_dim
        k = np.arange(h)[:, None] / self.spec.fps
        phase = time.monotonic() - self._t0
        out = np.zeros((h, a), np.float32)
        position_indices = list(self.spec.state_segments[0].indices)
        for d in range(a):
            t = self.spec.action_targets[d]
            out[:, d] = state[position_indices.index(t)]
            if t not in (6, 13):
                out[:, d] += 0.02 * np.sin(phase + k[:, 0] * 2.0 + d)
        return out


class Server:
    def __init__(self, args):
        self.args = args
        self.status = "loading"
        self.error: str | None = None
        self.backend = None
        self.spec: ModelSpec | None = None
        self.warmup_ms: list[float] = []
        self.lock = threading.Lock()  # one inference at a time
        self.stop = threading.Event()
        self.last_use = time.monotonic()
        self.n_predict = 0
        self.parent = os.getppid()

    # ------------------------------------------------------------------ model
    def load(self) -> None:
        try:
            ckpt = str(checkpoint_directory(self.args.checkpoint))
            import json
            self.spec = resolve_spec(ckpt, json.loads(self.args.model_overrides))
            log(f"checkpoint: {ckpt}")
            t0 = time.monotonic()
            if self.args.backend == "mock":
                self.backend = MockBackend(self.spec)
            else:
                self.backend = LeRobotBackend(ckpt, self.args.device, self.spec, self.args.tokenizer,
                                             self.args.gripper_floor, self.args.seed)
            log(f"模型已加载 {time.monotonic() - t0:.1f}s  {self.backend.info}")
            self.status = "warming"
            state, images = dummy_inputs(self.spec)
            for i in range(self.args.warmup):
                t = time.perf_counter()
                with self.lock:
                    out = self.backend.predict(state, images)
                ms = (time.perf_counter() - t) * 1000
                self.warmup_ms.append(round(ms, 2))
                if out.shape != (self.spec.horizon, self.spec.action_dim) or not np.isfinite(out).all():
                    raise RuntimeError(f"预热输出非法 {out.shape}")
                log(f"预热 {i + 1}/{self.args.warmup}: {ms:.1f} ms")
            self.status = "ready"
            log("READY")
        except Exception as exc:  # noqa: BLE001
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            log("加载失败:\n" + traceback.format_exc())

    def keep_warm(self) -> None:
        """GPU clocks drop when idle -> the first request after a pause is slow.  Run a dummy
        inference whenever nothing happened for keep_warm_s."""
        if self.status != "ready" or self.args.keep_warm_s <= 0:
            return
        if time.monotonic() - self.last_use < self.args.keep_warm_s:
            return
        state, images = dummy_inputs(self.spec)
        try:
            with self.lock:
                self.backend.predict(state, images)
        except Exception as exc:
            self.status, self.error = 'error', f'保温推理失败: {exc}'
            log(self.error)
        self.last_use = time.monotonic()

    def parent_gone(self) -> bool:
        if self.args.exit_with_parent and os.name == "posix" and os.getppid() != self.parent:
            log("父进程已退出 -> 关闭")
            return True
        return False

    # ------------------------------------------------------------------ protocol
    def health(self) -> dict:
        h = {"ok": True, "status": self.status, "error": self.error, "pid": os.getpid(),
             'instance_id': getattr(self.args, 'instance_id', ''),
             "warmup_ms": self.warmup_ms, "n_predict": self.n_predict}
        if self.backend is not None:
            h["backend"] = self.backend.info
        if self.spec is not None:
            s = self.spec
            h["spec"] = {"state_dim": s.state_dim, "action_dim": s.action_dim, "horizon": s.horizon,
                         "fps": s.fps, "images": [[i.key, i.height, i.width] for i in s.images],
                         "fingerprint": s.fingerprint, "output_semantics": "absolute_filtered"}
        return h

    def handle(self, conn: socket.socket) -> None:
        while not self.stop.is_set():
            ready, _, _ = select.select([conn], [], [], 0.1)
            if not ready:
                if self.parent_gone():
                    self.stop.set()
                    break
                self.keep_warm()
                continue
            header, arrays = recv_msg(conn)
            op = header.get("op")
            if op == "health":
                send_msg(conn, self.health())
            elif op == "predict":
                if self.status != "ready":
                    send_msg(conn, {"ok": False, "error": f"model not ready ({self.status})"})
                    continue
                try:
                    images = {k[4:]: v for k, v in arrays.items() if k.startswith("img:")}
                    self._check(arrays["state"], images)
                    t = time.perf_counter()
                    with self.lock:
                        actions = self.backend.predict(arrays["state"], images)
                    ms = (time.perf_counter() - t) * 1000
                    self.last_use = time.monotonic()
                    self.n_predict += 1
                    if actions.shape != (self.spec.horizon, self.spec.action_dim) or not np.isfinite(actions).all():
                        raise ValueError('模型输出形状或数值非法')
                    send_msg(conn, {"ok": True, "infer_ms": ms, "request_id": header.get('request_id')},
                             {"actions": np.asarray(actions, np.float32)})
                except (ValueError, KeyError, RuntimeError) as exc:
                    send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            elif op == "shutdown":
                send_msg(conn, {"ok": True})
                self.stop.set()
            else:
                send_msg(conn, {"ok": False, "error": f"unknown op {op!r}"})

    def _check(self, state, images) -> None:
        s = self.spec
        if state.shape != (s.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"state 形状应为 ({s.state_dim},)，收到 {state.shape}")
        for im in s.images:
            img = images.get(im.key)
            if img is None or img.shape != (im.height, im.width, 3) or img.dtype != np.uint8:
                raise ValueError(f"图像 {im.key} 应为 uint8 {(im.height, im.width, 3)}")

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.args.port))
        srv.listen(2)
        srv.settimeout(0.2)
        log(f"监听 127.0.0.1:{self.args.port}")
        threading.Thread(target=self.load, daemon=True, name="loader").start()
        while not self.stop.is_set():
            if self.parent_gone():
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                self.keep_warm()
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(5.0)
            try:
                self.handle(conn)
            except (ConnectionError, OSError, ValueError, KeyError) as exc:
                log(f"客户端断开: {exc}")
            finally:
                conn.close()
        srv.close()
        log("已退出")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8890)
    p.add_argument('--tokenizer', default='pretrained/paligemma_tokenizer')
    p.add_argument('--gripper-floor', choices=('checkpoint', 'on', 'off'), default='checkpoint')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model-overrides', default='{}')
    p.add_argument('--instance-id', default='')
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("lerobot", "mock"), default="lerobot")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--keep-warm-s", type=float, default=1.0)
    p.add_argument("--exit-with-parent", action="store_true")
    args = p.parse_args()
    try:
        Server(args).serve()
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
