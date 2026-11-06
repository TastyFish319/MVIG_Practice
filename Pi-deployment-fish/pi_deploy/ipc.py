"""Tiny loopback protocol between the console (Py3.10) and the model server (Py3.12).

One message = b"PI05" + uint32 header length + JSON header + raw array bytes.
The header lists the arrays ({name, dtype, shape}); no pickle, so the two Python / numpy
versions never have to agree on anything but this file.  A persistent TCP connection is kept
(no HTTP, no reconnect per request) to keep the round trip around 1 ms.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import numpy as np

MAGIC = b"PI05"
MAX_BYTES = 64 * 1024 * 1024


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("连接被对方关闭")
        buf += chunk
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    arrays = arrays or {}
    meta, blobs = [], []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        meta.append({"name": name, "dtype": arr.dtype.str, "shape": list(arr.shape)})
        blobs.append(arr.tobytes())
    head = json.dumps({**header, "arrays": meta}, allow_nan=False).encode()
    sock.sendall(MAGIC + struct.pack("!I", len(head)) + head + b"".join(blobs))


def recv_msg(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    pre = _recv_exact(sock, 8)
    if pre[:4] != MAGIC:
        raise ConnectionError("协议错误（magic 不匹配）")
    (n,) = struct.unpack("!I", pre[4:])
    if n > 65536:
        raise ConnectionError("消息头过大")
    header = json.loads(_recv_exact(sock, n))
    arrays = {}
    total = 0
    for m in header.pop("arrays", []):
        dt = np.dtype(m["dtype"])
        shape = tuple(int(v) for v in m["shape"])
        import math
        if dt.kind not in 'fiu' or dt.itemsize not in (1, 2, 4, 8) or len(shape) > 5 or any(v < 0 for v in shape):
            raise ConnectionError('非法数组类型或维度')
        size = math.prod(shape) * dt.itemsize
        total += size
        if total > MAX_BYTES or m['name'] in arrays:
            raise ConnectionError("数组过大")
        arrays[m["name"]] = np.frombuffer(_recv_exact(sock, size), dtype=dt).reshape(shape)
    return header, arrays


class PolicyClient:
    """Robot-side client.  Thread-safe (one request at a time)."""

    def __init__(self, host: str, port: int, timeout_s: float = 1.0):
        self.addr = (host, int(port))
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._request_id = 0

    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection(self.addr, timeout=self.timeout_s)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self.timeout_s)
            self._sock = s
        return self._sock

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def request(self, header: dict, arrays=None, timeout_s: float | None = None):
        with self._lock:
            try:
                s = self._connect()
                s.settimeout(timeout_s or self.timeout_s)
                send_msg(s, header, arrays)
                reply, out = recv_msg(s)
            except (OSError, ConnectionError, ValueError):
                self._drop()  # a timed-out reply would desynchronise the stream
                raise
        if not reply.get("ok", False):
            raise RuntimeError(f"模型服务报错: {reply.get('error')}")
        return reply, out

    def health(self, timeout_s: float = 1.0) -> dict:
        return self.request({"op": "health"}, timeout_s=timeout_s)[0]

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
        arrays = {"state": np.asarray(state, np.float32)}
        arrays.update({f"img:{k}": v for k, v in images.items()})
        t0 = time.monotonic()
        self._request_id += 1
        request_id = self._request_id
        reply, out = self.request({"op": "predict", "request_id": request_id}, arrays)
        if reply.get('request_id') != request_id:
            self.close()
            raise ConnectionError('推理请求编号不匹配')
        reply["roundtrip_ms"] = (time.monotonic() - t0) * 1000
        return out["actions"], reply

    def shutdown(self) -> None:
        try:
            self.request({"op": "shutdown"}, timeout_s=2.0)
        except Exception:  # noqa: BLE001 - best effort
            pass
        self.close()

    def close(self) -> None:
        with self._lock:
            self._drop()
