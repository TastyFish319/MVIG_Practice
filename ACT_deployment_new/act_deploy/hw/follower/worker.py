"""One process = one follower arm (CAN bus + motors) with its own watchdog, served over portal RPC.

    <device-python> -m act_deploy.hw.follower.worker --config configs/hardware.yaml --side right [--fake]

Started by act_deploy.hw.arms.FollowerArms; not normally run by hand.

Safety properties (see runtime.py, vendored unchanged from SuperInference):
  * stale command frames (> command_frame_max_age_s) are rejected
  * no new frame for command_lease_timeout_s  -> FAULT -> hold the current position (powered)
  * following error > threshold                -> FAULT -> hold
Differences to the SuperInference worker (docs/vendored.md):
  * portal.Server(errors=False) + wrapped methods: a rejected request returns an error to the
    caller instead of killing the worker (with errors=True a single stale frame would end the
    process -> zero torque -> the arm drops)
  * the worker does NOT die with the console: if the console crashes, the lease expires and the
    arm holds its pose; start.sh then runs the rescue tool (return to zero, then release)
  * writes logs/.workers/<side>.json (pid + port) so the rescue tool can find it
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from .config import load_teleoperation_config
from .endpoints import FakeFollowerEndpoint, create_follower_endpoint
from .runtime import WorkerControlRuntime

LOGGER = logging.getLogger("act_deploy.worker")


def _safe(fn):
    def call(*args):
        try:
            return {"ok": True, "value": fn(*args)}
        except Exception as exc:  # noqa: BLE001 - reported to the caller, worker keeps running
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return call


def serve_worker(config_path: Path, side: str, fake: bool, pid_dir: Path | None) -> None:
    import portal

    config = load_teleoperation_config(config_path)
    arm_config = config.arms[side]
    port = arm_config.follower.rpc_port
    endpoint = server = runtime = None
    stop_requested = threading.Event()
    stage = "creating the hardware endpoint"
    pid_file = pid_dir / f"{side}.json" if pid_dir else None

    def request_stop(_signum: int, _frame: Any) -> None:
        LOGGER.info("stop requested for %s worker", side)
        stop_requested.set()
        if runtime is not None and (server is None or not server.running):
            runtime.request_stop()

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C in the console must not drop the arm
    signal.signal(signal.SIGTERM, request_stop)
    try:
        LOGGER.info("worker %s%s: %s (CAN %s, port %s)", side, " FAKE" if fake else "", stage,
                    arm_config.follower.can_channel, port)
        endpoint = FakeFollowerEndpoint() if fake else create_follower_endpoint(arm_config.follower)
        runtime = WorkerControlRuntime(endpoint, role="follower", side=side, control=config.control)
        stage = "initialising the safe-hold loop"
        if not runtime.start(config.control.startup_timeout_s, cancel_event=stop_requested):
            return
        stage = f"starting RPC server on port {port}"
        server = portal.Server(port, name=f"follower_{side}", errors=False)
        for name in ("ping", "get_sample", "get_status", "begin_control_session", "set_phase",
                     "submit_control_frame", "end_control_session", "reset_fault"):
            server.bind(name, _safe(getattr(runtime, name)))
        server.start(block=False)
        if pid_file is not None:
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(json.dumps({"pid": os.getpid(), "port": port, "side": side,
                                            "fake": fake, "started": time.time()}))
        LOGGER.info("worker %s: ready on 127.0.0.1:%s", side, port)
        stage = "serving"
        while server.running and not stop_requested.wait(0.1):
            pass
    except Exception as exc:
        LOGGER.exception("worker %s failed while %s: %s", side, stage, exc)
        raise
    finally:
        if server is not None and server.running:
            server.close()
        if runtime is not None:
            # runtime.close() commands a hold first; endpoint.close() then releases the motors
            if runtime.close() and endpoint is not None:
                endpoint.close()
            else:
                LOGGER.error("skipping endpoint close for blocked %s worker", side)
        elif endpoint is not None:
            endpoint.close()
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)
        LOGGER.info("worker %s: exited (motors released)", side)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--side", required=True, choices=("left", "right"))
    p.add_argument("--fake", action="store_true")
    p.add_argument("--pid-dir", type=Path, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve_worker(args.config.resolve(), args.side, args.fake, args.pid_dir)


if __name__ == "__main__":
    main()
