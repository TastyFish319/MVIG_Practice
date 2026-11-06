"""Independent checkpoint workers and a single failure-safe report coordinator."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluation_devices(device, requested, checkpoint_count, available):
    if not isinstance(checkpoint_count, int) or checkpoint_count < 1:
        raise ValueError("Evaluation needs at least one checkpoint")
    auto = requested == "auto"
    if not auto and (isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 4):
        raise ValueError("Evaluation GPU count must be auto or an integer from 1 to 4")
    if device == "cpu":
        if not auto and requested != 1:
            raise ValueError("CPU evaluation uses one worker")
        return ["cpu"]
    if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("Use cpu, cuda, or cuda:N")
    if available < 1:
        raise ValueError("CUDA requested but no GPUs are visible")
    if device != "cuda":
        if int(device[5:]) >= available or (not auto and requested != 1):
            raise ValueError("An explicit cuda:N device selects one visible GPU")
        return [device]
    count = min(4, available) if auto else requested
    if count > available:
        raise ValueError(f"Requested {count} evaluation GPUs but only {available} are visible")
    return [f"cuda:{i}" for i in range(min(count, checkpoint_count))]


def checkpoint_assignments(tasks, devices):
    if not tasks or not devices:
        raise ValueError("Checkpoint tasks and devices must be nonempty")
    return [tasks[i::len(devices)] for i in range(min(len(devices), len(tasks)))]


def worker_command(plan):
    return [sys.executable, "-u", "-m", "acttrain.evaluation.runner", "--worker-plan", str(plan)]


def run_parallel_evaluation(config, tasks, devices):
    """Evaluate disjoint checkpoint shards; never publish a final ranking here."""
    output = Path(config["output"])
    if [task["index"] for task in tasks] != list(range(len(tasks))):
        raise ValueError("Checkpoint tasks require contiguous stable indices")
    shards = checkpoint_assignments(tasks, devices)
    work = output / ".workers"
    work.mkdir()
    processes, logs, plans = [], [], []
    completed, previous = 0, -1
    begin = time.monotonic()
    manages_signals = threading.current_thread() is threading.main_thread()
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)} if manages_signals else {}
    spawning, pending_signal = False, None

    def terminate(signum, frame):
        nonlocal pending_signal
        # A handler must not unwind Popen after the child exists but before registration.
        if spawning:
            pending_signal = signum
            return
        raise InterruptedError(f"Evaluation coordinator received {signal.Signals(signum).name}")

    for signum in previous_signals:
        signal.signal(signum, terminate)

    def progress(status, **extra):
        atomic_json(output / "progress.json", dict(status=status, expected_checkpoints=len(tasks),
                    completed=completed, workers=len(shards), elapsed_seconds=time.monotonic() - begin, **extra))

    try:
        progress("running")
        for number, shard in enumerate(shards):
            plan = work / f"worker_{number}.json"
            atomic_json(plan, {**config, "device": devices[number], "tasks": shard})
            log = (work / f"worker_{number}.log").open("xb")
            logs.append(log)
            spawning = True
            try:
                process = subprocess.Popen(worker_command(plan), stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                processes.append(process)
                plans.append(plan)
            finally:
                spawning = False
            if pending_signal is not None:
                terminate(pending_signal, None)
            print(f"Evaluation worker {number}: {devices[number]}, {len(shard)} checkpoints, PID {process.pid}", flush=True)
        while True:
            records = []
            for number, (process, plan) in enumerate(zip(processes, plans)):
                result_file = plan.with_suffix(".result.json")
                record = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
                records.append(record)
                if record.get("status") == "failed" or process.poll() not in (None, 0):
                    raise RuntimeError(f"Evaluation worker {number} failed: {record.get('error', 'nonzero exit')}; "
                                       f"see {work / f'worker_{number}.log'}")
            completed = sum(len(record.get("results", [])) for record in records)
            if completed != previous:
                progress("running")
                print(f"Evaluated {completed}/{len(tasks)} checkpoints", flush=True)
                previous = completed
            exit_codes = [process.poll() for process in processes]
            if all(code is not None for code in exit_codes):
                if any(code != 0 for code in exit_codes):
                    raise RuntimeError(f"Evaluation worker exited with nonzero status: {exit_codes}")
                # A process may finish after the status read above; re-read its final receipt.
                records = [json.loads(plan.with_suffix(".result.json").read_text(encoding="utf-8")) for plan in plans]
                results = []
                for shard, record in zip(shards, records):
                    rows = record.get("results", [])
                    if (record.get("status") != "passed" or
                            sorted(row["index"] for row in rows) != sorted(task["index"] for task in shard)):
                        raise RuntimeError("Evaluation worker exited without a complete checkpoint result")
                    results.extend(rows)
                return [row["result"] for row in sorted(results, key=lambda row: row["index"])]
            time.sleep(.2)
    except BaseException as error:
        progress("failed", error=str(error))
        raise
    finally:
        # A failed or interrupted coordinator must not leave GPU workers running.
        for signum in previous_signals:
            signal.signal(signum, signal.SIG_IGN)
        try:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
        finally:
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
