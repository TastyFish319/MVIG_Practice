"""Linux GPU FIFO 队列：原子入队/分配，显存与利用率同时达标才启动原流水线。

只使用标准库。全局 flock 保护排队顺序；每张 GPU 的 flock 持有至流水线退出，
并传给流水线进程，避免监控进程意外退出后同队列重复分配该卡。
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from . import console as ui
from .pipeline import ROOT, runtime_env


@dataclass(frozen=True)
class QueueOptions:
    memory_mb: int = 32768
    utilization: int = 10
    poll_seconds: float = 10
    min_free_memory_mb: int = 61440

    def __post_init__(self) -> None:
        if self.memory_mb < 0 or self.min_free_memory_mb < 0 or not 0 <= self.utilization <= 100:
            raise ValueError("显存阈值必须 >= 0，利用率阈值必须在 0~100 之间")
        if not math.isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError("检测间隔必须是大于 0 的有限数值")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _queue_dir() -> Path:
    return ROOT / "runs/.train_wait_queue" / socket.gethostname()


@contextmanager
def _locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _identity(pid: int) -> str | None:
    """启动时间 + boot ID 防止 PID 被复用；僵尸进程也视为已退出。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{stat[19]}"
    except (OSError, IndexError):
        return None


def _snapshot(job: dict, **changes) -> None:
    job.update(changes, updated_at=_now())
    _write_json(Path(job["run_dir"]) / "queue_status.json", job)


def _jobs(directory: Path) -> list[dict]:
    """必须持有全局锁；清掉被 kill -9 / 重启遗留的记录，避免堵塞队首。"""
    jobs = []
    for path in sorted(directory.glob("job-*.json")):
        job = _read_json(path)
        if job.get("identity") and _identity(job["pid"]) == job["identity"]:
            jobs.append(job)
            continue
        _snapshot(job, state="monitor_lost")
        status = Path(job["run_dir"]) / "status.json"
        if status.is_file() and _read_json(status).get("pipeline") == "queued":
            _write_json(status, {"pipeline": "cancelled", "stage": "cancelled", "error": "排队监控进程已退出"})
        path.unlink()
    return jobs


def queue_status() -> list[dict]:
    directory = _queue_dir()
    with _locked(directory):
        return _jobs(directory)


@contextmanager
def _update_signals():
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    original = {sig: signal.getsignal(sig) for sig in signals}

    def interrupt(signum, frame):
        raise InterruptedError(f"阈值更新被信号 {signum} 中断，正在恢复原等待监控")

    for sig in signals:
        signal.signal(sig, interrupt)
    try:
        yield signals
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def update_waiting_options(options: QueueOptions) -> list[dict]:
    """只更新等待任务的调度阈值；旧版监控按原条目交接，不取消/重新入队。

    正在运行的任务完全不动。新版监控每轮读阈值，无需再次交接。
    交接失败时恢复全部队列/状态/PID 文件并恢复旧监控。
    """
    directory = _queue_dir()
    backups, stopped, replacements, changes = {}, [], [], []
    with _update_signals() as signals, _locked(directory):
        jobs = [job for job in _jobs(directory) if job["state"] == "queued"]
        try:
            # 先建同目录硬链接备份，再暂停任何监控。回滚可直接 rename，
            # 不必在共享盘写满时重新分配文件数据；所有被更新文件均原子替换。
            for job in jobs:
                run = Path(job["run_dir"])
                paths = [directory / f"{job['id']}.json", run / "queue_status.json",
                         run / "status.json", run / "logs/pipeline.pid"]
                for path in paths:
                    backup = path.with_name(f".{path.name}.{uuid4().hex}.queue-backup") if path.exists() else None
                    if backup is not None:
                        os.link(path, backup)
                    backups[path] = backup
            for job in jobs:
                run = Path(job["run_dir"])
                job_path = directory / f"{job['id']}.json"
                old_pid = job["pid"]
                if job.get("monitor_version", 1) < 2:
                    proc = Path(f"/proc/{old_pid}")
                    command = [value.decode() for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
                    env = dict(value.decode().split("=", 1) for value in (proc / "environ").read_bytes().split(b"\0") if value)
                    cwd = (proc / "cwd").resolve()
                    # 前台等待的父进程仍在等旧 PID，不可在此悄悄替换。
                    if (proc / "fd/1").resolve() != (run / "logs/pipeline.log").resolve():
                        raise RuntimeError(f"{job['name']} 正以前台模式等待，不能自动交接旧监控")
                    if _identity(old_pid) != job["identity"]:
                        raise RuntimeError(f"{job['name']} 的监控进程已变化，请重试")
                    os.kill(old_pid, signal.SIGSTOP)
                    stopped.append(old_pid)
                    deadline = time.monotonic() + 2
                    while (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] not in ("T", "t"):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"无法暂停 {job['name']} 的等待监控")
                        time.sleep(0.01)
                    with (run / "logs/pipeline.log").open("ab") as log:
                        process = subprocess.Popen(
                            command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    replacements.append(process)
                    identity = _identity(process.pid)
                    if identity is None:
                        raise RuntimeError(f"{job['name']} 的新等待监控启动失败")
                    job.update(pid=process.pid, identity=identity, monitor_version=2)
                    status_path = run / "status.json"
                    status = _read_json(status_path)
                    status["pid"] = process.pid
                    _write_json(status_path, status)
                    pid_path = run / "logs/pipeline.pid"
                    partial = pid_path.with_name("pipeline.pid.update.partial")
                    partial.write_text(str(process.pid))
                    partial.replace(pid_path)
                job.update(memory_mb=options.memory_mb, utilization=options.utilization,
                           poll_seconds=options.poll_seconds, min_free_memory_mb=options.min_free_memory_mb)
                _write_json(job_path, job)
                _snapshot(job)
                changes.append({"name": job["name"], "old_pid": old_pid, "pid": job["pid"]})
        except BaseException as exc:
            for sig in signals:
                signal.signal(sig, signal.SIG_IGN)
            restore_errors = []
            # 新监控还阻塞在全局锁，尚无机会分配 GPU；强制退出避免 finally 等锁。
            for process in replacements:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            try:
                for path, backup in backups.items():
                    try:
                        if backup is None:
                            path.unlink(missing_ok=True)
                        else:
                            backup.replace(path)
                    except OSError as error:
                        restore_errors.append(f"{path}: {error}")
            finally:
                for pid in stopped:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        restore_errors.append(f"恢复监控 {pid}: {error}")
            if restore_errors:
                raise RuntimeError("等待监控交接失败，部分恢复需检查：" + "; ".join(restore_errors)) from exc
            raise
        # 同一把锁下旧监控已暂停、新记录已完整写好；旧版 finally 不能删原队列条目。
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        for pid in stopped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
    return changes


def query_gpus() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            raise ValueError(f"无法解析 nvidia-smi 输出：{row!r}")
        index, uuid, memory, utilization, free_memory = (value.strip() for value in row)

        def metric(value: str) -> float | None:
            try:
                number = float(value)
                return number if math.isfinite(number) and number >= 0 else None
            except ValueError:
                return None  # N/A / 不支持的指标不可作为空闲卡

        if not uuid.startswith("GPU-"):
            raise ValueError(f"无效 GPU UUID：{uuid!r}")
        gpus.append({"index": int(index), "uuid": uuid, "memory_mb": metric(memory),
                     "utilization": metric(utilization), "free_memory_mb": metric(free_memory)})
    if not gpus:
        raise ValueError("nvidia-smi 未返回任何 GPU")
    return gpus


def _allowed_gpus(gpus: list[dict]) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [gpu["uuid"] for gpu in gpus]
    tokens = [token.strip() for token in visible.split(",")]
    # 数字是 CUDA 枚举编号，未必等于 nvidia-smi 编号；从 CUDA 解析实际 UUID。
    if any(token.isdecimal() for token in tokens):
        import torch

        if torch.cuda.device_count() != len(tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES 包含不可见/无效 GPU")
        tokens = [str(torch.cuda.get_device_properties(i).uuid) for i in range(len(tokens))]
        tokens = [token if token.startswith("GPU-") else f"GPU-{token}" for token in tokens]
    allowed = []
    for token in tokens:
        matches = [
            gpu["uuid"] for gpu in gpus
            if token.startswith("GPU-") and gpu["uuid"].startswith(token)
        ]
        if len(matches) != 1 or matches[0] in allowed:
            raise ValueError(f"CUDA_VISIBLE_DEVICES 中的 GPU 不可用/重复：{token!r}")
        allowed.append(matches[0])
    return allowed


def submit(run_dir: Path, gpus: int, foreground: bool, *, options: QueueOptions) -> int:
    """由原 train CLI 在所有选择完成并保存配置后调用；默认脱离 SSH 后台监控。"""
    run_dir = run_dir.resolve()
    try:
        allowed = _allowed_gpus(query_gpus())
        if gpus > len(allowed):
            raise ValueError(f"需要 {gpus} 张 GPU，但当前可见 GPU 只有 {len(allowed)} 张")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        ui.error(f"无法提交 GPU 队列：{exc}")
        return 1
    directory = _queue_dir()
    log = run_dir / "logs/pipeline.log"
    with _locked(directory):
        _jobs(directory)
        sequence_file = directory / "sequence.json"
        sequence = json.loads(sequence_file.read_text()) + 1 if sequence_file.exists() else 1
        _write_json(sequence_file, sequence)
        job_path = directory / f"job-{sequence:020d}-{uuid4().hex}.json"
        job = {
            "id": job_path.stem, "name": run_dir.name, "run_dir": str(run_dir), "gpus": gpus,
            "state": "queued", "submitted_at": _now(), "allowed_uuids": allowed,
            "memory_mb": options.memory_mb, "utilization": options.utilization, "poll_seconds": options.poll_seconds,
            "min_free_memory_mb": options.min_free_memory_mb, "monitor_version": 2,
        }
        cmd = [sys.executable, "-m", "pitrain.gpu_queue", "--job", str(job_path)]
        with log.open("ab") as output:
            process = subprocess.Popen(
                cmd, cwd=str(ROOT), env=runtime_env(), stdin=subprocess.DEVNULL,
                stdout=None if foreground else output, stderr=None if foreground else subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            job.update(pid=process.pid, identity=_identity(process.pid))
            if job["identity"] is None:
                raise RuntimeError("GPU 监控进程未能启动")
            _write_json(job_path, job)
            _snapshot(job)
            _write_json(run_dir / "status.json", {"pipeline": "queued", "stage": "waiting_gpu", "pid": process.pid})
            (run_dir / "logs/pipeline.pid").write_text(str(process.pid))
        except BaseException:
            # 此时仍持有全局锁，worker 尚不能开始。SIGKILL 不执行需要同一锁的 finally。
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            job_path.unlink(missing_ok=True)
            raise
    ui.ok(f"已加入 GPU 队列（监控 PID {process.pid}；严格按提交顺序分配）")
    ui.kv_table("如何查看", {
        "空闲标准": f"已用 ≤ {options.memory_mb} MiB、剩余 ≥ {options.min_free_memory_mb} MiB、利用率 ≤ {options.utilization}%；每 {options.poll_seconds:g} 秒检查",
        "实时日志": f"tail -f {log}",
        "进度状态": f"cat {run_dir / 'status.json'}",
        "排队详情": f"cat {run_dir / 'queue_status.json'}",
        "全部队列": "bash scripts/train_wait.sh --queue-status",
        "取消/中止": f"kill -- -{process.pid}",
        "最终产物": run_dir / f"{run_dir.name}_statistics.zip",
    })
    if foreground:
        try:
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            return 130
    return 0


class _Cancelled(BaseException):
    pass


def _cancel(signum, frame) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise _Cancelled()


def _group_alive(pgid: int) -> bool:
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] not in ("Z", "X") and int(stat[2]) == pgid:
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def _stop_pipeline(process: subprocess.Popen) -> None:
    """pipeline 异常退出时也清理 DDP/评测后代，确认进程组退出才释放 GPU 锁。"""
    deadline = time.monotonic() + 5
    while _group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    process.wait()


def _reserve(directory: Path, job: dict, gpus: list[dict]) -> tuple[list[dict], list]:
    selected, locks = [], []
    for gpu in gpus:
        if gpu["uuid"] not in job["allowed_uuids"]:
            continue
        if gpu["memory_mb"] is None or gpu["utilization"] is None:
            continue
        if gpu["memory_mb"] > job["memory_mb"] or gpu["utilization"] > job["utilization"]:
            continue
        minimum_free = job.get("min_free_memory_mb", 0)
        if minimum_free and (gpu["free_memory_mb"] is None or gpu["free_memory_mb"] < minimum_free):
            continue
        lock = (directory / f"{gpu['uuid']}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        selected.append(gpu)
        locks.append(lock)
        if len(selected) == job["gpus"]:
            return selected, locks
    for lock in locks:
        lock.close()
    return [], []


def worker(job_path: Path) -> int:
    directory = job_path.parent
    locks = []
    job = None
    pipeline = None
    started = False
    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    try:
        # 提交者持有锁直至写好记录，防止后台进程提前读取半成品。
        with _locked(directory):
            if not job_path.exists():
                return 1
            job = _read_json(job_path)
        print(f"[{_now()}] 已入队：{job['name']}，需要 {job['gpus']} 张 GPU", flush=True)
        previous = None
        while True:
            with _locked(directory):
                jobs = _jobs(directory)
                waiting = [entry for entry in jobs if entry["state"] == "queued"]
                position = next(i for i, entry in enumerate(waiting, 1) if entry["id"] == job["id"])
                current = waiting[position - 1]
                for key in ("memory_mb", "utilization", "poll_seconds", "min_free_memory_mb"):
                    if key in current:
                        job[key] = current[key]  # 调度阈值热更新，不改变训练配置与队列位置
                error = None
                selected = []
                if position == 1:
                    try:
                        selected, locks = _reserve(directory, job, query_gpus())
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        error = str(exc)  # 采集失败时继续等，绝不误启动
                if selected:
                    job.update(state="running", gpu_indices=[gpu["index"] for gpu in selected],
                               gpu_uuids=[gpu["uuid"] for gpu in selected], started_at=_now())
                    _write_json(job_path, job)
                    _snapshot(job, queue_position=0, monitor_error=None)
                    break
                _snapshot(job, queue_position=position, monitor_error=error)
                message = (position, error)
                if message != previous:
                    print(f"[{_now()}] 等待 GPU：队列第 {position} 位；" + (f"采集失败：{error}" if error else "空闲卡不足或前方有任务"), flush=True)
                    previous = message
            time.sleep(job["poll_seconds"])

        env = runtime_env()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(job["gpu_uuids"])
        print(f"[{_now()}] 分配 GPU {job['gpu_indices']}：{env['CUDA_VISIBLE_DEVICES']}；启动原训练/评测/打包流水线", flush=True)
        started = True
        # 单独进程组便于清理 pipeline 退出后仍活着的训练子孙。创建期间延迟处理
        # 取消，确保先拿到待清理句柄；不把被屏蔽的信号掩码传给训练进程。
        cancel_requested = False

        def defer_cancel(signum, frame):
            nonlocal cancel_requested
            cancel_requested = True

        signal.signal(signal.SIGTERM, defer_cancel)
        signal.signal(signal.SIGINT, defer_cancel)
        try:
            pipeline = subprocess.Popen(
                [sys.executable, "-m", "pitrain.pipeline", "--run", job["run_dir"]],
                cwd=str(ROOT), env=env, pass_fds=tuple(lock.fileno() for lock in locks), process_group=0,
            )
        finally:
            signal.signal(signal.SIGTERM, _cancel)
            signal.signal(signal.SIGINT, _cancel)
        if cancel_requested:
            raise _Cancelled()
        code = pipeline.wait()
        _stop_pipeline(pipeline)
        pipeline = None
        _snapshot(job, state="done" if code == 0 else "failed", returncode=code, finished_at=_now())
        if code:
            status_path = Path(job["run_dir"]) / "status.json"
            status = _read_json(status_path)
            status.update(pipeline="failed", returncode=code)
            _write_json(status_path, status)
        return code
    except _Cancelled:
        if job is not None:
            _snapshot(job, state="cancelled", finished_at=_now())
            status_path = Path(job["run_dir"]) / "status.json"
            status = _read_json(status_path)
            status.update(pipeline="cancelled", stage="cancelled")
            _write_json(status_path, status)
        return 130
    except BaseException as exc:
        if job is not None:
            _snapshot(job, state="failed", error=repr(exc), finished_at=_now())
            if not started:
                _write_json(Path(job["run_dir"]) / "status.json", {"pipeline": "failed", "error": repr(exc)})
        raise
    finally:
        # 清理期间忽略后续取消信号，不能在子孙仍运行时提前释放锁。
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if pipeline is not None:
            _stop_pipeline(pipeline)
        with _locked(directory):
            job_path.unlink(missing_ok=True)
        for lock in locks:
            lock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", type=Path, required=True)
    return worker(ap.parse_args(argv).job)


if __name__ == "__main__":
    raise SystemExit(main())
