"""集中验收：真实后台进程/文件锁，临时目录内模拟 nvidia-smi 和流水线，不占 GPU。

在 Linux 上运行：python -m unittest tests.test_gpu_queue_acceptance -v
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAKE_SMI = '''import json, os, sys
from pathlib import Path
data = json.loads(Path(os.environ["GPU_FIXTURE"]).read_text())
if isinstance(data, dict):
    print("temporary NVML error", file=sys.stderr)
    sys.exit(1)
for row in data:
    index, memory, utilization = row[:3]
    free = row[3] if len(row) > 3 else 97871 - float(memory) if isinstance(memory, (float, int)) else "N/A"
    print(f"{index}, GPU-acceptance-{index}, {memory}, {utilization}, {free}")
'''

FAKE_PIPELINE = '''import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def runtime_env():
    return dict(os.environ)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    run = ap.parse_args().run
    def write(name, data):
        tmp = run / (name + ".partial")
        tmp.write_text(json.dumps(data))
        tmp.replace(run / name)
    write("status.json", {"pipeline": "running", "stage": "training"})
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) if (run / "spawn_child").exists() else None
    write("pipeline_started.json", {"pid": os.getpid(), "child_pid": child.pid if child else None,
                                   "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "time": time.monotonic()})
    while not (run / "release").exists():
        time.sleep(0.02)
    code = int((run / "release").read_text())
    write("stages.json", ["training", "evaluating", "packaging"])
    write("status.json", {"pipeline": "done" if code == 0 else "failed", "stage": "done"})
    raise SystemExit(code)
'''


@unittest.skipUnless(sys.platform == "linux", "GPU 队列使用 Linux flock 和 /proc")
class GPUQueueAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pitrain-queue-acceptance-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.stop_workers)
        package = self.root / "pitrain"
        (package / "cli").mkdir(parents=True)
        for relative in ("gpu_queue.py", "console.py", "hparams.py", "cli/train.py", "cli/train_wait.py"):
            shutil.copy2(ROOT / "pitrain" / relative, package / relative)
        (package / "__init__.py").touch()
        (package / "cli/__init__.py").touch()
        (package / "pipeline.py").write_text(FAKE_PIPELINE)
        # CLI 的数据读取边界用微型元信息替代，训练交互与配置生成代码使用原文件。
        (package / "nedf3.py").write_text("import json\ndef read_json(path):\n    return json.loads(path.read_text())\n")
        (package / "preprocess.py").write_text('META_FILE = "pitrain_meta.json"\n')
        (self.root / "torch.py").write_text(
            'import os\nfrom types import SimpleNamespace\n'
            'class cuda:\n'
            '    @staticmethod\n'
            '    def device_count(): return len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(","))\n'
            '    @staticmethod\n'
            '    def get_device_properties(i):\n'
            '        token = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[i]\n'
            '        return SimpleNamespace(uuid=f"GPU-acceptance-{[3,2,1,0][int(token)]}")\n'
        )
        dataset = self.root / "dataset/acceptance"
        dataset.mkdir(parents=True)
        self.write(dataset / "pitrain_meta.json", {
            "counts": {"train": {"episodes": 10, "frames": 10000}, "val": {"episodes": 2}},
            "prompt": "queue acceptance", "action_arms": ["right"], "fps": 30,
            "created_at": "2026-11-02T00:00:00",
        })
        for relative in ("pretrained/pi05_base/model.safetensors", "pretrained/paligemma_tokenizer/tokenizer.model"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("_env.sh", "train.sh", "train_wait.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        executable_dir = self.root / "bin"
        executable_dir.mkdir()
        (executable_dir / "python").symlink_to(sys.executable)
        nvidia = executable_dir / "nvidia-smi"
        nvidia.write_text(f"#!{sys.executable}\n" + FAKE_SMI)
        nvidia.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update(
            PYTHONPATH=str(self.root), GPU_FIXTURE=str(self.root / "gpu.json"),
            PATH=str(executable_dir) + os.pathsep + self.env["PATH"],
            CONDA_DEFAULT_ENV="pi_fish", PYTHONUNBUFFERED="1",
        )
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        self.metrics = [[index, 8000, 80] for index in range(4)]
        self.set_metrics(self.metrics)

    @staticmethod
    def write(path, value):
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(json.dumps(value))
        partial.replace(path)

    def set_metrics(self, value):
        self.write(self.root / "gpu.json", value)

    def run_cli(self, *args, original=False, env=None, expected=0):
        result = subprocess.run(
            ["bash", "scripts/train.sh" if original else "scripts/train_wait.sh", *args],
            cwd=self.root, env=env or self.env, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def submit(self, name, gpus=1, *, memory_limit=1024, **kwargs):
        self.run_cli("--dataset", "acceptance", "--gpus", str(gpus), "--name", name,
                     "--yes", "--poll-seconds", "0.05", "--idle-memory-mb", str(memory_limit), **kwargs)
        return self.root / "runs" / name

    def read(self, run, filename="queue_status.json"):
        return json.loads((run / filename).read_text())

    def wait_for(self, condition, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = condition()
            if result:
                return result
            time.sleep(0.02)
        logs = {str(path.relative_to(self.root)): path.read_text()[-2500:] for path in self.root.glob("runs/*/logs/pipeline.log")}
        self.fail(f"等待超时；后台日志：{logs}")

    def started(self, run):
        return (run / "pipeline_started.json").is_file()

    def finish(self, run, code=0):
        (run / "release").write_text(str(code))
        self.wait_for(lambda: self.read(run)["state"] in ("done", "failed"))

    def assert_waiting(self, *runs):
        time.sleep(0.18)
        for run in runs:
            self.assertFalse(self.started(run), f"{run.name} 提前启动")

    def stop_workers(self):
        groups = [int(path.read_text()) for path in self.root.glob("runs/*/logs/pipeline.pid")]
        groups.extend(json.loads(path.read_text())["pid"] for path in self.root.glob("runs/*/pipeline_started.json"))
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_original_configuration_and_dry_run_are_preserved(self):
        flags = ("--dataset", "acceptance", "--gpus", "2", "--velocity", "yes", "--steps", "400", "--yes", "--dry-run")
        self.run_cli(*flags, "--name", "original", original=True)
        self.run_cli(*flags, "--name", "wait")
        configs = [self.read(self.root / "runs" / name, "train_config.json") for name in ("original", "wait")]
        for config in configs:
            config.pop("output")
            config.pop("run_name")
        self.assertEqual(*configs)
        self.assertFalse((self.root / "runs/.train_wait_queue").exists())
        self.assertFalse(self.started(self.root / "runs/wait"))

    def test_memory_and_compute_gates_assign_noncontiguous_gpu_uuids(self):
        self.metrics = [[0, 4000, 0], [1, 0, 90], [2, 2000, 30], [3, "N/A", 0]]
        self.set_metrics(self.metrics)
        run = self.submit("both_thresholds", 2)
        self.wait_for(lambda: self.read(run).get("queue_position") == 1)
        self.assertEqual(self.read(run, "status.json")["pipeline"], "queued")
        self.assert_waiting(run)
        self.metrics[1][2] = 0
        self.set_metrics(self.metrics)
        self.assert_waiting(run)  # 只有一张满足，不能少卡启动
        self.metrics[2] = [2, 1024, 10]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(run))
        self.assertEqual(self.read(run, "pipeline_started.json")["gpus"], "GPU-acceptance-1,GPU-acceptance-2")
        self.finish(run)
        self.assertEqual(self.read(run, "stages.json"), ["training", "evaluating", "packaging"])

    def test_fifo_head_blocks_overtaking_and_spare_gpus_run_in_parallel(self):
        head, second, third = self.submit("head", 2), self.submit("second"), self.submit("third")
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        self.assert_waiting(head, second, third)
        self.metrics[1] = [1, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(head))
        self.assert_waiting(second, third)  # 模拟采集值仍为空闲，GPU 锁也必须生效
        self.metrics[2] = [2, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(second))
        self.assert_waiting(third)
        self.metrics[3] = [3, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(third))
        self.assertEqual(self.read(head)["gpu_indices"], [0, 1])
        self.assertEqual(self.read(second)["gpu_indices"], [2])
        self.assertEqual(self.read(third)["gpu_indices"], [3])
        self.run_cli("--queue-status")
        for run in (head, second, third):
            self.finish(run)

    def test_dead_queue_head_is_removed_and_pipeline_retains_gpu_lock(self):
        dead = self.submit("dead_head")
        os.killpg(self.read(dead)["pid"], signal.SIGKILL)
        running = self.submit("orphan_monitor")
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(running))
        self.assertEqual(self.read(dead)["state"], "monitor_lost")
        waiting = self.submit("after_orphan")
        os.kill(self.read(running)["pid"], signal.SIGKILL)  # 只杀监控，不杀持锁的流水线
        self.run_cli("--queue-status")
        self.assert_waiting(waiting)
        (running / "release").write_text("0")
        self.wait_for(lambda: self.started(waiting))
        self.finish(waiting)

    def test_nvml_error_waits_and_pipeline_failure_releases_gpu(self):
        first = self.submit("nvml_failure")
        self.set_metrics({"error": True})
        self.wait_for(lambda: self.read(first).get("monitor_error"))
        self.assert_waiting(first)
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(first))
        second = self.submit("after_failed_pipeline")
        self.assert_waiting(second)
        self.finish(first, 7)
        self.wait_for(lambda: self.started(second))
        self.assertEqual(self.read(first, "status.json")["pipeline"], "failed")
        self.assertEqual(self.read(first)["returncode"], 7)
        self.finish(second)

    def test_cancellation_removes_waiter_and_stops_running_group(self):
        waiting = self.submit("cancel_waiter")
        self.wait_for(lambda: self.read(waiting).get("queue_position") == 1)
        os.killpg(self.read(waiting)["pid"], signal.SIGTERM)
        self.wait_for(lambda: self.read(waiting)["state"] == "cancelled")
        self.assertFalse(self.started(waiting))
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        running = self.submit("cancel_running")
        self.wait_for(lambda: self.started(running))
        os.killpg(self.read(running)["pid"], signal.SIGTERM)
        self.wait_for(lambda: self.read(running, "status.json")["pipeline"] == "cancelled")
        self.assertEqual(self.read(running, "status.json")["pipeline"], "cancelled")
        next_run = self.submit("after_cancel")
        self.wait_for(lambda: self.started(next_run))
        self.finish(next_run)

    def test_visible_gpu_limit_invalid_options_and_help(self):
        env = dict(self.env, CUDA_VISIBLE_DEVICES="2,0")
        self.submit("impossible", 4, env=env, expected=1)
        self.assertFalse((self.root / "runs/.train_wait_queue").exists())
        self.metrics = [[index, 0, 0] for index in range(4)]
        self.set_metrics(self.metrics)
        run = self.submit("restricted", 2, env=env)
        self.wait_for(lambda: self.started(run))
        # 模拟 CUDA FASTEST_FIRST 顺序与 nvidia-smi 完全相反：CUDA 2/0 对应物理 1/3。
        self.assertEqual(set(self.read(run)["gpu_indices"]), {1, 3})
        self.finish(run)
        self.run_cli("--poll-seconds", "nan", expected=2)
        help_output = self.run_cli("--help").stdout
        self.assertIn("--idle-memory-mb", help_output)
        self.assertIn("--foreground", help_output)

    def test_submission_write_failure_does_not_deadlock_queue(self):
        failed = self.root / "runs/write_failure"
        (failed / "logs").mkdir(parents=True)
        script = '''import time
from pathlib import Path
from pitrain import gpu_queue as queue
original = queue._write_json
def disk_failure(path, value):
    if path.name == "status.json":
        time.sleep(0.2)  # 让 worker 安装信号处理器并等待全局锁
        raise OSError("simulated disk error")
    original(path, value)
queue._write_json = disk_failure
try:
    queue.submit(Path("runs/write_failure"), 1, False, options=queue.QueueOptions(poll_seconds=0.05))
except OSError:
    pass
else:
    raise AssertionError("missing simulated failure")
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=self.root, env=self.env,
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        run = self.submit("after_write_failure")
        self.wait_for(lambda: self.started(run))
        self.finish(run)

    def test_pipeline_crash_cleans_up_training_descendants_before_reallocation(self):
        first = self.submit("crash_with_child")
        (first / "spawn_child").touch()
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(first))
        descendant = self.read(first, "pipeline_started.json")["child_pid"]
        second = self.submit("after_pipeline_crash")
        os.kill(self.read(first, "pipeline_started.json")["pid"], signal.SIGKILL)
        self.wait_for(lambda: self.started(second))
        self.assertEqual(self.read(first)["state"], "failed")
        proc = Path(f"/proc/{descendant}/stat")
        if proc.exists():
            self.assertEqual(proc.read_text().rsplit(")", 1)[1].split()[0], "Z")
        self.finish(second)

    def test_foreground_ctrl_c_cancels_background_worker_and_pipeline(self):
        self.metrics[0] = [0, 0, 0]
        self.set_metrics(self.metrics)
        run = self.root / "runs/foreground"
        with (self.root / "foreground.log").open("w") as output:
            process = subprocess.Popen(
                ["bash", "scripts/train_wait.sh", "--dataset", "acceptance", "--gpus", "1", "--name", "foreground",
                 "--yes", "--foreground", "--poll-seconds", "0.05"],
                cwd=self.root, env=self.env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                self.wait_for(lambda: self.started(run))
                os.killpg(process.pid, signal.SIGINT)
                self.assertEqual(process.wait(timeout=5), 130)
                self.assertEqual(self.read(run)["state"], "cancelled")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def test_gpu_zero_resident_memory_is_allowed_but_free_memory_is_required(self):
        self.metrics[0] = [0, 29617, 0, 67668]
        self.metrics[1] = [1, 2000, 0, "N/A"]
        self.set_metrics(self.metrics)
        run = self.submit("resident_two_gpu", 2, memory_limit=32768)
        self.assert_waiting(run)
        self.metrics[1] = [1, 2000, 0, 60000]  # 已用显存很少，但剩余仍不足 60 GiB
        self.set_metrics(self.metrics)
        self.assert_waiting(run)
        self.metrics[1][3] = 70000
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(run))
        self.assertEqual(self.read(run)["gpu_indices"], [0, 1])
        self.finish(run)

    def test_thresholds_hot_reload_without_restarting_new_monitor(self):
        self.metrics[0] = [0, 29617, 0, 67668]
        self.set_metrics(self.metrics)
        run = self.submit("hot_reload")
        before = self.read(run)
        config = (run / "train_config.json").read_bytes()
        self.assert_waiting(run)
        self.run_cli("--update-waiting", "--poll-seconds", "0.05")
        self.wait_for(lambda: self.started(run))
        after = self.read(run)
        for key in ("id", "name", "submitted_at", "gpus", "pid", "identity"):
            self.assertEqual(before[key], after[key], key)
        self.assertEqual((run / "train_config.json").read_bytes(), config)
        self.finish(run)

    def make_legacy_waiter(self, run):
        job_path = next((self.root / "runs/.train_wait_queue").glob(f"*/{self.read(run)['id']}.json"))
        job = json.loads(job_path.read_text())
        job.pop("monitor_version", None)
        self.write(job_path, job)
        return job_path

    def test_legacy_monitor_handover_preserves_fifo_configs_and_running_task(self):
        self.metrics[3] = [3, 0, 0]
        self.set_metrics(self.metrics)
        running = self.submit("running_untouched", env=dict(self.env, CUDA_VISIBLE_DEVICES="GPU-acceptance-3"))
        self.wait_for(lambda: self.started(running))
        running_before = self.read(running)
        head, second = self.submit("handover_head", 2), self.submit("handover_second")
        configs = [(run / "train_config.json").read_bytes() for run in (head, second)]
        before = [self.read(run) for run in (head, second)]
        for run in (head, second):
            self.make_legacy_waiter(run)
        self.metrics[0] = [0, 29617, 0, 67668]
        self.set_metrics(self.metrics)
        self.run_cli("--update-waiting", "--poll-seconds", "0.05")
        self.assert_waiting(head, second)  # 仍然只有一张可用卡，后面任务不插队
        for run, previous, config in zip((head, second), before, configs):
            current = self.read(run)
            for key in ("id", "name", "submitted_at", "gpus"):
                self.assertEqual(current[key], previous[key], key)
            self.assertNotEqual(current["pid"], previous["pid"])
            self.assertEqual(current["memory_mb"], 32768)
            self.assertEqual(current["min_free_memory_mb"], 61440)
            self.assertEqual((run / "train_config.json").read_bytes(), config)
            self.assertEqual(int((run / "logs/pipeline.pid").read_text()), current["pid"])
        self.assertEqual(self.read(running), running_before)
        self.metrics[1] = [1, 0, 0]
        self.set_metrics(self.metrics)
        self.wait_for(lambda: self.started(head))
        self.assert_waiting(second)
        self.finish(head)
        self.wait_for(lambda: self.started(second))
        self.finish(second)
        self.finish(running)

    def test_legacy_handover_failure_or_termination_restores_original_waiters(self):
        # 两种故障在同一个事务验收中验证：写入失败，以及 SSH/kill 常见的 TERM。
        for fault in ("disk", "term"):
            with self.subTest(fault=fault):
                first, second = self.submit(f"rollback_{fault}_head", 2), self.submit(f"rollback_{fault}_second")
                paths = [self.make_legacy_waiter(run) for run in (first, second)]
                before = [path.read_bytes() for path in paths]
                script = '''import os, signal
from pitrain import gpu_queue as queue
original = queue._write_json
def fail(path, value):
    if path.name == "status.json" and "_second" in str(path):
        if FAULT == "term":
            os.kill(os.getpid(), signal.SIGTERM)
        raise OSError("simulated update failure")
    original(path, value)
queue._write_json = fail
try:
    queue.update_waiting_options(queue.QueueOptions(poll_seconds=0.05))
except (OSError, InterruptedError):
    pass
else:
    raise AssertionError("missing update failure")
'''.replace('FAULT', repr(fault))
                result = subprocess.run([sys.executable, "-c", script], cwd=self.root, env=self.env,
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for path, previous in zip(paths, before):
                    self.assertEqual(path.read_bytes(), previous)
                    pid = json.loads(previous)["pid"]
                    state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                    self.assertNotIn(state, ("T", "t", "Z"))
                # 恢复后原监控仍能按 FIFO 正常启动，不重新提交。
                self.metrics[0], self.metrics[1] = [0, 0, 0], [1, 0, 0]
                self.set_metrics(self.metrics)
                self.wait_for(lambda: self.started(first))
                self.assert_waiting(second)
                self.finish(first)
                self.wait_for(lambda: self.started(second))
                self.finish(second)
                self.metrics = [[index, 8000, 80] for index in range(4)]
                self.set_metrics(self.metrics)


if __name__ == "__main__":
    unittest.main()
