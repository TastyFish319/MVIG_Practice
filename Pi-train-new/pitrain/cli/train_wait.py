"""复用一键训练的交互；配置完成后提交 GPU 等待队列。"""

from __future__ import annotations

import argparse
from functools import partial

from .. import console as ui
from ..gpu_queue import QueueOptions, queue_status, submit, update_waiting_options
from . import train


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        epilog="其余参数与 train.sh 完全相同；--dry-run 只生成配置，不入队。",
    )
    ap.add_argument("-h", "--help", action="store_true")
    ap.add_argument("--idle-memory-mb", type=int, default=32768, help="空闲显存占用上限（MiB，默认 32768 = 32 GiB）")
    ap.add_argument("--min-free-memory-mb", type=int, default=61440, help="启动前每卡最少剩余显存（MiB，默认 61440 = 60 GiB）")
    ap.add_argument("--idle-utilization", type=int, default=10, help="空闲 GPU 利用率上限（%%，默认 10）")
    ap.add_argument("--poll-seconds", type=float, default=10, help="检测间隔秒数（默认 10）")
    ap.add_argument("--queue-status", action="store_true", help="查看当前服务器、本项目的等待/运行队列")
    ap.add_argument("--update-waiting", action="store_true", help="将调度阈值应用到等待任务，保留任务/顺序/训练配置，不动运行任务")
    args, training_args = ap.parse_known_args(argv)
    if args.help:
        ap.print_help()
        print("\n原有训练参数：")
        return train.main(["--help"])
    try:
        options = QueueOptions(args.idle_memory_mb, args.idle_utilization, args.poll_seconds, args.min_free_memory_mb)
    except ValueError as exc:
        ap.error(str(exc))
    if args.update_waiting:
        changes = update_waiting_options(options)
        ui.table("已更新等待任务的调度阈值（训练配置与队列顺序不变）", ["名称", "原监控 PID", "监控 PID"],
                 [[entry["name"], entry["old_pid"], entry["pid"]] for entry in changes])
        return 0
    if args.queue_status:
        ui.table(
            "GPU 等待队列（按提交顺序）",
            ["序号", "名称", "状态", "卡数", "GPU 索引", "监控 PID"],
            [
                [i, job["name"], job["state"], job["gpus"], job.get("gpu_indices", []), job["pid"]]
                for i, job in enumerate(queue_status(), 1)
            ],
        )
        return 0
    return train.main(training_args, launch=partial(submit, options=options))


if __name__ == "__main__":
    raise SystemExit(main())
