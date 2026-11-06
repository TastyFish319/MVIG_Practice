"""完整流水线：训练 → 评测 → 报告打包。可前台运行，也可被一键脚本放到后台运行。

    python -m pitrain.pipeline --run runs/<名称>            # 全流程
    python -m pitrain.pipeline --run runs/<名称> --skip-train  # 只重跑评测 + 报告
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .nedf3 import read_json

ROOT = Path(__file__).resolve().parents[1]


def runtime_env() -> dict:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")              # 所有权重/tokenizer 都在本地，禁止联网
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def train_command(cfg_path: Path, gpus: int) -> list[str]:
    if gpus == 1:
        return [sys.executable, "-m", "pitrain.train", "--config", str(cfg_path)]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={gpus}", "-m", "pitrain.train", "--config", str(cfg_path)]


def update_status(run_dir: Path, **kw) -> None:
    path = run_dir / "status.json"
    data = read_json(path) if path.is_file() else {}
    data.update(kw)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}", flush=True)


def run(run_dir: Path, skip_train: bool = False) -> int:
    run_dir = Path(run_dir).resolve()
    cfg = read_json(run_dir / "train_config.json")
    gpus = int(cfg["gpus"])
    env = runtime_env()
    t0 = time.time()
    update_status(run_dir, pipeline="running", pid=os.getpid())
    try:
        if not skip_train:
            banner(f"[1/3] 训练  ({gpus} × GPU)")
            cmd = train_command(run_dir / "train_config.json", gpus)
            print("命令:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        banner("[2/3] 评测（best + final checkpoint）")
        update_status(run_dir, stage="evaluating")
        cmd = [sys.executable, "-m", "pitrain.evaluate", "--run", str(run_dir), "--gpus", str(gpus),
               "--tokenizer", str(ROOT / cfg["tokenizer"]) if not Path(cfg["tokenizer"]).is_absolute() else cfg["tokenizer"]]
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        banner("[3/3] 生成报告并打包")
        update_status(run_dir, stage="packaging")
        from .report import package

        z = package(run_dir)
        update_status(run_dir, stage="done", pipeline="done", zip=str(z),
                      pipeline_hours=round((time.time() - t0) / 3600, 2), done_at=datetime.now(UTC).isoformat())
        banner(f"全部完成 ✔  报告包：{z}  ({z.stat().st_size / 1e6:.1f} MB)")
        return 0
    except BaseException as exc:
        update_status(run_dir, pipeline="failed", error=repr(exc))
        print(f"\n✘ 流水线失败：{exc!r}\n  查看 {run_dir}/status.json 与日志", flush=True)
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)
    return run(args.run, args.skip_train)


if __name__ == "__main__":
    raise SystemExit(main())
