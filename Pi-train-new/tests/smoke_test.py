"""本地端到端冒烟测试（CPU，微型网络，约几分钟）：
预处理好的小数据集 → 训练 6 步(含验证/top-k保存/final) → 评测 → 报告 zip。

    python -m tests.smoke_test --dataset dataset/test_6ep
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_tiny_pretrained(out: Path) -> None:
    import importlib

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    importlib.import_module("tests.smoke_patch")  # 打补丁：替换成微型网络
    cfg = PI05Config(device="cpu", dtype="bfloat16")
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))}
    PI05Policy(cfg).save_pretrained(out)
    for f in (ROOT / "tests/fixtures/pi05_base_processor").glob("*.json"):
        shutil.copy2(f, out / f.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/test_6ep")
    ap.add_argument("--velocity", action="store_true")
    args = ap.parse_args()
    from pitrain.hparams import TrainConfig

    tiny = ROOT / "runs/_tiny_pi05"
    if not (tiny / "model.safetensors").exists():
        build_tiny_pretrained(tiny)
    name = "smoke_vel" if args.velocity else "smoke"
    run = ROOT / "runs" / name
    shutil.rmtree(run, ignore_errors=True)
    (run / "logs").mkdir(parents=True)
    cfg = TrainConfig(
        dataset=str(ROOT / args.dataset),
        output=str(run),
        run_name=name,
        pretrained=str(tiny),
        tokenizer=str(ROOT / "pretrained/paligemma_tokenizer"),
        use_velocity=args.velocity,
        gpus=1,
        batch_size_per_gpu=2,
        steps=6,
        num_workers=0,
        warmup_steps=2,
        val_every=3,
        val_loss_samples=8,
        val_infer_samples=4,
        keep_top_k=1,
        log_every=2,
        num_inference_steps=2,
        gradient_checkpointing=False,
        train_expert_only=True,
        device="cpu",
        scheduler="steplr",
        steplr_step_size=2,
        steplr_gamma=0.5,
    )
    cfg.save(run / "train_config.json")
    env = {
        **os.environ,
        "PITRAIN_TEST_PATCH": "tests.smoke_patch",
        "PYTHONPATH": str(ROOT),
        "HF_HUB_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.call([sys.executable, "-m", "pitrain.pipeline", "--run", str(run)], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
