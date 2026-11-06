"""Local CSV/curve recording around the pinned LeRobot trainer, without changing updates."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
from functools import wraps
import logging
import os
from pathlib import Path
import time

import numpy as np

from acttrain.data.core import write_json

FIELDS = ("step", "epoch", "window_steps", "elapsed_s", "loss", "l1_loss", "kld_loss", "grad_norm", "lr", "update_s", "dataloading_s", "backbone_lr")
VALIDATION_FIELDS = ("step", "epoch", "val_loss", "val_acc", "windows", "validation_s",
                     "train_prior_loss", "train_prior_acc", "train_prior_windows", "train_prior_s")


def append_row(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) if value else float("nan") for key, value in row.items()} for row in csv.DictReader(stream)]


def render_training_curves(csv_path):
    source = Path(csv_path)
    os.environ.setdefault("MPLCONFIGDIR", str(source.parent / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(source)
    if not rows:
        raise ValueError("No recorded training updates to plot")
    has_epoch = "epoch" in rows[0] and all(np.isfinite(row["epoch"]) for row in rows)
    xkey = "epoch" if has_epoch else "step"
    x = [row[xkey] for row in rows]
    validations = read_rows(source.parent / "validation_metrics.csv") if (source.parent / "validation_metrics.csv").exists() else []
    figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    axis.plot(x, [row["l1_loss"] for row in rows], color="#1f77b4", label="train_loss (posterior L1)")
    if validations:
        prior_training = [row for row in validations if np.isfinite(row.get("train_prior_loss", float("nan")))]
        if prior_training:
            axis.plot([row[xkey] for row in prior_training], [row["train_prior_loss"] for row in prior_training],
                      "o-", color="#9467bd", label="train_prior_loss")
        axis.plot([row[xkey] for row in validations], [row["val_loss"] for row in validations], "o--", color="#ff7f0e", label="val_loss")
        accuracy_axis = axis.twinx()
        accuracy_axis.plot([row[xkey] for row in validations], [row["val_acc"] for row in validations], "-.", color="#2ca02c", label="val_acc (gripper)")
        accuracy_axis.set_ylim(0, 1.02)
        accuracy_axis.set_ylabel("Gripper threshold agreement")
        handles, labels = axis.get_legend_handles_labels()
        other_handles, other_labels = accuracy_axis.get_legend_handles_labels()
        axis.legend(handles + other_handles, labels + other_labels, loc="best")
    else:
        axis.legend()
    axis.set_xlabel(xkey)
    axis.set_ylabel("Normalized action L1 loss")
    axis.grid(alpha=.2)
    figure.savefig(source.parent / "training_curves.png", dpi=180)
    figure.savefig(source.parent / "training_curves.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for axis, key in zip(axes.flat, ("loss", "l1_loss", "kld_loss", "grad_norm", "lr", "update_s")):
        axis.plot(x, [row[key] for row in rows], label=key)
        if key == "lr" and "backbone_lr" in rows[0]:
            axis.plot(x, [row["backbone_lr"] for row in rows], label="backbone_lr")
        if key == "update_s":
            axis.plot(x, [row["dataloading_s"] for row in rows], label="dataloading_s")
        axis.set_xlabel(xkey)
        axis.set_title(key)
        axis.grid(alpha=.2)
        axis.legend()
    figure.savefig(source.parent / "diagnostics.png", dpi=160)
    plt.close(figure)
    encoded = base64.b64encode((source.parent / "training_curves.png").read_bytes()).decode()
    diagnostics = base64.b64encode((source.parent / "diagnostics.png").read_bytes()).decode()
    html = ("<!doctype html><meta charset=utf-8><title>ACT 训练曲线</title>"
            "<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%}</style>"
            "<h1>ACT 训练与验证曲线</h1><p>train_loss：训练 batch 的归一化动作重建 L1（使用 CVAE posterior）；"
            "train_prior_loss / val_loss：分别在训练集 / 验证集上以同一权重进行 prior 推理，不提供示范动作，"
            "使用相同训练集std、32步窗口、采样步距和Episode等权口径，可直接对照。"
            "这两项在保存checkpoint时计算；train_prior_loss仅用于观测，不参与反向传播。"
            "posterior train_loss与prior指标条件不同，不能把差值完全解释成过拟合。"
            "旧记录若缺少train_prior_loss，不会补画或估算该曲线。"
            "val_acc 仅为夹爪0.5阈值一致率，不是关节准确率或完整任务成功率；使用右侧0到1坐标。"
            "验证指标按 Episode 等权。epoch 为累计训练样本数除以合法训练窗口数。</p>"
            f'<img alt="train/val curves" src="data:image/png;base64,{encoded}">'
            "<h2>训练诊断</h2><p>总训练 loss 包含 L1+KL加权项，单独绘制，不能直接与 val_loss 比较。"
            "梯度范数是裁剪前数值；时间单位为秒。CSV 每100步汇总一次，并保留首步和末尾不足100步的记录。</p>"
            f'<img alt="training diagnostics" src="data:image/png;base64,{diagnostics}">')
    (source.parent / "curves.html").write_text(html, encoding="utf-8")


class Recorder:
    def __init__(self, output, log_every=100, legal_train_windows=1):
        self.directory = Path(output) / "metrics"
        self.log_every = log_every
        self.legal_train_windows = legal_train_windows
        self.started, self.last_step, self.seen_samples = time.monotonic(), 0, 0
        self.pending = []
        self.tensorboard = None
        self.validation_errors = []
        if log_every < 1 or legal_train_windows < 1:
            raise ValueError("Positive log interval and training window count required")

    def update(self, tracker, output_dict, batch_size=1):
        self.last_step = int(tracker.steps) + 1
        self.seen_samples += batch_size
        values = tracker.to_dict(use_avg=False)
        values.update(output_dict or {})
        self.pending.append({key: float(values.get(key, float("nan"))) for key in FIELDS[4:]})
        if self.last_step == 1 or self.last_step % self.log_every == 0:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        values = {key: float(np.mean([value[key] for value in self.pending])) for key in FIELDS[4:]}
        self.write_window(self.last_step, values, self.seen_samples, len(self.pending))
        self.pending.clear()

    def log_scalars(self, prefix, row, step):
        from torch.utils.tensorboard import SummaryWriter
        if self.tensorboard is None:
            self.tensorboard = SummaryWriter(log_dir=str(self.directory.parent / "tensorboard"))
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.tensorboard.add_scalar(prefix + "/" + key, value, step)
        self.tensorboard.flush()

    def write_window(self, step, values, total_samples, window_steps):
        self.last_step, self.seen_samples = int(step), int(total_samples)
        row = {"step": self.last_step, "epoch": self.seen_samples / self.legal_train_windows,
               "window_steps": window_steps, "elapsed_s": time.monotonic() - self.started,
               **{key: float(values.get(key, float("nan"))) for key in FIELDS[4:]}}
        append_row(self.directory / "train_metrics.csv", FIELDS, row)
        self.log_scalars("train", row, step)

    def checkpoint(self, callback, **kwargs):
        if callback is None:
            return
        try:
            values = callback(**kwargs)
            row = {"step": int(kwargs["step"]), "epoch": self.seen_samples / self.legal_train_windows,
                   "val_loss": values["val_loss"], "val_acc": values["val_acc"], "windows": values["windows"],
                   **{key: values.get(key, float("nan")) for key in VALIDATION_FIELDS[5:]}}
            append_row(self.directory / "validation_metrics.csv", VALIDATION_FIELDS, row)
            self.log_scalars("prior", row, row["step"])
            best_path = self.directory.parent / "best_validation.json"
            if not best_path.exists() or row["val_loss"] < __import__("json").loads(best_path.read_text())["val_loss"]:
                import os
                write_json(best_path, {"step":row["step"], "val_loss":row["val_loss"],
                    "checkpoint":os.path.relpath(kwargs["checkpoint_dir"], self.directory.parent),
                    "criterion":"episode_macro_val_prior_normalized_L1"})
        except Exception as exc:
            self.validation_errors.append({"step": int(kwargs["step"]), "error": str(exc)})
            write_json(self.directory / "validation_errors.json", self.validation_errors)
            raise RuntimeError("checkpoint验证失败，已保留权重；停止后续流水线") from exc

    def finish(self, status):
        if not self.last_step:
            return
        self.flush()
        receipt = {"status": status, "last_step": self.last_step, "seen_samples": self.seen_samples,
                   "validation_failures": len(self.validation_errors), "curves": "pending",
                   "elapsed_s": time.monotonic() - self.started}
        receipt["samples_per_s_including_setup_and_validation"] = self.seen_samples / max(receipt["elapsed_s"], 1e-9)
        try:
            render_training_curves(self.directory / "train_metrics.csv")
            receipt["curves"] = "passed"
        except Exception as exc:
            receipt["curves"] = str(exc)
            logging.warning("CSV retained; curve rendering failed: %s", exc)
        write_json(self.directory / "status.json", receipt)
        if self.tensorboard is not None:
            self.tensorboard.close()


@contextmanager
def capture_training(trainer, output, log_every=100, legal_train_windows=1, validation_callback=None):
    recorder = Recorder(output, log_every, legal_train_windows)
    original_update = trainer.update_policy
    original_save = getattr(trainer, "save_checkpoint", None)

    @wraps(original_update)
    def update(*args, **kwargs):
        tracker, extra = original_update(*args, **kwargs)
        batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
        size = len(batch["action"]) if isinstance(batch, dict) and "action" in batch else 1
        recorder.update(tracker, extra, size)
        return tracker, extra

    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        recorder.checkpoint(validation_callback, **kwargs)
        return result

    trainer.update_policy = update
    if original_save is not None:
        trainer.save_checkpoint = save
    status = "interrupted_or_failed"
    try:
        yield recorder
        status = "completed"
    finally:
        trainer.update_policy = original_update
        if original_save is not None:
            trainer.save_checkpoint = original_save
        try:
            recorder.finish(status)
        except Exception:
            logging.exception("Could not finalize metric files; original training result/exception is preserved")


def main():
    parser = argparse.ArgumentParser(description="Regenerate ACT curves from saved metric CSVs; no training")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    render_training_curves(args.run / "metrics/train_metrics.csv")


if __name__ == "__main__":
    main()
