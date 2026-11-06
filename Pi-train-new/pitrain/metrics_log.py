"""训练指标记录：TensorBoard（成熟、可交互）+ CSV（可长期保存）+ PNG（报告用）。"""
from __future__ import annotations

import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: Path, enabled: bool):
        self.enabled = enabled
        self.dir = Path(run_dir) / "metrics"
        self.tb = None
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(Path(run_dir) / "tensorboard"))

    def _append(self, name: str, row: dict) -> None:
        path = self.dir / name
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def log(self, kind: str, step: int, values: dict) -> None:
        if not self.enabled:
            return
        self._append(f"{kind}.csv", {"step": step, **values})
        if self.tb is not None:
            for k, v in values.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"{kind}/{k}", v, step)
            self.tb.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


def read_csv(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def plot_curves(run_dir: Path, best_steps: list[int] | None = None) -> Path | None:
    """生成 training_curves.png：loss / 验证误差 / 学习率 / 梯度范数。

    服务器没有中文字体，所以图内文字一律用英文；中文解释放在 HTML 报告里。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train = read_csv(run_dir / "metrics/train.csv")
    val = read_csv(run_dir / "metrics/val.csv")
    if not train:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    s = [r["step"] for r in train]
    ax = axes[0, 0]
    ax.plot(s, [r["loss"] for r in train], color="#4C72B0", alpha=0.35, lw=0.8, label="train loss")
    ax.plot(s, _ema([r["loss"] for r in train]), color="#4C72B0", lw=1.8, label="train loss (EMA)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", color="#DD8452", lw=1.8, label="val loss")
    ax.set_yscale("log")
    ax.set_title("Flow-matching loss (lower is better)")
    ax.legend()
    ax = axes[0, 1]
    if val:
        vs = [r["step"] for r in val]
        ax.plot(vs, [r["val_joint_mae_deg_first"] for r in val], "o-", color="#55A868", label="joint MAE, first n steps (deg)")
        ax.plot(vs, [r["val_joint_mae_deg_full"] for r in val], "s--", color="#8172B3", label="joint MAE, full chunk (deg)")
        ax2 = ax.twinx()
        ax2.plot(vs, [r["val_nmae"] for r in val], "^-", color="#C44E52", label="val_nmae (best selection)")
        ax2.set_ylabel("val_nmae")
        for b in best_steps or []:
            ax.axvline(b, color="gold", lw=2, alpha=0.6)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    ax.set_title("Validation inference error (gold = kept best checkpoints)")
    axes[1, 0].plot(s, [r["lr"] for r in train], color="#937860")
    axes[1, 0].set_title("Learning rate")
    axes[1, 1].plot(s, [r["grad_norm"] for r in train], color="#DA8BC3", alpha=0.6)
    axes[1, 1].set_title("Gradient norm (before clipping)")
    for a in axes.flat:
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / "metrics/training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _ema(xs: list[float], alpha: float = 0.1) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out

