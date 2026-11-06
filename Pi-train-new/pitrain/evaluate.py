"""训练后评测：在验证集上抽 ≤8 条 episode，用每个保留的 checkpoint 逐帧推理、画图、打分、排名。

    python -m pitrain.evaluate --run runs/<名称> --gpus 4

多卡方式：主进程按 GPU 数启动 N 个子进程（各自 CUDA_VISIBLE_DEVICES=i），
每个子进程负责一部分 episode（对所有 checkpoint），最后主进程汇总打分。
不用 NCCL 通信，任何一张卡出问题都容易定位。

每帧 t 的推理结果是一个 chunk：pred[t, 0..49]。由此得到三种曲线：
    step1 曲线   ：pred[t, 0]            —— 模型对"下一步"的判断
    执行轨迹     ：每隔 n 步取一次 chunk 的前 n 步拼起来（模拟真机分段执行，但观测来自真实录像）
    滤波后夹爪   ：执行轨迹的夹爪经软下限滤波（真正会发给机器人的指令）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import ACTION, LEADER, STATE, load_meta
from .gripper import GripperFilter
from .nedf3 import read_json

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 选择
def pick_episodes(val_names: list[str], max_n: int, seed: int) -> list[str]:
    from .split import stratified_split

    if len(val_names) <= max_n:
        return sorted(val_names)
    return stratified_split(val_names, max_n / len(val_names), seed)[:max_n]


def list_checkpoints(run_dir: Path) -> list[dict]:
    root = run_dir / "checkpoints"
    out = []
    topk = read_json(root / "topk.json") if (root / "topk.json").is_file() else {"ranking": []}
    for item in topk["ranking"]:
        if (root / item["dir"]).is_dir():
            out.append({"name": item["dir"], "path": str(root / item["dir"]), "step": item["step"],
                        "tag": f"best#{item['rank']}"})
    if (root / "final").is_dir():
        step = read_json(root / "final/pitrain_contract.json").get("step")
        dup = [c for c in out if c["step"] == step]
        if dup:  # final 恰好也是 best：不重复评测，只加标签
            dup[0]["tag"] += "+final"
        else:
            out.append({"name": "final", "path": str(root / "final"), "step": step, "tag": "final"})
    return out


# ------------------------------------------------------------------ 推理（子进程）
def infer_episode(runner, ds_root: Path, ep_index: int, meta: dict, batch_size: int):
    import torch

    from .data import open_split

    chunk = runner.contract["chunk_size"]
    ds = open_split(ds_root, "val", meta["fps"], chunk)
    ds_ep = type(ds)(ds.repo_id, root=ds.root, episodes=[ep_index],
                     delta_timestamps={ACTION: [i / meta["fps"] for i in range(chunk)]})
    loader = torch.utils.data.DataLoader(ds_ep, batch_size=batch_size, shuffle=False, num_workers=4)
    raws, filt, state, leader, gt_chunk, pad = [], [], [], [], [], []
    for batch in loader:
        state.append(batch[STATE].numpy())
        leader.append(batch[LEADER].numpy())
        gt_chunk.append(batch[ACTION].numpy())
        pad.append(batch["action_is_pad"].numpy())
        out = runner.predict(batch, seed=0)
        raws.append(out["raw"])
        filt.append(out["filtered"])
    return {k: np.concatenate(v) for k, v in
            {"pred": raws, "pred_filtered": filt, "state": state, "leader": leader, "gt": gt_chunk, "pad": pad}.items()}


def worker(args) -> int:
    from .inference import Pi05Runner
    from .train import _test_hook

    _test_hook()
    plan = read_json(Path(args.plan))
    meta = load_meta(Path(plan["dataset"]))
    my_eps = plan["episodes"][args.shard::args.num_shards]
    out_dir = Path(plan["out_dir"])
    for ck in plan["checkpoints"]:
        if not my_eps:
            break
        runner = Pi05Runner(Path(ck["path"]), Path(plan["tokenizer"]), device=plan.get("device", "cuda"),
                            gripper_filter=plan.get("gripper_filter"))
        for ep in my_eps:
            t0 = time.time()
            res = infer_episode(runner, Path(plan["dataset"]), ep["index_in_split"], meta, plan["batch_size"])
            d = out_dir / ck["name"]
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{ep['episode']}.npz", pred=res["pred"], pred_filtered=res["pred_filtered"],
                                state=res["state"], leader=res["leader"], gt=res["gt"], pad=res["pad"])
            print(f"[GPU{args.shard}] {ck['name']} {ep['episode']}: {len(res['pred'])} 帧, {time.time() - t0:.1f}s", flush=True)
        del runner
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


# ------------------------------------------------------------------ 指标
def exec_trajectory(pred: np.ndarray, n_exec: int) -> np.ndarray:
    """模拟"每 n 步重新推理一次"的执行轨迹：t=0,n,2n... 的 chunk 前 n 步依次拼接。"""
    t_len = len(pred)
    out = np.empty((t_len, pred.shape[-1]), np.float64)
    for t in range(0, t_len, n_exec):
        k = min(n_exec, t_len - t)
        out[t:t + k] = pred[t, :k]
    return out


def first_close_time(g: np.ndarray, thr: float, fps: int) -> float | None:
    idx = np.flatnonzero(g < thr)
    return float(idx[0] / fps) if len(idx) else None


def episode_metrics(res: dict, meta: dict, n_exec: int, thr: float, filt: GripperFilter) -> dict:
    from sklearn.metrics import f1_score

    pred, gt, valid = res["pred"].astype(np.float64), res["gt"].astype(np.float64), ~res["pad"].astype(bool)
    names = meta["action_names"]
    grip = np.array([n.endswith("gripper") for n in names])
    joints = ~grip
    err = np.abs(pred - gt)                                       # [T, C, A]
    m_first = valid[:, :n_exec]
    e_first = err[:, :n_exec][m_first]                            # [N, A]
    e_full = err[valid]
    leader = gt[:, 0]                                             # 主臂真值 [T, A]
    traj = exec_trajectory(pred, n_exec)
    e_traj = np.abs(traj - leader)
    # chunk 衔接处跳变：执行轨迹在边界处的步长，减去主臂真值在同一处的自然步长
    b = np.arange(n_exec, len(traj), n_exec)
    jump = np.abs(traj[b] - traj[b - 1])[:, joints] - np.abs(leader[b] - leader[b - 1])[:, joints]
    out = {
        "frames": len(pred),
        "joint_mae_first_deg": float(np.rad2deg(e_first[:, joints]).mean()),
        "joint_mae_full_deg": float(np.rad2deg(e_full[:, joints]).mean()),
        "joint_p95_first_deg": float(np.rad2deg(np.percentile(e_first[:, joints], 95))),
        "exec_traj_mae_deg": float(np.rad2deg(e_traj[:, joints]).mean()),
        "boundary_jump_deg": float(np.rad2deg(np.clip(jump, 0, None)).mean()) if len(b) else 0.0,
        "per_joint_mae_first": {n: float(np.rad2deg(e_first[:, i]).mean()) if not grip[i] else float(e_first[:, i].mean())
                                for i, n in enumerate(names)},
    }
    if grip.any():
        gt_g, pr_g = leader[:, grip], traj[:, grip]
        out["gripper_mae_first"] = float(e_first[:, grip].mean())
        y_true, y_pred = (gt_g < thr).ravel(), (pr_g < thr).ravel()
        # 两边都没有"闭合"帧时 F1 无定义，视为完全一致
        both_empty = not y_true.any() and not y_pred.any()
        out["gripper_f1"] = 1.0 if both_empty else float(f1_score(y_true, y_pred))
        timings = []
        for c in range(gt_g.shape[1]):
            a, p = first_close_time(gt_g[:, c], thr, meta["fps"]), first_close_time(pr_g[:, c], thr, meta["fps"])
            if a is None and p is None:
                timings.append(0.0)
            elif a is None or p is None:
                timings.append(2.0)          # 一方闭合一方不闭合：记 2 秒的大偏差
            else:
                timings.append(abs(a - p))
        out["grasp_timing_s"] = float(np.mean(timings))
        filtered = filt(pr_g)
        out["gripper_filtered_min"] = float(filtered.min())
    return out


def score(metrics: dict, scoring: dict) -> tuple[float, dict]:
    subs, total, wsum = {}, 0.0, 0.0
    for key, spec in scoring["metrics"].items():
        if key not in metrics:
            continue
        v = metrics[key]
        s = 100.0 * v if spec["tolerance"] is None else 100.0 * float(np.exp(-v / spec["tolerance"]))
        subs[key] = round(s, 2)
        total += spec["weight"] * s
        wsum += spec["weight"]
    return round(total / max(wsum, 1e-9), 2), subs


# ------------------------------------------------------------------ 画图
JOINT_COLORS = {"leader": "#222222", "follower": "#9E9E9E", "pred": "#1f77b4", "step1": "#8fbbe0", "filtered": "#ff7f0e"}


def plot_episode(res: dict, meta: dict, n_exec: int, filt: GripperFilter, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = meta["action_names"]
    a = len(names)
    t = np.arange(len(res["pred"])) / meta["fps"]
    traj = exec_trajectory(res["pred"], n_exec)
    cols = 2 if a > 7 else 1
    rows = int(np.ceil(a / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13 if cols == 1 else 18, 2.1 * rows), sharex=True, squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i % rows, i // rows]
        is_g = name.endswith("gripper")
        conv = (lambda x: x) if is_g else np.rad2deg
        ax.plot(t, conv(res["leader"][:, i]), color=JOINT_COLORS["leader"], lw=1.6, label="leader (target)")
        ax.plot(t, conv(res["state"][:, i]), "--", color=JOINT_COLORS["follower"], lw=1.2, label="follower (state)")
        ax.plot(t, conv(res["pred"][:, 0, i]), color=JOINT_COLORS["step1"], lw=0.8, alpha=0.8, label="pred step-1")
        ax.plot(t, conv(traj[:, i]), color=JOINT_COLORS["pred"], lw=1.4, label=f"pred executed (n={n_exec})")
        if is_g:
            ax.plot(t, filt(traj[:, i]), color=JOINT_COLORS["filtered"], lw=1.4, label="gripper cmd (filtered)")
            ax.axhline(filt.floor, color=JOINT_COLORS["filtered"], ls=":", lw=0.8)
        ax.set_ylabel(f"{name}\n{'[0-1]' if is_g else '[deg]'}", fontsize=8)
        ax.grid(alpha=0.25)
        if i == 0 or (is_g and i == a - 1):
            ax.legend(fontsize=7, loc="upper right", ncol=5 if is_g else 4)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def plot_compare(results: dict[str, dict], meta: dict, n_exec: int, title: str, path: Path) -> None:
    """同一条 episode，不同 checkpoint 的执行轨迹叠在一起比较。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = meta["action_names"]
    cols = 2 if len(names) > 7 else 1
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13 if cols == 1 else 18, 2.0 * rows), sharex=True, squeeze=False)
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    first = next(iter(results.values()))
    t = np.arange(len(first["pred"])) / meta["fps"]
    for i, name in enumerate(names):
        ax = axes[i % rows, i // rows]
        conv = (lambda x: x) if name.endswith("gripper") else np.rad2deg
        ax.plot(t, conv(first["leader"][:, i]), color="#222222", lw=1.8, label="leader (target)")
        for j, (ck, res) in enumerate(results.items()):
            ax.plot(t, conv(exec_trajectory(res["pred"], n_exec)[:, i]), color=palette[j % len(palette)], lw=1.1, label=ck)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=len(results) + 1)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


# ------------------------------------------------------------------ 主流程
def run(run_dir: Path, gpus: int, tokenizer: Path, batch_size: int = 32, max_episodes: int | None = None) -> Path:
    run_dir = Path(run_dir)
    cfg = read_json(run_dir / "train_config.json")
    meta = load_meta(Path(cfg["dataset"]))
    scoring = read_json(ROOT / "configs/eval_scoring.json")
    max_n = max_episodes or scoring["max_episodes"]
    out_dir = run_dir / "eval"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"{run_dir}/checkpoints 下没有可评测的权重")
    val_recs = [r for r in meta["episodes"] if r["split"] == "val"]
    chosen = pick_episodes([r["episode"] for r in val_recs], max_n, cfg.get("seed", 42))
    episodes = [r for r in val_recs if r["episode"] in chosen]
    gf = meta["gripper_profile"]["suggested"]
    plan = {"dataset": cfg["dataset"], "tokenizer": str(tokenizer), "out_dir": str(pred_dir),
            "checkpoints": ckpts, "episodes": episodes, "batch_size": batch_size, "gripper_filter": gf,
            "device": cfg.get("device", "cuda")}
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"评测 {len(ckpts)} 个 checkpoint × {len(episodes)} 条 episode，使用 {gpus} 张 GPU", flush=True)

    # 1) 多 GPU 并行推理
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devs = visible.split(",") if visible else [str(i) for i in range(gpus)]
    procs = []
    for i in range(min(gpus, len(episodes))):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": devs[i % len(devs)]}
        cmd = [sys.executable, "-m", "pitrain.evaluate", "worker", "--plan", str(out_dir / "plan.json"),
               "--shard", str(i), "--num-shards", str(min(gpus, len(episodes)))]
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(ROOT)))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"评测子进程失败，返回码 {codes}")

    # 2) 指标、打分、画图
    n_exec = cfg["n_action_steps"]
    thr = scoring["gripper_closed_threshold"]
    filt = GripperFilter(**gf)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    summary = {"checkpoints": [], "episodes": [e["episode"] for e in episodes], "n_exec": n_exec,
               "gripper_filter": gf, "scoring": scoring}
    all_res: dict[str, dict[str, dict]] = {}
    for ck in ckpts:
        per_ep = {}
        for ep in episodes:
            with np.load(pred_dir / ck["name"] / f"{ep['episode']}.npz") as z:
                res = {k: z[k] for k in z.files}
            all_res.setdefault(ep["episode"], {})[f"{ck['name']} ({ck['tag']})"] = res
            per_ep[ep["episode"]] = episode_metrics(res, meta, n_exec, thr, filt)
            plot_episode(res, meta, n_exec, filt, f"{ck['name']} [{ck['tag']}]  |  {ep['episode']}",
                         plot_dir / f"{ck['name']}__{ep['episode']}.png")
        keys = [k for k in scoring["metrics"] if k in next(iter(per_ep.values()))]
        agg = {k: float(np.mean([m[k] for m in per_ep.values()])) for k in keys}
        joints_tbl = {n: float(np.mean([m["per_joint_mae_first"][n] for m in per_ep.values()])) for n in meta["action_names"]}
        total, subs = score(agg, scoring)
        summary["checkpoints"].append({**ck, "score": total, "subscores": subs, "metrics": agg,
                                       "per_joint_mae_first": joints_tbl, "per_episode": per_ep})
    for ep, res in all_res.items():
        plot_compare(res, meta, n_exec, f"{ep}: executed trajectories of all checkpoints", plot_dir / f"compare__{ep}.png")
    summary["checkpoints"].sort(key=lambda c: -c["score"])
    for r, c in enumerate(summary["checkpoints"], 1):
        c["rank"] = r
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "ranking.csv", summary)
    print("排名：", [(c["name"], c["score"]) for c in summary["checkpoints"]], flush=True)
    return out_dir


def _write_csv(path: Path, summary: dict) -> None:
    import csv

    keys = list(summary["checkpoints"][0]["metrics"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "checkpoint", "tag", "step", "score", *keys])
        for c in summary["checkpoints"]:
            w.writerow([c["rank"], c["name"], c["tag"], c["step"], c["score"], *[round(c["metrics"][k], 4) for k in keys]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("worker")
    w.add_argument("--plan", required=True)
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--gpus", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "pretrained/paligemma_tokenizer")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args(argv)
    if args.cmd == "worker":
        return worker(args)
    if args.run is None:
        ap.error("需要 --run")
    run(args.run, args.gpus, args.tokenizer, args.batch_size, args.max_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
