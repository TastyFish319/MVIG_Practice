"""一键预处理（交互式）。

python -m pitrain.cli.preprocess            # 全交互
python -m pitrain.cli.preprocess --raw-folder blue_block_v2 --yes   # 全部用默认值
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .. import console as ui
from ..nedf3 import read_json
from ..preprocess import META_FILE, PreprocessConfig, run_preprocess
from ..review import find_review_files, load_approved
from ..split import parse_manual, stratified_split

ROOT = Path(__file__).resolve().parents[2]
TOTAL_STEPS = 6


def load_defaults() -> dict:
    try:
        return read_json(ROOT / "configs/preprocess_defaults.json")
    except ValueError as exc:
        raise SystemExit(f"默认配置读取失败：{exc}") from exc


def discover_raw(raw_root: Path) -> list[tuple[Path, int, int]]:
    found = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        n = sum(1 for e in d.iterdir() if e.is_dir() and e.name.startswith("episode_"))
        if n:
            found.append((d, n, len(find_review_files(d, raw_root))))
    return found


def progress_bar() -> Progress:
    return Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("剩余"),
        TimeRemainingColumn(),
        console=ui.console,
    )


def default_prompt(defaults: dict, folder: str) -> str:
    for prefix, prompt in defaults["task_prompts"].items():
        if folder.startswith(prefix):
            return prompt
    return defaults["default_prompt"]


def show_split(names: list[str], val: list[str]) -> None:
    rows = []
    vs = set(val)
    for i in range(0, len(names), 10):
        chunk = names[i : i + 10]
        rows.append(
            [" ".join(f"[bold yellow]{n[-4:]}[/bold yellow]" if n in vs else f"[dim]{n[-4:]}[/dim]" for n in chunk)]
        )
    ui.table(
        f"划分预览：训练 {len(names) - len(val)} 条 / 验证 {len(val)} 条（[bold yellow]黄色[/bold yellow] = 验证集）",
        ["episode 编号（按顺序）"],
        rows,
    )
    ui.console.print("[key]验证集序号：[/key]" + ", ".join(n[-4:] for n in val))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=ROOT / "raw_data")
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    ap.add_argument("--raw-folder", help="raw_data 下的数据文件夹名")
    ap.add_argument("--name", help="输出数据集名称")
    ap.add_argument("--val-ratio", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arms", choices=["right", "both"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int, help="只处理前 N 条（调试用）")
    ap.add_argument("--no-verify-hash", action="store_true", help="跳过源文件 sha256 校验（更快）")
    ap.add_argument("--yes", action="store_true", help="所有询问都采用默认值/自动确认")
    args = ap.parse_args(argv)
    d = load_defaults()
    yes = args.yes

    ui.banner("π0.5 数据预处理", "raw_data → 审核筛选 → 划分 → 对齐/缩放 → dataset (LeRobot v3)")

    # 1. 选择原始数据
    ui.step(1, TOTAL_STEPS, "选择原始数据文件夹")
    raws = discover_raw(args.raw_root)
    if not raws:
        ui.error(f"{args.raw_root} 下没有包含 episode_* 的文件夹")
        return 1
    if args.raw_folder:
        match = [r for r in raws if r[0].name == args.raw_folder]
        if not match:
            ui.error(f"找不到 {args.raw_folder}")
            return 1
        folder = match[0][0]
    else:
        ui.table(
            "raw_data 中的数据",
            ["#", "文件夹", "episode 数", "审核记录"],
            [[i, r[0].name, r[1], "✔" if r[2] else "[err]缺失[/err]"] for i, r in enumerate(raws, 1)],
        )
        folder = raws[0 if yes else ui.choose("选择编号", [r[0].name for r in raws])][0]
    ui.ok(f"数据文件夹：{folder}")

    # 2. 审核记录
    ui.step(2, TOTAL_STEPS, "读取审核记录")
    pairs = find_review_files(folder, args.raw_root)
    if not pairs:
        ui.error("没有找到成对的 *_approved.json / *_review.json，请放到数据文件夹内")
        return 1
    pair = (
        pairs[0]
        if (len(pairs) == 1 or yes)
        else pairs[ui.choose("选择审核记录", [p.approved_file.name for p in pairs])]
    )
    verify = not args.no_verify_hash
    n_ep = sum(1 for e in folder.iterdir() if e.name.startswith("episode_"))
    with progress_bar() as prog:
        task_id = prog.add_task("校验源文件 sha256" if verify else "检查 episode", total=n_ep)
        task, episodes, stats = load_approved(folder, pair, verify_hash=verify, progress=lambda: prog.advance(task_id))
        prog.update(task_id, total=len(episodes))
    ui.kv_table(
        "审核统计",
        {
            "任务": task,
            "审核记录": pair.approved_file.name,
            "总数": stats["total"],
            "通过": f"[ok]{stats['approved']}[/ok]",
            "拒绝": stats["rejected"],
            "通过但有技术警告": stats["warning_approved"],
        },
    )
    if args.limit:
        episodes = episodes[: args.limit]
        ui.warn(f"调试模式：只处理前 {len(episodes)} 条")

    # 3. 处理参数
    ui.step(3, TOTAL_STEPS, "预处理参数（回车 = 默认值）")
    prompt = default_prompt(d, folder.name)
    if not yes:
        prompt = ui.ask_value("任务语言指令 (prompt)", prompt, hint="pi0.5 是视觉-语言-动作模型，需要一句英文任务描述")
    arms = args.arms or ("right" if d["action_arms"] == ["right"] else "both")
    if not yes and not args.arms:
        arms = ["right", "both"][
            ui.choose("动作（模型输出）包含哪些臂？", ["仅右臂 7 维（推荐，左臂本任务几乎不动）", "双臂 14 维"], 1)
        ]
    fps = d["fps"] if yes else ui.ask_int("采样频率 fps", d["fps"], 5, 90, hint="相机原生 30Hz")
    trim = d["trim_idle"] if yes else ui.confirm("裁剪首尾静止段？（避免模型学会开局发呆）", d["trim_idle"])

    # 4. 划分
    ui.step(4, TOTAL_STEPS, "划分训练集 / 验证集")
    names = [e.name for e in episodes]
    ratio = (
        args.val_ratio
        if args.val_ratio is not None
        else (d["val_ratio"] if yes else ui.ask_float("验证集占比", d["val_ratio"], 0.01, 0.9))
    )
    seed = args.seed if args.seed is not None else d["seed"]
    val = stratified_split(names, ratio, seed)
    while True:
        show_split(names, val)
        if yes:
            break
        ans = ui.ask_value(
            "确认？ [y]确认  [r]换一组随机  [m]手动输入  [q]退出",
            "y",
            valid=lambda v: v.lower() in ("y", "r", "m", "q"),
        ).lower()
        if ans == "y":
            break
        if ans == "q":
            ui.warn("已退出，未写入任何文件")
            return 130
        if ans == "r":
            seed += 1
            val = stratified_split(names, ratio, seed)
            ui.info(f"新随机种子 seed={seed}")
        else:
            try:
                val = parse_manual(
                    ui.ask_value("输入验证集编号（空格/逗号分隔，如 3 15 27）", " ".join(n[-4:] for n in val)), names
                )
            except ValueError as exc:
                ui.error(str(exc))
    ui.ok(f"验证集 {len(val)} 条已确认")

    # 5. 输出名称
    ui.step(5, TOTAL_STEPS, "确认输出")
    default_name = f"{folder.name}_{arms}_{fps}hz_{len(names) - len(val)}tr{len(val)}va"
    name = args.name or (
        default_name
        if yes
        else ui.ask_value("数据集名称", default_name, valid=lambda v: bool(re.fullmatch(r"[\w.-]+", v)))
    )
    out = args.dataset_root / name
    if out.exists():
        ui.error(f"{out} 已存在；请换名字或先删除")
        return 1
    cfg = PreprocessConfig(
        raw_folder=folder.name,
        output=str(out),
        prompt=prompt,
        val_episodes=val,
        fps=fps,
        image_size=d["image_size"],
        cameras=d["cameras"],
        action_arms=["right"] if arms == "right" else ["right", "left"],
        max_age_ms=d["max_age_ms"],
        velocity_window_s=d["velocity_window_s"],
        trim_idle=trim,
        idle_joint_tol_rad=d["idle_joint_tol_rad"],
        idle_grip_tol=d["idle_grip_tol"],
        idle_margin_s=d["idle_margin_s"],
        workers=args.workers or d["workers"],
        seed=seed,
        val_ratio=ratio,
    )
    ui.kv_table(
        "即将执行",
        {
            "输出": out,
            "prompt": prompt,
            "动作臂": cfg.action_arms,
            "fps": fps,
            "相机": cfg.cameras,
            "图像": f"{cfg.image_size}×{cfg.image_size}",
            "训练/验证": f"{len(names) - len(val)} / {len(val)}",
            "并行进程": cfg.workers,
        },
    )
    if not yes and not ui.confirm("开始预处理？", True):
        ui.warn("已取消")
        return 130

    # 6. 执行
    ui.step(6, TOTAL_STEPS, "处理中（每条 episode：解码 → 对齐 → 缩放 → 写入）")
    with progress_bar() as prog:
        tid = prog.add_task("episodes", total=len(episodes))

        def on_episode(rec, split):
            prog.advance(tid)
            prog.console.print(
                f"  [dim]{rec['episode']}[/dim] → [{'yellow' if split == 'val' else 'green'}]{split}[/]"
                f"  {rec['frames']} 帧 ({rec['seconds']}s)，裁剪 {rec['trim']} / {rec['raw_frames']}"
            )

        try:
            out = run_preprocess(cfg, episodes, task, (pair.approved_file, pair.review_file), on_episode)
        except Exception as exc:
            ui.error(f"预处理失败：{exc}")
            ui.info(f"半成品保留在 {out}.partial 便于排查")
            raise
    try:
        meta = read_json(out / META_FILE)
    except ValueError as exc:
        ui.error(f"数据集已生成，但读取 {META_FILE} 失败：{exc}")
        return 1
    gp = meta["gripper_profile"]
    ui.kv_table(
        "完成 ✔",
        {
            "数据集": out,
            "训练集": f"{meta['counts']['train']['episodes']} 条 / {meta['counts']['train']['frames']} 帧",
            "验证集": f"{meta['counts']['val']['episodes']} 条 / {meta['counts']['val']['frames']} 帧",
            "state 维度": f"{len(meta['state_names'])}（+速度可选 {len(meta['velocity_names'])}）",
            "action 维度": len(meta["action_names"]),
            "夹爪画像": f"抓持时从臂中位 {gp.get('follower_hold_median', float('nan')):.3f}，"
            f"主臂中位 {gp.get('leader_hold_median', float('nan')):.3f}",
            "建议滤波": gp["suggested"],
            "耗时": f"{meta['elapsed_s']}s",
        },
    )
    ui.ok("下一步：bash scripts/train.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
