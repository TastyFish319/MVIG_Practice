"""生成 HTML 报告并打包 <训练名称>_statistics.zip（只含审阅必需的轻量文件）。

zip 内容::

    report.html                 主报告（双击打开，图片用相对路径引用）
    metrics/training_curves.png 训练曲线
    metrics/train.csv / val.csv 原始数值
    eval/plots/*.png            分关节轨迹图
    eval/summary.json / ranking.csv
    train_config.json           本次超参数
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from .metrics_log import read_csv
from .nedf3 import read_json

TEMPLATE = Template(r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ name }} · π0.5 训练报告</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#1f2430;--sub:#6b7280;--acc:#5b5bd6;--ok:#16a34a;--warn:#d97706;--line:#e5e7eb}
*{box-sizing:border-box}body{margin:0;font:14px/1.6 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(120deg,#4f46e5,#7c3aed);color:#fff;padding:28px 40px}
header h1{margin:0;font-size:24px}header p{margin:4px 0 0;opacity:.85}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:8px 40px;z-index:5}
nav a{margin-right:18px;color:var(--acc);text-decoration:none;font-weight:600}
main{max-width:1500px;margin:auto;padding:24px 40px}
.card{background:var(--card);border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.06);padding:20px 24px;margin-bottom:22px}
h2{margin:0 0 12px;font-size:18px;border-left:4px solid var(--acc);padding-left:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:#f3f4ff;border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:20px;color:var(--acc)}.kpi span{color:var(--sub);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}
th{background:#fafafa;color:var(--sub);font-weight:600;position:sticky;top:42px}td:first-child,th:first-child{text-align:left}
tr.top td{background:#f0fdf4;font-weight:600}
.bar{height:8px;border-radius:4px;background:#e5e7eb;overflow:hidden;min-width:80px}.bar i{display:block;height:100%;background:linear-gradient(90deg,#f59e0b,#16a34a)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;background:#eef2ff;color:#4338ca;margin-left:4px}
img{max-width:100%;border-radius:8px;border:1px solid var(--line)}
details{margin:8px 0}summary{cursor:pointer;font-weight:600;padding:6px 0}
.note{color:var(--sub);font-size:13px}.muted{color:var(--sub)}
.tabs input{display:none}.tabs label{display:inline-block;padding:6px 14px;margin:0 6px 8px 0;border-radius:8px;background:#eef2ff;cursor:pointer;font-weight:600}
.tabs input:checked+label{background:var(--acc);color:#fff}.tabpane{display:none}
{% for c in ckpts %}#t{{ loop.index }}:checked~#p{{ loop.index }}{display:block}{% endfor %}
</style></head><body>
<header><h1>π0.5 训练报告 · {{ name }}</h1><p>生成时间 {{ now }} · 数据集 {{ cfg.dataset_name }} · {{ cfg.gpus }}×GPU</p></header>
<nav><a href="#overview">概览</a><a href="#curves">训练曲线</a><a href="#ranking">评测排名</a><a href="#joints">分关节误差</a><a href="#plots">轨迹图</a><a href="#help">指标说明</a></nav>
<main>
<section class="card" id="overview"><h2>概览</h2>
<div class="grid">
<div class="kpi"><b>{{ best.name if best else '-' }}</b><span>综合评分第一的 checkpoint</span></div>
<div class="kpi"><b>{{ best.score if best else '-' }}</b><span>最高综合评分 / 100</span></div>
<div class="kpi"><b>{{ "%.2f"|format(best.metrics.joint_mae_first_deg) if best else '-' }}°</b><span>近期关节误差（前 {{ cfg.n_action_steps }} 步）</span></div>
<div class="kpi"><b>{{ status.hours or '-' }} h</b><span>训练耗时</span></div>
<div class="kpi"><b>{{ cfg.steps }}</b><span>训练步数</span></div>
<div class="kpi"><b>{{ cfg.batch_size_per_gpu * cfg.gpus }}</b><span>全局 batch</span></div>
<div class="kpi"><b>{{ "%.1e"|format(cfg.lr) }}</b><span>初始学习率（{{ cfg.scheduler }}）</span></div>
<div class="kpi"><b>{{ '是' if cfg.use_velocity else '否' }}</b><span>state 是否含速度</span></div>
</div>
<details><summary>全部超参数</summary><table>{% for k,v in cfg_items %}<tr><td>{{k}}</td><td>{{v}}</td></tr>{% endfor %}</table></details>
</section>

<section class="card" id="curves"><h2>训练曲线</h2>
<img src="metrics/training_curves.png" alt="curves">
<p class="note">左上：训练/验证 flow-matching loss（对数坐标）。右上：验证集真实推理误差，金色竖线为保留的 best 权重。
更详细的交互曲线：服务器上运行 <code>tensorboard --logdir runs/{{ name }}/tensorboard</code>。</p>
{% if val_rows %}<details><summary>每次验证的数值</summary><table><tr><th>step</th><th>val_loss</th><th>val_nmae</th><th>关节MAE前n步(°)</th><th>关节MAE整段(°)</th><th>夹爪MAE</th><th>保存</th></tr>
{% for r in val_rows %}<tr><td>{{ r.step|int }}</td><td>{{ "%.4f"|format(r.val_loss) }}</td><td>{{ "%.4f"|format(r.val_nmae) }}</td><td>{{ "%.2f"|format(r.val_joint_mae_deg_first) }}</td><td>{{ "%.2f"|format(r.val_joint_mae_deg_full) }}</td><td>{{ "%.3f"|format(r.val_gripper_mae_first) }}</td><td>{{ '★' if r.kept|int else '' }}</td></tr>{% endfor %}</table></details>{% endif %}
</section>

{% if summary %}
<section class="card" id="ranking"><h2>评测排名（{{ summary.episodes|length }} 条验证 episode）</h2>
<table><tr><th>排名</th><th>checkpoint</th><th>综合分</th><th></th>{% for k,s in metric_specs %}<th title="{{s.label}}">{{ s.label }}</th>{% endfor %}</tr>
{% for c in summary.checkpoints %}<tr class="{{ 'top' if c.rank==1 }}"><td>#{{c.rank}}</td><td>{{c.name}}<span class="tag">{{c.tag}}</span></td><td>{{c.score}}</td>
<td><div class="bar"><i style="width:{{c.score}}%"></i></div></td>
{% for k,s in metric_specs %}<td>{% if k in c.metrics %}{{ "%.3f"|format(c.metrics[k]) }}{{s.unit}}<br><span class="muted">{{ c.subscores[k] }}分</span>{% else %}-{% endif %}</td>{% endfor %}</tr>{% endfor %}
</table>
<p class="note">评测 episode：{{ summary.episodes|join(', ') }}。执行步长 n = {{ summary.n_exec }}；夹爪滤波 knee={{ summary.gripper_filter.knee }}, floor={{ summary.gripper_filter.floor }}。</p>
</section>

<section class="card" id="joints"><h2>分关节误差（前 n 步平均；关节单位 °，夹爪单位 0~1）</h2>
<table><tr><th>checkpoint</th>{% for j in joint_names %}<th>{{ j }}</th>{% endfor %}</tr>
{% for c in summary.checkpoints %}<tr><td>{{ c.name }}</td>{% for j in joint_names %}<td>{{ "%.3f"|format(c.per_joint_mae_first[j]) }}</td>{% endfor %}</tr>{% endfor %}</table>
</section>

<section class="card" id="plots"><h2>轨迹图</h2>
<p class="note">黑实线 = 主臂(leader，学习目标)；灰虚线 = 从臂(follower，模型输入)；浅蓝 = 每帧预测的下一步；
深蓝 = 每 n 步重新推理的执行轨迹；橙色 = 滤波后真正发给夹爪的指令（橙色点线为下限 floor）。</p>
<div class="tabs">
{% for c in ckpts %}<input type="radio" name="tab" id="t{{loop.index}}" {{ 'checked' if loop.first }}><label for="t{{loop.index}}">#{{c.rank}} {{c.name}}</label>{% endfor %}
<input type="radio" name="tab" id="tcmp"><label for="tcmp">checkpoint 对比</label>
{% for c in ckpts %}<div class="tabpane" id="p{{loop.index}}">{% for ep in summary.episodes %}
<details {{ 'open' if loop.first }}><summary>{{ ep }} — 关节MAE(前n步) {{ "%.2f"|format(c.per_episode[ep].joint_mae_first_deg) }}°</summary><img loading="lazy" src="eval/plots/{{c.name}}__{{ep}}.png"></details>{% endfor %}</div>{% endfor %}
<div class="tabpane" id="pcmp">{% for ep in summary.episodes %}<details {{ 'open' if loop.first }}><summary>{{ ep }}</summary><img loading="lazy" src="eval/plots/compare__{{ep}}.png"></details>{% endfor %}</div>
</div>
<style>#tcmp:checked~#pcmp{display:block}</style>
</section>
{% endif %}

<section class="card" id="help"><h2>指标说明</h2>
<ul>
<li><b>综合分</b> = Σ 权重 × 子分。误差类子分 = 100·exp(−误差/容差)（误差=容差时 37 分，误差=容差/2 时 61 分）；F1 类子分 = 100·F1。权重与容差在 <code>configs/eval_scoring.json</code> 可调。</li>
{% for k,s in metric_specs %}<li><b>{{ s.label }}</b>（权重 {{ s.weight }}{% if s.tolerance %}，容差 {{ s.tolerance }}{{ s.unit }}{% endif %}）：{{ explain[k] }}</li>{% endfor %}
<li>注意：这是<b>离线开环</b>评测（观测来自真实录像），不等于真机闭环成功率；它衡量"模型在示范状态下能否复现主臂动作"。</li>
</ul></section>
</main></body></html>
""")

EXPLAIN = {
    "joint_mae_first_deg": "每一帧预测的 chunk 中，真正会被执行的前 n 步与主臂真值的平均绝对误差。最贴近部署表现。",
    "exec_traj_mae_deg": "把每隔 n 步推理一次的结果拼成连续轨迹，与主臂轨迹比较。",
    "joint_mae_full_deg": "整个 chunk（50 步≈1.7s）的平均误差，反映远期规划能力。",
    "joint_p95_first_deg": "前 n 步误差的 95 分位数：偶发大错的程度，越小越稳。",
    "gripper_mae_first": "夹爪原始输出（滤波前）与主臂夹爪的误差。",
    "gripper_f1": "以 0.5 为阈值把夹爪分成开/合，预测与主臂一致程度（F1）。",
    "grasp_timing_s": "第一次闭合时刻的偏差（秒）。一方闭合另一方未闭合记 2 秒。",
    "boundary_jump_deg": "两个 chunk 衔接处执行轨迹的额外跳变（扣除主臂自然运动），越小动作越平滑。",
}


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    cfg = read_json(run_dir / "train_config.json")
    cfg["dataset_name"] = Path(cfg["dataset"]).name
    status = read_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    summary_path = run_dir / "eval/summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    metric_specs, joint_names, ckpts, best = [], [], [], None
    if summary:
        first = summary["checkpoints"][0]
        metric_specs = [(k, s) for k, s in summary["scoring"]["metrics"].items() if k in first["metrics"]]
        joint_names = list(first["per_joint_mae_first"])
        ckpts = summary["checkpoints"]
        best = first
    html = TEMPLATE.render(
        name=run_dir.name, now=datetime.now().strftime("%Y-%m-%d %H:%M"), cfg=cfg, status=status,
        cfg_items=[(k, v) for k, v in cfg.items() if k != "extra"], val_rows=read_csv(run_dir / "metrics/val.csv"),
        summary=summary, metric_specs=metric_specs, joint_names=joint_names, ckpts=ckpts, best=best, explain=EXPLAIN)
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def package(run_dir: Path) -> Path:
    """打包审阅所需文件（不含权重、npz 原始预测、tensorboard 事件文件）。"""
    run_dir = Path(run_dir)
    report = build(run_dir)
    zpath = run_dir / f"{run_dir.name}_statistics.zip"
    files = [report, run_dir / "train_config.json", run_dir / "status.json"]
    files += sorted((run_dir / "metrics").glob("*.csv")) + sorted((run_dir / "metrics").glob("*.png"))
    files += [run_dir / "eval/summary.json", run_dir / "eval/ranking.csv", run_dir / "checkpoints/topk.json"]
    files += sorted((run_dir / "eval/plots").glob("*.png"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.is_file():
                z.write(f, Path(run_dir.name) / f.relative_to(run_dir))
    return zpath


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="重新生成报告并打包")
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    z = package(args.run)
    print(json.dumps({"zip": str(z), "size_mb": round(z.stat().st_size / 1e6, 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
