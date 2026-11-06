"""Whole-episode checkpoint evaluation with inspectable curves and ranking reports."""
from __future__ import annotations

import csv
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from acttrain.data.core import ACTION_NAMES, read_json, write_json
from acttrain.evaluation.episode_io import load_trace, predict_episode
from acttrain.evaluation.metrics import (WEIGHTS, error_values, forecast_mask, limit_report, merge_limit_reports,
                                ranking_components, ranking_key, ranking_score, report_from_errors, validate_limits)

STYLE = "body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:20px;line-height:1.6}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.notice{background:#fff3cd;padding:12px}a{color:#176a9e}details{margin:14px 0}"


def href(path):
    return escape(quote(str(path), safe="/"), quote=True)


def limits_notice(deployment):
    label = ("用户声明已确认的限制；仅检查已配置项，不构成完整实机部署验证。" if deployment["confirmed"] else
             "实机限制未确认：当前仅为未核实参考阈值，参考超限不参与排名，不能判定可安全部署。")
    missing = [name for name, rule in deployment["full"]["rules"].items() if rule["status"] == "not_configured"]
    return f'<p class="notice">{label}<br>来源：{escape(deployment["source"])}<br>未检查：{escape(", ".join(missing) or "无（仅限列出的检查项）")}</p>'


def error_table(errors):
    sections = []
    for scope, title in (("full", "所有有效未来32步"), ("first4", "有效前四步"), ("first_step", "曲线逐帧第一步")):
        rows = []
        for name, item in errors[scope]["axes"].items():
            unit = "rad" if name != "right_gripper" else "[0,1]"
            rows.append(f'<tr><td>{name} ({unit})</td>' + ''.join(f'<td>{item[k]:.6g}</td>' for k in ("mae", "median_abs", "max_abs", "rmse")) + '</tr>')
        sections.append(f'<h3>{title}，样本数 {errors[scope]["samples"]}</h3><table><tr><th>轴</th><th>平均绝对误差</th><th>中位绝对误差</th><th>最大绝对误差</th><th>RMSE</th></tr>' + ''.join(rows) + '</table>')
    return ''.join(sections)


def safety_table(report):
    fields = {"joint_position": ("joint_min_rad", "joint_max_rad"), "gripper_position": ("gripper_min", "gripper_max"),
              "joint_step": ("joint_step_max_rad",), "joint_velocity": ("joint_velocity_max_rad_s",),
              "joint_acceleration": ("joint_acceleration_max_rad_s2",), "gripper_step": ("gripper_step_max",),
              "gripper_velocity": ("gripper_velocity_max_s",)}
    def fmt(value):
        if value is None: return "—"
        x = np.asarray(value)
        return f"{float(x):.4g}" if x.ndim == 0 else '[' + ', '.join(f'{v:.4g}' for v in x) + ']'
    rows = []
    for name, rule in report["full"]["rules"].items():
        first = report["first4"]["rules"][name]
        full_value = f'{rule["violating_steps"]}/{rule["checked_steps"]}' if rule["status"] == "checked" else "未配置"
        first_value = f'{first["violating_steps"]}/{first["checked_steps"]}' if first["status"] == "checked" else "未配置"
        bounds = ' / '.join(fmt(report["limits"][key]) for key in fields[name])
        observed = fmt(rule["observed_max_abs"])
        rows.append(f'<tr><td>{name}</td><td>{bounds}</td><td>{observed}</td><td>{full_value}</td><td>{first_value}</td></tr>')
    return '<div class="scroll"><table><tr><th>约束（rad、s、归一化夹爪单位）</th><th>阈值（位置为下限/上限）</th><th>预测最大绝对运动值（关节按6轴列出）</th><th>完整预测超限步/检查步</th><th>前四步超限步/检查步</th></tr>' + ''.join(rows) + '</table></div>'


def save_episode(output, result, errors, deployment, metadata):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = result["prediction_chunks"]
    rows = len(result["target"])
    index = np.arange(rows)
    origin = index // 4 * 4
    stitched = chunks[origin, index % 4]
    valid = forecast_mask(chunks, rows)[3]
    np.savez_compressed(output / "predictions.npz", **result, prediction_first4=stitched, valid_horizon_mask=valid)
    fields = ["row", "observation_timestamp_ns", "action_timestamp_ns", "target_time_s"]
    for name in ACTION_NAMES:
        fields += [f"{name}_{kind}" for kind in ("prediction", "target", "error", "prediction_first4", "recorded_follower")]
    if "prediction_filtered" in result:
        fields += ["right_gripper_filtered"]
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in range(rows):
            values = [row, int(result["observation_timestamp_ns"][row]), int(result["action_timestamp_ns"][row]), result["target_time_s"][row]]
            for j in range(7):
                prediction, target = result["prediction"][row, j], result["target"][row, j]
                values.extend((prediction, target, prediction - target, stitched[row, j], result["recorded_follower"][row, j]))
            if "prediction_filtered" in result:
                values.append(result["prediction_filtered"][row,6])
            writer.writerow(values)
    pictures = []
    for j, name in enumerate(ACTION_NAMES):
        factor, unit = (180 / np.pi, "deg") if j < 6 else (1., "normalized opening")
        figure, axis = plt.subplots(figsize=(10, 4.6), layout="constrained")
        time = result["target_time_s"]
        axis.plot(time, result["target"][:, j] * factor, color="#2067a1", label=f"Recorded {metadata['action_source']} target")
        axis.plot(time, result["prediction"][:, j] * factor, "--", color="#df7925", label="ACT first step, queried every frame")
        axis.plot(time, stitched[:, j] * factor, color="#17958b", alpha=.65, linewidth=1, label="ACT first-4 stitched (offline)")
        if metadata["action_source"] == "leader":
            axis.plot(time, result["recorded_follower"][:, j] * factor, color="#777", alpha=.6, linewidth=1, label="Recorded follower at target time")
        if j == 6 and "prediction_filtered" in result:
            axis.plot(time, result["prediction_filtered"][:,6], color="#be51a4", label="Filtered gripper (not scored)")
        limits = deployment["limits"]
        for bound in (("joint_min_rad", "joint_max_rad") if j < 6 else ("gripper_min", "gripper_max")):
            value = limits[bound]
            if value is not None:
                v = float(np.broadcast_to(value, (6,))[j]) if j < 6 else float(value)
                axis.axhline(v * factor, color="#a63a3a", linestyle=":", linewidth=.8,
                             label=(("Configured " if deployment["confirmed"] else "Unverified reference ")
                                    + ("lower bound" if "min" in bound else "upper bound")))
        metric = errors["first_step"]["axes"][name]
        axis.set_title(f"{metadata['checkpoint_name']} | {Path(metadata['episode']).name} | {name}\n"
                       f"First-step abs error: mean={metric['mae'] * factor:.4g}, median={metric['median_abs'] * factor:.4g}, max={metric['max_abs'] * factor:.4g} {unit}", fontsize=10)
        axis.set_xlabel("Target time from first observation (s)")
        axis.set_ylabel(f"Angle ({unit})" if j < 6 else "Gripper opening")
        axis.grid(alpha=.2)
        axis.legend(loc="best", fontsize=8)
        filename = f"joint_{j + 1}.png" if j < 6 else "gripper.png"
        try:
            figure.savefig(output / filename, dpi=150)
        finally:
            plt.close(figure)
        # Successful rendering is required before this episode can enter the ranking.
        if not (output / filename).is_file() or (output / filename).stat().st_size < 100:
            raise RuntimeError(f"Missing or empty comparison plot: {filename}")
        pictures.append(f'<h3>{escape(name)}</h3><img loading="lazy" src="{filename}" alt="{filename}">')
    write_json(output / "metrics.json", {**metadata, "errors": errors, "deployment": deployment,
                                        "saved_joint_unit": "rad", "prediction_clipping": False})
    html = f'<!doctype html><meta charset="utf-8"><title>Episode comparison</title><style>{STYLE}</style>'
    html += f'<h1>{escape(metadata["checkpoint_name"])} / {escape(metadata["episode"])}</h1><a href="../index.html">返回 checkpoint</a>'
    html += limits_notice(deployment) + '<p>每帧输入录制观测。曲线第一步目标晚50ms；绿色线按每四步重新取录制观测拼接，不是闭环机器人轨迹。关节CSV/NPZ单位rad，图片单位deg。CSV包含夹爪、两种预测、目标和follower实录；NPZ额外保存全部32步预测及有效尾段掩码。</p>'
    html += '<p><a href="predictions.csv">逐关节预测CSV</a>（完整32步预测另存于运行目录的predictions.npz）</p>'
    html += ''.join(pictures) + error_table(errors) + safety_table(deployment)
    (output / "index.html").write_text(html, encoding="utf-8")


def evaluate_checkpoint(runtime, dataset, entries, root, std, output, stride, batch_size, limits=None):
    if stride < 1 or batch_size < 1:
        raise ValueError("Positive batch size and legacy stride required")
    limits = validate_limits(limits)
    entries = sorted(entries, key=lambda e: e["dataset_episode_index"])
    if not entries or len({e["dataset_episode_index"] for e in entries}) != len(entries):
        raise ValueError("Validation episodes must have contiguous split-local indices")
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    action_source = manifest.get("action_source", "follower")
    cursor, episode_reports, deployments = 0, [], []
    all_errors = {scope: [] for scope in ("full", "first4", "first_step")}
    all_baselines = {scope: [] for scope in all_errors}
    for entry in entries:
        cursor = entry.get("_dataset_offset", cursor)
        trace = load_trace(root, entry)
        prediction = predict_episode(runtime, dataset, cursor, entry["rows"], execute_steps=1, batch_size=batch_size,
                                     expected_state=trace["state"], expected_episode_index=entry["dataset_episode_index"])
        chunks = prediction["prediction_chunks"]
        errors = error_values(chunks, trace["target"])
        report = report_from_errors(errors, std)
        from sklearn.metrics import f1_score, confusion_matrix
        _, _, target_rows, mask = forecast_mask(chunks, len(trace["target"]))
        true_grip = trace["target"][np.minimum(target_rows, len(trace["target"])-1),6][mask] >= .5
        pred_grip = chunks[:,:,6][mask] >= .5
        report["gripper_classification"] = {"threshold":.5,"open_f1":float(f1_score(true_grip,pred_grip,zero_division=0)),
            "macro_f1":float(f1_score(true_grip,pred_grip,labels=[False,True],average="macro",zero_division=0)),
            "confusion_matrix":confusion_matrix(true_grip,pred_grip,labels=[False,True]).tolist()}
        baseline = np.broadcast_to(trace["state"][:, None, 7:14], chunks.shape)
        baseline_errors = error_values(baseline, trace["target"])
        for scope in all_errors:
            all_errors[scope].append(errors[scope])
            all_baselines[scope].append(baseline_errors[scope])
        deployment = limit_report(chunks, trace["state"], prediction["query_rows"], limits)
        deployments.append(deployment)
        identifier = f"val_{entry['dataset_episode_index']:04d}_{Path(entry['source_relative']).name}"
        folder = output / identifier
        folder.mkdir()
        metadata = {"episode": entry["source_relative"], "checkpoint_name": output.name,
                    "action_source": action_source, "state_source": "follower", "rows": entry["rows"]}
        if getattr(runtime,"gripper_filter",None):
            from acttrain.model.gripper import soft_floor
            filtered = prediction["prediction"].copy()
            filtered[:,6] = soft_floor(filtered[:,6], **runtime.gripper_filter)
            prediction["prediction_filtered"] = filtered
            metadata["gripper_filter"] = runtime.gripper_filter
        save_episode(folder, {**trace, **prediction}, report, deployment, metadata)
        write_json(folder / "status.json", {"status": "completed"})
        components = ranking_components(report, std)
        episode_reports.append({"source_relative": entry["source_relative"], "folder": identifier,
                                "errors": report, "deployment": deployment, "ranking_components": components,
                                "ranking_score": ranking_score(components)})
        cursor += entry["rows"]
        print(f"  Full validation and seven plots complete: {entry['source_relative']} ({entry['rows']} queries)", flush=True)

    errors = report_from_errors({k: np.concatenate(v) for k, v in all_errors.items()}, std)
    baseline = report_from_errors({k: np.concatenate(v) for k, v in all_baselines.items()}, std)
    components = {key: float(np.mean([e["ranking_components"][key] for e in episode_reports])) for key in WEIGHTS}
    deployment = merge_limit_reports(deployments)
    full = errors["full"]
    result = {"errors": errors, "episodes": episode_reports, "deployment": deployment,
              "ranking_score": ranking_score(components), "ranking_components": components,
              "ranking_metric": "episode_macro_normalized_error", "ranking_weights":dict(WEIGHTS),
              "policy": {"windows": sum(e["rows"] for e in entries), "joint_mae_rad": full["joint_mae_rad"], "joint_mae_deg": full["joint_mae_deg"],
                         "gripper_mae": full["gripper_mae"], "full_chunk_normalized_mae": full["normalized_mae"],
                         "executed_normalized_mae": errors["first4"]["normalized_mae"]},
              "hold_baseline": baseline, "baseline_ranking_score": baseline["first4"]["normalized_mae"],
              "query_stride": 1, "tail_policy": "error uses real targets only; limit checks cover every raw forecast"}
    links = ''.join(f'<li><a href="{href(e["folder"] + "/index.html")}">{escape(e["source_relative"])}</a> — score {e["ranking_score"]:.5g}</li>' for e in episode_reports)
    html = f'<!doctype html><meta charset="utf-8"><title>Checkpoint validation</title><style>{STYLE}</style><h1>{escape(output.name)}</h1><a href="../index.html">返回综合排名</a>'
    html += limits_notice(deployment) + '<h2>本次选中的验证 episode 曲线</h2><ul>' + links + '</ul>' + error_table(errors) + safety_table(deployment)
    (output / "index.html").write_text(html, encoding="utf-8")
    return result


def write_report(output, results, manifest_hash):
    ranked = sorted(results, key=ranking_key)
    if not ranked or any(item["deployment"]["limits"] != ranked[0]["deployment"]["limits"] for item in ranked):
        raise ValueError("Ranking requires completed checkpoints evaluated with the same limit configuration")
    for item in ranked:
        report = output / item["report"]
        if not report.is_file():
            raise RuntimeError(f"Checkpoint plot report is missing: {report}")
    receipt = {"status": "passed", "dataset_manifest_sha256": manifest_hash, "best_checkpoint": ranked[0]["checkpoint"],
               "ranking_weights": WEIGHTS, "joint_gripper_weights": [.8, .2], "checkpoints": ranked,
               "ranking": "confirmed configured-limit first4 violations first; then episode-equal normalized error score",
               "evaluation": "offline recorded observations, not closed-loop success or deployment safety certification"}
    rows, details = [], []
    for number, item in enumerate(ranked, 1):
        errors, deployment = item["errors"], item["deployment"]
        full, first = errors["full"], errors["first4"]
        med = np.mean([full["axes"][n]["median_abs_deg"] for n in ACTION_NAMES[:6]])
        maximum = max(full["axes"][n]["max_abs_deg"] for n in ACTION_NAMES[:6])
        grip = full["axes"]["right_gripper"]
        safety = deployment["first4"]
        safety_label = f'{safety["violating_steps"]}/{safety["checked_steps"]}'
        if not deployment["confirmed"]: safety_label += "（参考，不参与排名）"
        name = escape(item["checkpoint"])
        rows.append(f'<tr><td>{number}</td><td><a href="{href(item["report"])}">{name}</a></td>'
                    f'<td>{item["ranking_score"]:.5g}</td><td>{first["joint_mae_deg"]:.4g}</td><td>{full["joint_mae_deg"]:.4g}</td>'
                    f'<td>{med:.4g}</td><td>{maximum:.4g}</td><td>{grip["mae"]:.4g}</td><td>{grip["median_abs"]:.4g}</td><td>{grip["max_abs"]:.4g}</td><td>{safety_label}</td></tr>')
        details.append(f'<details><summary>#{number} {name} — 各关节详细指标与限制检查</summary>' + error_table(errors) + safety_table(deployment) + '</details>')
    html = f'<!doctype html><meta charset="utf-8"><title>ACT 综合验证排名</title><style>{STYLE}</style><h1>ACT 离线验证综合排名</h1>'
    html += limits_notice(ranked[0]["deployment"])
    html += '<p>每个 checkpoint 对本次选中的验证 episode 的每帧观测推理，生成六关节及夹爪曲线。点击 checkpoint 查看各 episode 的图片和CSV/NPZ。输入仅为观测，预测不裁剪。v2相对关节已恢复为绝对位置，与录制leader标签比较；灰线为对应时刻follower实录。</p>'
    html += f'<p>综合分越低越好：{WEIGHTS["first4"]:g}×前4步MAE + {WEIGHTS["mean"]:g}×完整预测MAE + {WEIGHTS["median"]:g}×中位绝对误差 + {WEIGHTS["maximum"]:g}×最大绝对误差。各轴除以本数据集绝对action训练std；六关节均值占80%，夹爪占20%，再按episode等权平均。以下表格误差列为全部有效预测合并统计，逐轴详表可展开。尾部缺少真值的预测不计分。曲线包含逐帧第一步和按前4步拼接的预测。</p>'
    html += '<p>仅当用户声明确认了限制时，先按前4步是否超限、超限比例排序，再按误差分排序；未确认参考阈值不影响排名。位置、步长、速度、加速度按20Hz和每次查询的真实follower锚点检查，不跨episode。误差排除无真值尾段；数值限制检查无需真值，覆盖全部32步原始预测。离线误差和阈值检查不代表真实任务成功率。</p>'
    html += '<div class="scroll"><table><tr><th>排名</th><th>Checkpoint / 曲线入口</th><th>综合分</th><th>前4步关节MAE°</th><th>关节MAE°</th><th>各轴中位绝对误差均值°</th><th>最坏关节绝对误差°</th><th>夹爪MAE</th><th>夹爪中位绝对误差</th><th>夹爪最大绝对误差</th><th>前4步超限/检查步</th></tr>'
    html += ''.join(rows) + '</table></div>' + ''.join(details)
    (output / "index.html").write_text(html, encoding="utf-8")
    write_json(output / "summary.json", receipt)
    write_json(output / "best_checkpoint.json", {"pretrained_model": ranked[0]["checkpoint"],
                                                "score": ranked[0]["ranking_score"], "ranking": receipt["ranking"],
                                                "configured_limits_confirmed": ranked[0]["deployment"]["confirmed"],
                                                "first4_limit_violations": ranked[0]["deployment"]["first4"]["violating_steps"]})
    return receipt
