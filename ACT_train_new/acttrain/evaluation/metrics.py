"""Error statistics and explicitly sourced motion-limit checks for offline ACT."""
from __future__ import annotations

import numpy as np

from acttrain.data.core import ACTION_NAMES

WEIGHTS = {"first4": .40, "mean": .25, "median": .20, "maximum": .15}
LIMIT_KEYS = {"confirmed", "source", "joint_min_rad", "joint_max_rad", "joint_step_max_rad",
              "joint_velocity_max_rad_s", "joint_acceleration_max_rad_s2", "gripper_min", "gripper_max",
              "gripper_step_max", "gripper_velocity_max_s"}


def validate_limits(value):
    if value is None:
        value = {"confirmed": False, "source": "Unverified ACT client defaults; not measured hardware limits",
                 "joint_step_max_rad": .15, "gripper_min": -.05, "gripper_max": 1.05}
    if not isinstance(value, dict) or set(value) - LIMIT_KEYS:
        raise ValueError("Unknown deployment-limit fields")
    if not isinstance(value.get("confirmed", False), bool):
        raise ValueError("confirmed must be a boolean")
    if not isinstance(value.get("source", "Unspecified"), str):
        raise ValueError("Limit source must be a descriptive string")
    result = {"confirmed": value.get("confirmed", False), "source": value.get("source", "Unspecified")}
    configured = 0
    for key in LIMIT_KEYS - {"confirmed", "source"}:
        item = value.get(key)
        if item is None:
            result[key] = None
            continue
        x = np.asarray(item, np.float64)
        if isinstance(item, bool) or not np.isfinite(x).all() or (x.shape not in ((), (6,)) if key.startswith("joint_") else x.shape != ()):
            raise ValueError(f"Invalid finite scalar / six-joint bounds: {key}")
        if ("step_max" in key or "velocity_max" in key or "acceleration_max" in key) and np.any(x <= 0):
            raise ValueError(f"Limit must be positive: {key}")
        result[key] = x.tolist()
        configured += 1
    for low, high in (("joint_min_rad", "joint_max_rad"), ("gripper_min", "gripper_max")):
        if result[low] is not None and result[high] is not None and np.any(np.asarray(result[low]) > np.asarray(result[high])):
            raise ValueError("Minimum deployment bound exceeds maximum")
    if result["confirmed"] and (not configured or result["source"] in ("", "Unspecified")):
        raise ValueError("Confirmed limits need explicit numeric limits and a source description")
    return result


def forecast_mask(chunks, rows, queries=None):
    chunks = np.asarray(chunks, np.float64)
    if chunks.ndim != 3 or chunks.shape[-1] != 7 or not len(chunks) or not 1 <= chunks.shape[1] <= 32 or not np.isfinite(chunks).all():
        raise ValueError("Expected finite nonempty [queries,H<=32,7] predictions")
    queries = np.arange(len(chunks), dtype=np.int64) if queries is None else np.asarray(queries)
    if queries.shape != (len(chunks),) or not np.issubdtype(queries.dtype, np.integer) or np.any(queries < 0) or np.any(queries >= rows):
        raise ValueError("Invalid within-episode query rows")
    indices = queries[:, None] + np.arange(chunks.shape[1])
    return chunks, queries, indices, indices < rows


def error_values(chunks, target, queries=None):
    target = np.asarray(target, np.float64)
    if target.ndim != 2 or target.shape[1] != 7 or not len(target) or not np.isfinite(target).all():
        raise ValueError("Expected finite episode targets [rows,7]")
    chunks, _, indices, valid = forecast_mask(chunks, len(target), queries)
    errors = np.abs(chunks - target[np.minimum(indices, len(target) - 1)])
    return {"full": errors[valid], "first4": errors[:, :4][valid[:, :4]],
            "first_step": errors[:, :1][valid[:, :1]]}


def report_from_errors(values, std):
    std = np.asarray(std, np.float64)
    if std.shape != (7,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Need seven finite positive training standard deviations")
    report = {}
    for scope, error in values.items():
        error = np.asarray(error, np.float64)
        if error.ndim != 2 or error.shape[1] != 7 or not len(error) or not np.isfinite(error).all() or np.any(error < 0):
            raise ValueError("Invalid masked error samples")
        mean, median, maximum, rmse = error.mean(0), np.median(error, axis=0), error.max(0), np.sqrt(np.square(error).mean(0))
        axes = {}
        for j, name in enumerate(ACTION_NAMES):
            axes[name] = dict(mae=float(mean[j]), median_abs=float(median[j]), max_abs=float(maximum[j]), rmse=float(rmse[j]), p95_abs=float(np.quantile(error[:,j], .95)))
            if j < 6:
                axes[name].update({f"{key}_deg": float(np.rad2deg(v)) for key, v in list(axes[name].items())})
        report[scope] = {"samples": len(error), "axes": axes,
                         "normalized_mae": float((error / std).mean()),
                         "joint_mae_rad": float(mean[:6].mean()), "joint_mae_deg": float(np.rad2deg(mean[:6]).mean()),
                         "gripper_mae": float(mean[6])}
    return report


def error_report(chunks, target, std, queries=None):
    return report_from_errors(error_values(chunks, target, queries), std)


def ranking_components(report, std):
    std = np.asarray(std, np.float64)
    def component(scope, stat):
        axes = np.array([report[scope]["axes"][name][stat] for name in ACTION_NAMES]) / std
        return float(.8 * axes[:6].mean() + .2 * axes[6])
    return {"first4": component("first4", "mae"), "mean": component("full", "mae"),
            "median": component("full", "median_abs"), "maximum": component("full", "max_abs")}


def ranking_score(components):
    return float(sum(WEIGHTS[key] * components[key] for key in WEIGHTS))


def ranking_key(item):
    deployment = item.get("deployment", {})
    scope = deployment.get("first4", {})
    violations = scope.get("violating_steps", 0)
    fraction = violations / max(1, scope.get("checked_steps", 0))
    return ((int(violations > 0), fraction, item["ranking_score"]) if deployment.get("confirmed")
            else (0, 0., item["ranking_score"]))


def limit_report(chunks, state, queries, limits, fps=20):
    limits = validate_limits(limits)
    state = np.asarray(state, np.float64)
    if state.ndim != 2 or state.shape[1] != 14 or not np.isfinite(state).all() or fps != 20:
        raise ValueError("Expected finite state14 and the 20Hz deployment time contract")
    chunks, queries, _, _ = forecast_mask(chunks, len(state), queries)
    # Numerical command limits need no future ground truth. Check the entire raw chunk.
    valid = np.ones(chunks.shape[:2], dtype=bool)
    horizon = chunks.shape[1]
    joints, grip = chunks[:, :, :6], chunks[:, :, 6]
    delta = np.diff(np.concatenate((state[queries, None, 7:13], joints), axis=1), axis=1)
    grip_delta = np.diff(np.column_stack((state[queries, 13], grip)), axis=1)
    velocity = delta * fps
    prior_velocity = (state[queries, 7:13] - state[np.maximum(queries - 1, 0), 7:13]) * fps
    acceleration = np.diff(np.concatenate((prior_velocity[:, None], velocity), axis=1), axis=1) * fps
    acceleration_valid = valid.copy()
    acceleration_valid[queries == 0, 0] = False  # No measured previous velocity at the episode start.
    rules = {}

    def position(name, values, lo, hi):
        enabled = limits[lo] is not None or limits[hi] is not None
        bad = np.zeros(values.shape, bool)
        if limits[lo] is not None: bad |= values < np.asarray(limits[lo])
        if limits[hi] is not None: bad |= values > np.asarray(limits[hi])
        rules[name] = (enabled, bad.any(-1) if bad.ndim == 3 else bad, valid)

    def motion(name, values, key, mask=valid):
        enabled = limits[key] is not None
        bad = np.abs(values) > np.asarray(limits[key]) if enabled else np.zeros(values.shape, bool)
        rules[name] = (enabled, bad.any(-1) if bad.ndim == 3 else bad, mask)

    position("joint_position", joints, "joint_min_rad", "joint_max_rad")
    position("gripper_position", grip, "gripper_min", "gripper_max")
    motion("joint_step", delta, "joint_step_max_rad")
    motion("joint_velocity", velocity, "joint_velocity_max_rad_s")
    motion("joint_acceleration", acceleration, "joint_acceleration_max_rad_s2", acceleration_valid)
    motion("gripper_step", grip_delta, "gripper_step_max")
    motion("gripper_velocity", grip_delta * fps, "gripper_velocity_max_s")
    observed = {"joint_position": joints, "gripper_position": grip, "joint_step": delta,
                "joint_velocity": velocity, "joint_acceleration": acceleration,
                "gripper_step": grip_delta, "gripper_velocity": grip_delta * fps}
    result = {"confirmed": limits["confirmed"], "limits": limits, "source": limits["source"],
              "coverage": "all raw predictions, including forecasts beyond the recorded target tail",
              "assessment": "configured_limits_only" if limits["confirmed"] else "unverified_reference_only"}
    for scope, length in (("full", horizon), ("first4", min(4, horizon))):
        combined = np.zeros((len(chunks), length), bool)
        checked = np.zeros_like(combined)
        detail = {}
        for name, (enabled, bad, mask) in rules.items():
            mask = mask[:, :length]
            if enabled:
                combined |= bad[:, :length] & mask
                checked |= mask
            detail[name] = {"status": "checked" if enabled else "not_configured",
                            "checked_steps": int(mask.sum()) if enabled else 0,
                            "violating_steps": int((bad[:, :length] & mask).sum()) if enabled else 0}
            values = observed[name][:, :length][mask]
            detail[name].update(observed_min=values.min(axis=0).tolist() if len(values) else None,
                                observed_max=values.max(axis=0).tolist() if len(values) else None,
                                observed_max_abs=np.abs(values).max(axis=0).tolist() if len(values) else None)
        count, bad_count = int(checked.sum()), int(combined.sum())
        result[scope] = {"checked_steps": count, "violating_steps": bad_count,
                         "violation_fraction": bad_count / count if count else None, "rules": detail}
    return result


def merge_limit_reports(reports):
    first = reports[0]
    result = {key: first[key] for key in ("confirmed", "limits", "source", "assessment", "coverage")}
    for scope in ("full", "first4"):
        checked = sum(r[scope]["checked_steps"] for r in reports)
        bad = sum(r[scope]["violating_steps"] for r in reports)
        detail = {}
        for name, rule in first[scope]["rules"].items():
            detail[name] = {"status": rule["status"],
                            "checked_steps": sum(r[scope]["rules"][name]["checked_steps"] for r in reports),
                            "violating_steps": sum(r[scope]["rules"][name]["violating_steps"] for r in reports)}
            for key in ("observed_min", "observed_max", "observed_max_abs"):
                arrays = [r[scope]["rules"][name][key] for r in reports if r[scope]["rules"][name][key] is not None]
                detail[name][key] = (np.min(arrays, axis=0).tolist() if key == "observed_min" else np.max(arrays, axis=0).tolist()) if arrays else None
        result[scope] = {"checked_steps": checked, "violating_steps": bad,
                         "violation_fraction": bad / checked if checked else None, "rules": detail}
    return result
