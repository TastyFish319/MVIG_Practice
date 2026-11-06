"""Compare ACT checkpoints on held-out demonstrations, without hardware or training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from acttrain.data.core import ACTION_NAMES, file_hash, legal_window_starts, read_json, write_json
from acttrain.model.inference import OBSERVATION_KEYS, checkpoint_directory
from acttrain.model.inference import ACTInference
from acttrain.model.velocity_inference import VelocityInference
from acttrain.data.registry import resolve_dataset
import random
from acttrain.evaluation.parallel import atomic_json, evaluation_devices, run_parallel_evaluation


def evaluation_starts(rows, horizon=32, stride=4):
    valid = legal_window_starts(rows, horizon)
    if not len(valid) or stride < 1:
        raise ValueError("Need a positive stride and at least one full action window")
    return sorted(set(range(0, len(valid), stride)) | {valid[-1]})


def action_metrics(prediction, target, train_std, execute_steps=4):
    prediction, target, train_std = (np.asarray(value, np.float64) for value in (prediction, target, train_std))
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 7
            or not len(prediction) or not 1 <= execute_steps <= prediction.shape[1]
            or train_std.shape != (7,) or np.any(train_std <= 0)
            or not all(np.isfinite(value).all() for value in (prediction, target, train_std))):
        raise ValueError("Invalid predictions, targets, training std or execution horizon")
    error = np.abs(prediction - target)
    return {"windows": len(prediction), "executed_steps": execute_steps,
            "executed_normalized_mae": float((error[:, :execute_steps] / train_std).mean()),
            "full_chunk_normalized_mae": float((error / train_std).mean()),
            "joint_mae_rad": float(error[:, :, :6].mean()),
            "joint_mae_deg": float(np.rad2deg(error[:, :, :6]).mean()),
            "gripper_mae": float(error[:, :, 6].mean()),
            "gripper_threshold_agreement": float(((prediction[:, :, 6] >= .5) == (target[:, :, 6] >= .5)).mean()),
            "per_axis_mae": dict(zip(ACTION_NAMES, error.mean(axis=(0, 1)).tolist())),
            "per_horizon_joint_mae_rad": error[:, :, :6].mean(axis=(0, 2)).tolist(),
            "gripper_out_of_range_fraction": float(((prediction[:, :, 6] < 0) | (prediction[:, :, 6] > 1)).mean())}


def discover_checkpoints(root):
    root = Path(root).expanduser().resolve()
    directory = root / "checkpoints" if (root / "checkpoints").is_dir() else root
    children = sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    candidates = children if children else [root]
    return sorted(set(checkpoint_directory(path) for path in candidates))


def checkpoint_reference(checkpoint, output):
    import os
    try:
        return Path(os.path.relpath(checkpoint, output)).as_posix()
    except ValueError:
        return Path(checkpoint).resolve().as_posix()


def validate_checkpoint_source(runtime, manifest, manifest_hash, require_v2=False):
    contract = getattr(runtime, "action_contract", None)
    if contract is None:
        if require_v2:
            raise ValueError("V2 evaluation requires a checkpoint with act_action_contract.json")
        return
    for key, default in (("state_source", "follower"), ("action_source", "follower")):
        if contract.get(key, default) != manifest.get(key, default):
            raise ValueError(f"Checkpoint and dataset {key} differ")
    if contract.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint was trained on a different dataset manifest")


def evaluate_checkpoint(runtime, dataset, entries, root, std, output, stride, batch_size, limits=None):
    from acttrain.evaluation.episodes import evaluate_checkpoint as full_episode_evaluation
    return full_episode_evaluation(runtime, dataset, entries, root, std, output, stride, batch_size, limits)


def write_report(output, results, manifest_hash):
    from acttrain.evaluation.episodes import write_report as full_episode_report
    return full_episode_report(output, results, manifest_hash)


def evaluation_metadata(root, max_episodes=8, seed=42):
    root = resolve_dataset(root)
    if read_json(root / "validation.json")["status"] != "passed":
        raise ValueError("Validate the dataset first")
    manifest = read_json(root / "manifest.json")
    if (manifest["fps"], manifest["action_horizon"], manifest["action_offset_frames"]) != (20, 32, 1):
        raise ValueError("Dataset differs from the right-arm ACT temporal contract")
    entries = [entry for entry in manifest["episodes"] if entry["split"] == "val"]
    if not entries or any(manifest["split"][entry["source_relative"]] != "val" for entry in entries):
        raise ValueError("Missing or inconsistent held-out validation episodes")
    cursor = 0
    for entry in entries:
        entry["_dataset_offset"] = cursor
        cursor += entry["rows"]
    if max_episodes and len(entries) > max_episodes:
        entries = sorted(random.Random(seed).sample(entries, max_episodes), key=lambda e:e["dataset_episode_index"])
    stats = read_json(root / "normalization_train.json")
    return dict(root=root, manifest=manifest, entries=entries, stats=stats,
                std=np.asarray(stats["action"]["std"]), manifest_hash=file_hash(root / "manifest.json"))


def open_dataset(root, manifest):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(f"yam_local/{manifest['task']}_val", root=root / "val", video_backend="pyav")


def configure_runtime(device, threads):
    import torch
    torch.set_num_threads(threads)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))


def evaluate_task(task, context, config):
    checkpoint = Path(task["checkpoint"])
    output = Path(config["output"])
    print(f"Checkpoint {task['index'] + 1}: {checkpoint} on {config['device']}", flush=True)
    kind = VelocityInference if (checkpoint / "act_observation_contract.json").is_file() else ACTInference
    runtime = kind(checkpoint, config["device"], expected_stats=context["stats"])
    runtime.gripper_filter = config.get("gripper_filter")
    try:
        validate_checkpoint_source(runtime, context["manifest"], context["manifest_hash"], True)
        folder = output / f"checkpoint_{task['index']:03d}_{checkpoint.parent.name}"
        folder.mkdir()
        result = evaluate_checkpoint(runtime, context["dataset"], context["entries"], context["root"],
                                     context["std"], folder, config["stride"], config["batch_size"],
                                     limits=config["limits"])
        result.update(report=f"{folder.name}/index.html", checkpoint=checkpoint_reference(checkpoint, output),
                      model_sha256=file_hash(checkpoint / "model.safetensors"))
        write_json(folder / "metrics.json", result)
        write_json(folder / "status.json", {"status": "completed"})
        return result
    finally:
        del runtime
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan_path.parent != Path(plan["output"]).resolve() / ".workers":
        raise ValueError("Worker plan must be inside its output .workers directory")
    receipt = dict(status="running", results=[], device=plan["device"], pid=os.getpid(),
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    result_path = plan_path.with_suffix(".result.json")
    begin = time.monotonic()

    def publish(status, **extra):
        receipt.update(status=status, elapsed_seconds=time.monotonic() - begin, **extra)
        atomic_json(result_path, receipt)

    try:
        publish("running")
        configure_runtime(plan["device"], plan["cpu_threads"])
        context = evaluation_metadata(Path(plan["root"]), plan["max_episodes"], plan["seed"])
        configure_scoring(plan.get("scoring"))
        context["dataset"] = open_dataset(context["root"], context["manifest"])
        for task in plan["tasks"]:
            result = evaluate_task(task, context, plan)
            receipt["results"].append(dict(index=task["index"], result=result))
            publish("running")
        publish("passed")
    except BaseException as error:
        publish("failed", error=str(error))
        raise


def configure_scoring(value):
    from acttrain.evaluation.metrics import WEIGHTS
    default = {"first4":.40,"mean":.25,"median":.20,"maximum":.15}
    if value is None:
        weights = default
    else:
        if not isinstance(value,dict) or set(value) != {"weights"} or not isinstance(value["weights"],dict) or set(value["weights"]) != set(default):
            raise ValueError("评分配置需且仅需weights及四个既定分项")
        weights = value["weights"]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):
            raise ValueError("评分权重必须有限、非负且总和为1")
    WEIGHTS.clear()
    WEIGHTS.update(weights)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Internal subprocess mode has its own required arguments, never a training entrypoint.
    if argv[:1] == ["--worker-plan"]:
        if len(argv) != 2:
            raise ValueError("Worker mode requires exactly one plan path")
        run_worker(Path(argv[1]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True, help="Run, checkpoints, step or pretrained_model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", choices=("auto", "1", "2", "4"), default="auto",
                        help="Independent checkpoint workers; auto uses up to four visible GPUs")
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU threads per evaluation worker")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1, help="Legacy argument; full validation always queries every frame")
    parser.add_argument("--limits", type=Path, help="JSON motion limits with explicit source and confirmed flag")
    parser.add_argument("--max-episodes", type=int, default=8, help="0=全部val；否则固定seed抽至多N条")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scoring", type=Path)
    parser.add_argument("--gripper-floor", type=float)
    parser.add_argument("--gripper-threshold", type=float)
    args = parser.parse_args(argv)
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists() or output.is_symlink() or min(args.batch_size, args.stride, args.cpu_threads) < 1 or args.max_episodes < 0:
        parser.error("Choose a new output directory and positive batch size/stride/thread count")
    begin = time.monotonic()
    context = evaluation_metadata(root, args.max_episodes, args.seed)
    root = context["root"]
    scoring = read_json(args.scoring) if args.scoring else None
    configure_scoring(scoring)
    gripper_filter = None
    if args.gripper_floor is not None or args.gripper_threshold is not None:
        if args.gripper_floor is None or args.gripper_threshold is None:
            parser.error("gripper-floor与gripper-threshold必须一起设置")
        from acttrain.model.gripper import soft_floor
        soft_floor([0.0], args.gripper_floor, args.gripper_threshold)
        gripper_filter = dict(floor=args.gripper_floor, threshold=args.gripper_threshold)
    from acttrain.evaluation.metrics import validate_limits
    limits = validate_limits(read_json(args.limits.expanduser().resolve()) if args.limits else None)
    if args.stride != 1:
        print("Legacy --stride is ignored: full validation queries every observation frame.", flush=True)
    checkpoints = discover_checkpoints(args.checkpoints)
    import torch
    devices = evaluation_devices(args.device, args.gpus if args.gpus == "auto" else int(args.gpus),
                                 len(checkpoints), torch.cuda.device_count() if args.device.startswith("cuda") else 0)
    tasks = [dict(index=i, checkpoint=str(checkpoint)) for i, checkpoint in enumerate(checkpoints)]
    config = dict(root=str(root), output=str(output), batch_size=args.batch_size, stride=args.stride,
                  cpu_threads=args.cpu_threads, limits=limits, max_episodes=args.max_episodes, seed=args.seed, scoring=scoring, gripper_filter=gripper_filter)
    output.mkdir(parents=True)
    write_json(output / "selection.json", {"seed":args.seed,"max_episodes":args.max_episodes,
        "episodes":[e["source_relative"] for e in context["entries"]],"dataset_manifest_sha256":context["manifest_hash"]})
    from acttrain.evaluation.metrics import WEIGHTS
    write_json(output / "scoring.json", {"version":"act_custom_v1" if scoring else "act_compatible_v1","weights":WEIGHTS,
        "joint_gripper_weights":[.8,.2], "std":context["std"].tolist(), "scale":"absolute_action_train_std", "raw_predictions":True})
    results = []
    atomic_json(output / "progress.json", {"status": "running", "expected_checkpoints": len(checkpoints), "completed": 0})
    try:
        if len(devices) > 1:
            results = run_parallel_evaluation(config, tasks, devices)
        else:
            configure_runtime(devices[0], args.cpu_threads)
            context["dataset"] = open_dataset(root, context["manifest"])
            for task in tasks:
                results.append(evaluate_task(task, context, {**config, "device": devices[0]}))
                atomic_json(output / "progress.json", {"status": "running", "expected_checkpoints": len(tasks),
                                                       "completed": len(results)})
        write_report(output, results, context["manifest_hash"])
    except BaseException as error:
        completed = read_json(output / "progress.json").get("completed", len(results))
        atomic_json(output / "progress.json", {"status": "failed", "expected_checkpoints": len(tasks),
                                               "completed": completed, "error": str(error)})
        raise
    execution = dict(status="passed", requested_gpus=args.gpus, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=devices, workers=len(devices), batch_size_per_gpu=args.batch_size,
                     cpu_threads_per_worker=args.cpu_threads, elapsed_seconds=time.monotonic() - begin)
    write_json(output / "evaluation_execution.json", execution)
    atomic_json(output / "progress.json", {**execution, "expected_checkpoints": len(tasks), "completed": len(results)})
    print(f"Completed {len(results)} checkpoints. Open {output / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    main()
