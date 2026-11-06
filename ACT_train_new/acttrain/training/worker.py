"""Relative-joint ACT training on a Linux CUDA server, for 1/2/4 GPUs."""
import argparse
import math
from datetime import timedelta
import os
from pathlib import Path
import sys
from acttrain.data.core import read_json
from acttrain.model.relative import launch_coordinates, relative_statistics
from acttrain.model.velocity import velocity_training, velocity_state_statistics, prepare_batch, decode_actions
from acttrain.training.distributed import capture_distributed, main_process_action
from acttrain.training.config import build_lerobot_config as build_config


def validate_training_manifest(manifest):
    if (manifest.get("action_semantics") != "future_leader_right_absolute" or
            manifest.get("state_source") != "follower" or manifest.get("action_source") != "leader" or
            (manifest.get("fps"), manifest.get("action_horizon"), manifest.get("action_offset_frames")) != (20, 32, 1)):
        raise ValueError("ACT v2 requires follower states and leader actions. Reconvert with blue_block_v2_leader.json; old follower-action datasets cannot be used.")


def apply_policy_overrides(cfg, lr=None, dropout=None, backbone_lr=None):
    if lr is not None and (not math.isfinite(lr) or lr <= 0):
        raise ValueError("Learning rate must be finite and positive")
    if dropout is not None and (not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("Dropout must be finite and in [0,1)")
    if backbone_lr is not None and (not math.isfinite(backbone_lr) or backbone_lr <= 0):
        raise ValueError("Backbone learning rate must be finite and positive")
    if lr is not None:
        cfg.policy.optimizer_lr = lr
    if dropout is not None:
        cfg.policy.dropout = dropout
    if backbone_lr is not None:
        cfg.policy.optimizer_lr_backbone = backbone_lr


def build_scheduler_config(args):
    from acttrain.training.schedule import GroupStepLRSchedulerConfig, StepLRSchedulerConfig
    StepLRSchedulerConfig(step_size=args.steplr_step_size, gamma=args.steplr_gamma)
    values = {}
    for role in ("policy", "backbone"):
        enabled = getattr(args, role + "_steplr")
        interval = getattr(args, role + "_steplr_step_size")
        gamma = getattr(args, role + "_steplr_gamma")
        values.update({role + "_enabled": args.steplr if enabled is None else enabled,
                       role + "_step_size": args.steplr_step_size if interval is None else interval,
                       role + "_gamma": args.steplr_gamma if gamma is None else gamma})
    config = GroupStepLRSchedulerConfig(**values)
    return config if config.policy_enabled or config.backbone_enabled else None


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--save-freq", type=int, default=1000,
                        help="Optimizer steps between checkpoint saves and paired train/val prior probes; also run at the final step")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--val-stride", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "no"), default="bf16")
    parser.add_argument("--lr", "--policy-lr", dest="lr", type=float, help="Policy initial learning rate")
    parser.add_argument("--backbone-lr", type=float, help="Visual backbone initial learning rate")
    parser.add_argument("--steplr", action="store_true", help="Decay policy and backbone LR by optimizer steps")
    parser.add_argument("--steplr-step-size", type=int, default=2000,
                        help="Optimizer updates between StepLR decays (not epochs or per-rank batches)")
    parser.add_argument("--steplr-gamma", type=float, default=0.5, help="StepLR decay factor in (0,1)")
    for role in ("policy", "backbone"):
        parser.add_argument(f"--{role}-steplr", action=argparse.BooleanOptionalAction, default=None,
                            help=f"Enable/disable {role} StepLR independently; overrides legacy --steplr")
        parser.add_argument(f"--{role}-steplr-step-size", type=int, help=f"{role} decay interval in optimizer steps")
        parser.add_argument(f"--{role}-steplr-gamma", type=float, help=f"{role} decay multiplier in (0,1)")
    parser.add_argument("--dropout", type=float, help="Override ACT dropout for this run")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--use-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not args.execute or sys.platform != "linux":
        parser.error("Training requires --execute on a Linux CUDA server")
    _, local_rank = launch_coordinates(world_size=args.gpus)
    if min(args.steps, args.batch_size_per_gpu, args.save_freq, args.log_freq, args.val_stride) < 1 or args.workers_per_gpu < 0:
        parser.error("Invalid training parameters")
    try:
        scheduler = build_scheduler_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        raise FileExistsError("Choose a new experiment directory")
    manifest = read_json(args.dataset / "manifest.json")
    validate_training_manifest(manifest)
    cfg = build_config(args.dataset, args.output, args.steps, args.batch_size_per_gpu, args.workers_per_gpu, seed=args.seed)
    apply_policy_overrides(cfg, args.lr, args.dropout, backbone_lr=args.backbone_lr)
    cfg.job_name = "acttrain_velocity" if args.use_velocity else "acttrain_position"
    cfg.save_freq, cfg.log_freq = args.save_freq, args.log_freq
    cache = (args.cache_root or args.output.resolve().parent / ".cache").resolve()
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Need {args.gpus} visible CUDA GPUs")
    torch.cuda.set_device(local_rank)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; use --precision no")
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from acttrain.training.schedule import step_lr_preset
    accelerator = Accelerator(
        mixed_precision=args.precision, gradient_accumulation_steps=1, step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(split_batches=False, even_batches=True),
        rng_types=["generator"], kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                                                   InitProcessGroupKwargs(timeout=timedelta(minutes=30))])
    if accelerator.num_processes != args.gpus:
        raise RuntimeError("Accelerator process count differs from --gpus")
    cfg.policy.device = str(accelerator.device)

    def warm_backbone():
        from torchvision.models import ResNet18_Weights
        ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)

    main_process_action(accelerator, warm_backbone)
    stats = relative_statistics(args.dataset, manifest["episodes"])
    state_stats = (velocity_state_statistics(args.dataset, manifest["episodes"],
                   read_json(args.dataset / "normalization_train.json")["observation.state"])
                   if args.use_velocity else read_json(args.dataset / "normalization_train.json")["observation.state"])
    metadata = {"version": 3 if args.use_velocity else 2, "world_size": args.gpus, "precision": args.precision,
                "per_gpu_batch": args.batch_size_per_gpu, "global_batch": args.gpus * args.batch_size_per_gpu,
                "optimizer_steps": args.steps, "save_freq": cfg.save_freq, "prior_probe_stride": args.val_stride,
                "seed": cfg.seed, "resume_supported": False,
                "action": "future_joint_minus_current_observation_joint; absolute_gripper",
                "state_source": "follower", "action_source": "leader",
                "optimizer_lr": cfg.policy.optimizer_lr, "optimizer_lr_backbone": cfg.policy.optimizer_lr_backbone,
                "scheduler": scheduler.metadata() if scheduler else None,
                "dropout": cfg.policy.dropout,
                "observation_contract": "state14_plus_right_velocity7_lag2" if args.use_velocity else "state14", "state_stats": state_stats,
                "action_stats": stats, "workers_per_gpu": args.workers_per_gpu}
    from lerobot.scripts import lerobot_train as trainer
    from acttrain.training.validation import prior_validation

    def validate(**kwargs):
        with accelerator.autocast():
            return prior_validation(args.dataset, kwargs["policy"], kwargs["preprocessor"], kwargs["postprocessor"],
                                    stride=args.val_stride, batch_size=args.batch_size_per_gpu,
                                    action_decoder=decode_actions, action_std=stats["std"], state_transform=prepare_batch if args.use_velocity else None)

    # Capture calls the relative save hook, which writes semantics before validation.
    from acttrain.model.relative import relative_training
    adapter = velocity_training(trainer, args.dataset, stats, state_stats) if args.use_velocity else relative_training(trainer, args.dataset, stats)
    with step_lr_preset(cfg.policy, scheduler), adapter:
        with capture_distributed(trainer, accelerator, cfg.output_dir, args.steps,
                                 manifest["counts"]["train"]["valid_chunks"], args.log_freq, cfg.seed,
                                 validation_callback=validate, run_metadata=metadata, save_every=cfg.save_freq):
            trainer.train(cfg, accelerator=accelerator)


if __name__ == "__main__":
    main()
