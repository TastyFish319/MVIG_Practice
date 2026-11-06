"""Four-rank launch, deterministic legal-window sampling and rank-safe telemetry."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import logging
import os

from acttrain.data.core import write_json
from acttrain.training.metrics import FIELDS, Recorder

REDUCED_FIELDS = FIELDS[4:]


def launch_coordinates(environ=None):
    environ = os.environ if environ is None else environ
    if int(environ.get("WORLD_SIZE", "0")) != 4 or int(environ.get("LOCAL_WORLD_SIZE", "0")) != 4:
        raise ValueError("Launch with torchrun --standalone --nnodes=1 --nproc-per-node=4")
    rank, local_rank = int(environ.get("RANK", "-1")), int(environ.get("LOCAL_RANK", "-1"))
    if not 0 <= rank < 4 or rank != local_rank:
        raise ValueError("Expected one node with global/local ranks 0..3")
    return rank, local_rank


def seeded_sampler_class(base_class, seed):
    import torch

    class SeededEpisodeSampler(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epoch = 0
            self.generator = torch.Generator()

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            if not self.shuffle:
                return iter(self.indices)
            self.generator.manual_seed(seed + self.epoch)
            order = torch.randperm(len(self.indices), generator=self.generator).tolist()
            return (self.indices[index] for index in order)

    return SeededEpisodeSampler


def metric_payload(values, samples, device="cpu"):
    import torch
    if samples < 1:
        raise ValueError("A nonempty local batch is required")
    return torch.tensor([samples, *[float(values.get(key, float("nan"))) * samples for key in REDUCED_FIELDS]],
                        dtype=torch.float64, device=device)


def decode_payload(payload):
    values = payload.detach().cpu().tolist()
    samples = int(values[0])
    if samples < 1:
        raise ValueError("No global samples were reduced")
    return samples, {key: value / samples for key, value in zip(REDUCED_FIELDS, values[1:])}


def main_process_action(accelerator, callback):
    """All ranks participate; propagate rank0 setup failures instead of hanging peers."""
    import torch
    error = None
    failed = torch.zeros(1, device=accelerator.device, dtype=torch.int32)
    if accelerator.is_main_process:
        try:
            callback()
        except Exception as exc:
            error = exc
            failed.fill_(1)
    failed = accelerator.reduce(failed, reduction="sum")
    if failed.item():
        raise RuntimeError(f"Rank0 setup failed: {error or 'see rank0 traceback/log'}") from error


@contextmanager
def capture_distributed(trainer, accelerator, output, total_steps, legal_windows, log_every, seed,
                        validation_callback=None, run_metadata=None, save_every=None):
    recorder = Recorder(output, log_every, legal_windows) if accelerator.is_main_process else None
    original_update = trainer.update_policy
    original_save = getattr(trainer, "save_checkpoint", None)
    original_sampler = getattr(trainer, "EpisodeAwareSampler", None)
    if original_sampler is not None:
        trainer.EpisodeAwareSampler = seeded_sampler_class(original_sampler, seed)
    pending, window_steps, global_samples = None, 0, 0

    @wraps(original_update)
    def update(*args, **kwargs):
        nonlocal pending, window_steps, global_samples
        tracker, extra = original_update(*args, **kwargs)
        batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
        if not isinstance(batch, dict) or "action" not in batch:
            raise ValueError("Expected official ACT training batch")
        values = tracker.to_dict(use_avg=False)
        values.update(extra or {})
        optimizer = kwargs.get("optimizer", args[3] if len(args)>3 else None)
        if optimizer is not None and len(optimizer.param_groups)==2:
            values["backbone_lr"] = optimizer.param_groups[1]["lr"]
        # Accumulate Python/scalar telemetry on CPU; transfer only at logging boundaries.
        payload = metric_payload(values, len(batch["action"]))
        pending = payload if pending is None else pending + payload
        window_steps += 1
        step = int(tracker.steps) + 1
        saving = save_every is not None and step % save_every == 0
        if step == 1 or step % log_every == 0 or step == total_steps or saving:
            # Every rank must enter this collective, including ranks that never write files.
            reduced = accelerator.reduce(pending.to(accelerator.device), reduction="sum")
            samples, means = decode_payload(reduced)
            global_samples += samples
            if recorder is not None:
                recorder.write_window(step, means, global_samples, window_steps)
                if step == 1 and run_metadata is not None:
                    write_json(recorder.directory.parent / "distributed_config.json", run_metadata)
            pending, window_steps = None, 0
        return tracker, extra

    def save(*args, **kwargs):
        if not accelerator.is_main_process:
            raise RuntimeError("Only rank0 may save checkpoints")
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
        if original_sampler is not None:
            trainer.EpisodeAwareSampler = original_sampler
        if recorder is not None:
            try:
                # No collectives here: official trainer may have destroyed its group.
                recorder.finish(status)
            except Exception:
                logging.exception("Could not finalize distributed curves; training result preserved")
