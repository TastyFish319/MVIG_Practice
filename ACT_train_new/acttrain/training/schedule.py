"""Optional optimizer-step StepLR preset for ACT v3's official LeRobot trainer."""
from contextlib import contextmanager
from dataclasses import dataclass
import math

from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim.lr_scheduler import LambdaLR, StepLR


@LRSchedulerConfig.register_subclass("act_step_lr")
@dataclass
class StepLRSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int | None = None
    step_size: int = 2000
    gamma: float = 0.5

    def __post_init__(self):
        if isinstance(self.step_size, bool) or not isinstance(self.step_size, int) or self.step_size < 1:
            raise ValueError("StepLR step_size must be a positive integer of optimizer steps")
        if not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError("StepLR gamma must be finite and in (0,1)")
        if self.num_warmup_steps is not None:
            raise ValueError("StepLR does not use warmup")

    def build(self, optimizer, num_training_steps):
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


@dataclass
class GroupStepDecay:
    """Callable state is persisted by PyTorch LambdaLR alongside its step counter."""
    enabled: bool
    step_size: int
    gamma: float

    def __call__(self, step):
        return self.gamma ** (step // self.step_size) if self.enabled else 1.0


@LRSchedulerConfig.register_subclass("act_group_step_lr")
@dataclass
class GroupStepLRSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int | None = None
    policy_enabled: bool = False
    policy_step_size: int = 2000
    policy_gamma: float = .5
    backbone_enabled: bool = False
    backbone_step_size: int = 2000
    backbone_gamma: float = .5

    def __post_init__(self):
        if self.num_warmup_steps is not None:
            raise ValueError("StepLR does not use warmup")
        for role in ("policy", "backbone"):
            if not isinstance(getattr(self, role + "_enabled"), bool):
                raise ValueError("StepLR enabled flags must be booleans")
            StepLRSchedulerConfig(step_size=getattr(self, role + "_step_size"), gamma=getattr(self, role + "_gamma"))

    def build(self, optimizer, num_training_steps):
        # LeRobot 0.4.4 ACTPolicy.get_optim_params orders policy then model.backbone.
        if len(optimizer.param_groups) != 2:
            raise ValueError("Independent ACT StepLR requires policy and backbone optimizer groups")
        return LambdaLR(optimizer, [GroupStepDecay(self.policy_enabled, self.policy_step_size, self.policy_gamma),
                                    GroupStepDecay(self.backbone_enabled, self.backbone_step_size, self.backbone_gamma)])

    def metadata(self):
        return {"type": self.type, "unit": "optimizer_step", **{
            role: {"enabled": getattr(self, role + "_enabled"), "step_size": getattr(self, role + "_step_size"),
                   "gamma": getattr(self, role + "_gamma")} for role in ("policy", "backbone")}}


@contextmanager
def step_lr_preset(policy, scheduler):
    """Let cfg.validate() select StepLR while retaining ACT's optimizer groups.

    LeRobot resets cfg.scheduler from the policy preset during validation. Scope
    this override to v3 training, and let the official trainer step/save it.
    """
    if scheduler is None:
        yield
        return
    policy_class = type(policy)
    original = policy_class.get_scheduler_preset
    policy_class.get_scheduler_preset = lambda self: scheduler
    try:
        yield
    finally:
        policy_class.get_scheduler_preset = original
