"""严格训练配置与LeRobot版本隔离层。"""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import importlib.metadata
import math

@dataclass(frozen=True)
class TrainingConfig:
    gpus: int = 1
    use_velocity: bool = True
    steps: int = 10000
    batch_size_per_gpu: int = 16
    workers_per_gpu: int = 2
    save_freq: int = 1000
    log_freq: int = 100
    val_stride: int = 16
    precision: str = "bf16"
    lr: float = 1e-5
    backbone_lr: float = 1e-5
    dropout: float = .05
    policy_steplr: bool = False
    policy_steplr_step_size: int = 2000
    policy_steplr_gamma: float = .5
    backbone_steplr: bool = False
    backbone_steplr_step_size: int = 2000
    backbone_steplr_gamma: float = .5
    eval_batch_size: int = 16
    eval_max_episodes: int = 8
    eval_seed: int = 42
    seed: int = 42
    keep_top_k: int = 0
    light_report: bool = True

    def __post_init__(self):
        for name in ('gpus','steps','batch_size_per_gpu','save_freq','log_freq','val_stride',
                     'policy_steplr_step_size','backbone_steplr_step_size','eval_batch_size'):
            value = getattr(self,name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}必须为正整数")
        for name in ('workers_per_gpu','keep_top_k','eval_max_episodes'):
            value=getattr(self,name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name}必须为非负整数")
        for name in ('seed','eval_seed'):
            if type(getattr(self,name)) is not int or not 0 <= getattr(self,name) < 2**32:
                raise ValueError(f"{name}必须为[0,2**32)内整数")
        for name in ('use_velocity','policy_steplr','backbone_steplr','light_report'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f"{name}必须为布尔值")
        for name in ('lr','backbone_lr','dropout','policy_steplr_gamma','backbone_steplr_gamma'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"{name}必须为有限数值")
        if self.gpus not in (1,2,4) or self.precision not in ('bf16','no'):
            raise ValueError("仅支持1/2/4卡，precision=bf16/no")
        if min(self.lr,self.backbone_lr)<=0 or not 0<=self.dropout<1:
            raise ValueError("学习率必须>0，dropout需在[0,1)")
        if not all(0<getattr(self,r+'_steplr_gamma')<1 for r in ('policy','backbone')):
            raise ValueError("StepLR gamma需在(0,1)")

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)-{f.name for f in fields(cls)}:
            raise ValueError("配置必须是对象，且不能包含未知键")
        return cls(**value)
    def to_dict(self):
        return asdict(self)

def build_lerobot_config(dataset_root, output, steps=10000, batch_size=16, workers=2, seed=42):
    from acttrain.data.registry import check_manifest
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    if importlib.metadata.version('lerobot') != '0.4.4':
        raise ValueError('必须安装lerobot==0.4.4')
    dataset_root=Path(dataset_root).resolve()
    manifest=check_manifest(dataset_root)
    policy=ACTConfig(chunk_size=32,n_action_steps=4,device='cuda',optimizer_lr=1e-5,
                     optimizer_lr_backbone=1e-5,dropout=.05,kl_weight=10,push_to_hub=False)
    policy.drop_n_last_frames=31
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id=f"yam_local/{manifest['task']}_train",
        root=str(dataset_root/'train'),video_backend='pyav'), policy=policy,output_dir=Path(output).resolve(),
        job_name='acttrain',seed=seed,batch_size=batch_size,num_workers=workers,steps=steps,
        save_freq=1000,log_freq=100,eval_freq=0,wandb=WandBConfig(enable=False))
