"""一键配置与后台训练→评测→统计。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import fields
from acttrain.cli.common import console,entry,project_root,safe_name,ask,select
from acttrain.data.core import read_json,write_json,file_hash
from acttrain.data.registry import discover,resolve_dataset
from acttrain.training.config import TrainingConfig
from acttrain.training.pipeline import training_command,evaluation_command,run

@entry
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=project_root());p.add_argument('--dataset',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--name');p.add_argument('--devices')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--foreground',action='store_true')
    defaults=TrainingConfig()
    for field in fields(TrainingConfig):
        default=getattr(defaults,field.name)
        kind=bool if type(default) is bool else type(default)
        opts={'default':None}
        if kind is bool:opts['action']=argparse.BooleanOptionalAction
        else:opts['type']=kind
        p.add_argument('--'+field.name.replace('_','-'),**opts)
    a=p.parse_args(argv);base=a.root.resolve()
    if a.dataset:dataset=resolve_dataset(a.dataset)
    else:
        dataset=select('选择已验证的数据集',[(target,f'{item.name} — train {m["counts"]["train"]["episodes"]} / val {m["counts"]["val"]["episodes"]}') for item,target,m in discover(base/'dataset')])
    values=read_json(a.config) if a.config else {}
    if not isinstance(values,dict):raise ValueError('训练配置必须为对象')
    values.update({f.name:getattr(a,f.name) for f in fields(TrainingConfig) if getattr(a,f.name) is not None})
    interactive=a.config is None and a.name is None
    if interactive:
        console.print('[bold cyan]回车采用默认值；batch是每卡batch，steps为优化器更新次数。[/bold cyan]')
        labels={'gpus':'GPU数量（1/2/4）','use_velocity':'使用100ms历史速度 y/n','steps':'训练步数',
          'batch_size_per_gpu':'每卡batch','workers_per_gpu':'每卡数据进程','save_freq':'checkpoint间隔',
          'log_freq':'日志间隔','val_stride':'checkpoint验证stride','lr':'策略初始lr','backbone_lr':'视觉骨干初始lr',
          'dropout':'dropout','keep_top_k':'保留top-k+final（0=全部）','eval_max_episodes':'最终评测最多episode（0=全val）'}
        for key,label in labels.items():
            default=values.get(key,getattr(defaults,key))
            values[key]=ask(label,default,bool if type(default) is bool else type(default))
        for role,label in (('policy','策略'),('backbone','视觉骨干')):
            enabled=ask(label+'启用StepLR y/n',values.get(role+'_steplr',False),bool)
            values[role+'_steplr']=enabled
            if enabled:
                for suffix,typ in (('step_size',int),('gamma',float)):
                    key=role+'_steplr_'+suffix;values[key]=ask(label+' '+suffix,values.get(key,getattr(defaults,key)),typ)
    config=TrainingConfig.from_dict(values)
    name=safe_name(a.name or ask('训练名称','act_run1'))
    output=base/'runs'/name;job=base/'runs'/'.jobs'/name
    if output.exists() or job.exists():raise FileExistsError('训练名或后台任务记录已存在，请换名称')
    devices=a.devices
    if interactive:devices=ask('GPU物理编号，逗号分隔（先确认空闲）',os.environ.get('CUDA_VISIBLE_DEVICES',','.join(str(i) for i in range(config.gpus))))
    if devices is not None:
        ids=[x.strip() for x in devices.split(',')]
        if len(ids)!=config.gpus or len(set(ids))!=len(ids) or not all(x.isdigit() for x in ids):
            raise ValueError('设备编号必须互不重复，数量等于所选卡数')
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(ids)
    console.print(f'[cyan]state{21 if config.use_velocity else 14} | {config.gpus}卡 | 每卡{config.batch_size_per_gpu} / 总batch {config.gpus*config.batch_size_per_gpu} | {config.steps} steps[/cyan]')
    plan={'config':config.to_dict(),'dataset':str(dataset),'output':str(output),'manifest_sha256':file_hash(dataset/'manifest.json')}
    if a.dry_run:
        console.print(plan);console.print(training_command(config,dataset,output));console.print(evaluation_command(config,dataset,output));return
    from acttrain.cli.doctor import inspect_environment
    inspect_environment(config.gpus,dataset)
    if config.precision=='bf16':
        import torch
        if not torch.cuda.is_bf16_supported():raise ValueError('当前设备不支持BF16，请用--precision no')
    if interactive and not ask('确认开始本次训练？ y/n','n',bool):return
    job.mkdir(parents=True,exist_ok=False);write_json(job/'plan.json',plan)
    if a.foreground:return run(job/'plan.json')
    with (job/'run.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','acttrain.training.pipeline','--plan',str(job/'plan.json')],
          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
          env={**os.environ,'PYTHONUNBUFFERED':'1'})
    write_json(job/'launch.json',{'supervisor_pid':process.pid,'log':'run.log','visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
    console.print(f'[green]后台已启动 PID={process.pid}；日志：{job/"run.log"}；状态：{job/"status.json"}[/green]')

if __name__=='__main__':main()
