"""选择审核数据、确认划分并预处理。"""
import argparse
from pathlib import Path
import math
from acttrain.cli.common import console,entry,project_root,select,ask,safe_name
from acttrain.data.core import read_json,write_json,split_episodes

ALLOWED={'approved_file','review_file','task_id','task_description','seed','val_fraction','validation_episodes',
         'fps','action_offset_frames','action_horizon','max_source_age_ms','image_size','lerobot_version','action_source','output'}

def validate_config(cfg):
    if not isinstance(cfg,dict) or set(cfg)-ALLOWED:
        raise ValueError('预处理配置有未知键')
    for key in ('seed','fps','action_offset_frames','action_horizon','image_size'):
        if type(cfg.get(key)) is not int:raise ValueError(f'{key}必须为整数')
    if (cfg['fps'],cfg['action_offset_frames'],cfg['action_horizon'],cfg['image_size']) != (20,1,32,256):
        raise ValueError('兼容契约固定为20Hz、偏移1帧、32步、256图像')
    if not 0<=cfg['seed']<2**32:raise ValueError('seed超出范围')
    for key in ('val_fraction','max_source_age_ms'):
        value=cfg.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError(f'{key}必须为有限数值')
    if not 0<cfg['val_fraction']<1 or cfg['max_source_age_ms']<=0:
        raise ValueError('验证比例须在(0,1)，源龄须>0')
    if cfg['lerobot_version']!='0.4.4' or cfg['action_source']!='leader':raise ValueError('仅支持LeRobot0.4.4与leader动作')
    for key in ('approved_file','review_file','task_id','task_description','output'):
        if not isinstance(cfg.get(key),str) or not cfg[key].strip():raise ValueError(f'{key}不能为空')
    safe_name(cfg['task_id'])
    return cfg

@entry
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=project_root());p.add_argument('--raw-root',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--yes',action='store_true',help='采用已预览的配置，不交互确认')
    args=p.parse_args(argv);base=args.root.resolve();raw=(args.raw_root or base/'raw_data').resolve()
    if args.yes and args.config is None:raise ValueError('非交互预处理需要--config与--yes')
    if args.config:
        cfg=read_json(args.config)
    else:
        pairs=[]
        for approved in sorted(raw.rglob('*_approved.json')):
            review=approved.with_name(approved.name.replace('_approved.json','_review.json'))
            if review.is_file():pairs.append(((approved,review),str(approved.relative_to(raw))))
        approved,review=select('选择原始数据审核记录',pairs)
        info=read_json(approved)
        ratio=ask('验证集比例',.2,float)
        name=safe_name(ask('新数据集名称',info['task']+'_act'))
        cfg=dict(approved_file=str(approved),review_file=str(review),task_id=info['task'],
            task_description=ask('任务说明',info['task']),seed=42,val_fraction=ratio,fps=20,
            action_offset_frames=1,action_horizon=32,max_source_age_ms=100,image_size=256,
            lerobot_version='0.4.4',action_source='leader',output=str(base/'dataset'/name))
    if args.output:cfg['output']=str(args.output.resolve())
    validate_config(cfg)
    approved=read_json(raw/cfg['approved_file'])
    names=approved.get('approved_episodes',[])
    split=split_episodes(names,cfg['seed'],cfg['val_fraction'],cfg.get('validation_episodes'))
    validation=[name for name,part in split.items() if part=='val']
    console.print('[bold cyan]验证集预览（整条episode，不会跨训练/验证）[/bold cyan]')
    for name in validation:console.print('  '+name)
    console.print(f'训练 {len(names)-len(validation)} 条 / 验证 {len(validation)} 条；seed={cfg["seed"]}')
    if not args.yes:
        manual=ask('可输入验证episode完整相对路径，以逗号分隔；回车采用预览','')
        if manual:
            validation=[x.strip() for x in manual.split(',') if x.strip()]
            split=split_episodes(names,cfg['seed'],cfg['val_fraction'],validation)
            console.print([n for n,s in split.items() if s=='val'])
        if not ask('确认开始预处理？ y/n','n',bool):return
    cfg['validation_episodes']=validation
    target=(raw/cfg['output']).resolve()
    # Store the effective configuration outside the dataset, which the writer must create atomically.
    job=base/'dataset'/'.plans';job.mkdir(parents=True,exist_ok=True)
    import uuid
    config_path=job/(safe_name(target.name)+'_'+uuid.uuid4().hex[:8]+'.json')
    write_json(config_path,cfg)
    from acttrain.data.convert import convert
    output=convert(raw,config_path,target)
    console.print(f'[green]数据集已完成并通过官方读取验证：{output}[/green]')

if __name__=='__main__':main()
