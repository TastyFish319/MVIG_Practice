"""后台监督：训练成功→保留策略→评测成功→统计ZIP。"""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import shutil
from acttrain.data.core import read_json, write_json
from acttrain.training.config import TrainingConfig


def training_command(config,dataset,output):
    c=config.to_dict()
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
             f'--nproc-per-node={config.gpus}','--module','acttrain.training.worker',
             '--dataset',str(dataset),'--output',str(output),'--execute']
    for key in ('gpus','steps','batch_size_per_gpu','workers_per_gpu','save_freq','log_freq','val_stride',
                'precision','lr','backbone_lr','dropout','seed'):
        command += ['--'+key.replace('_','-'),str(c[key])]
    command += ['--use-velocity' if config.use_velocity else '--no-use-velocity']
    for role in ('policy','backbone'):
        command += ['--'+('' if c[role+'_steplr'] else 'no-')+role+'-steplr',
            '--'+role+'-steplr-step-size',str(c[role+'_steplr_step_size']),
            '--'+role+'-steplr-gamma',str(c[role+'_steplr_gamma'])]
    return command


def evaluation_command(config,dataset,output):
    return [sys.executable,'-u','-m','acttrain.evaluation.runner','--dataset',str(dataset),
            '--checkpoints',str(output),'--output',str(output/'evaluation'),'--gpus',str(config.gpus),
            '--max-episodes',str(config.eval_max_episodes),'--seed',str(config.eval_seed),
            '--batch-size',str(config.eval_batch_size)]


def retain_checkpoints(run,top_k):
    if top_k==0:return
    import csv
    with (run/'metrics/validation_metrics.csv').open(encoding='utf-8') as stream:
        records=list(csv.DictReader(stream))
    checkpoints=run/'checkpoints'
    numeric={int(p.name):p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit() and not p.is_symlink()}
    if not numeric or {int(r['step']) for r in records} != set(numeric):
        raise ValueError('checkpoint与验证结果不一一对应，拒绝精简权重')
    kept={int(r['step']) for r in sorted(records,key=lambda r:float(r['val_loss']))[:top_k]} | {max(numeric)}
    removed=[]
    for step,path in numeric.items():
        if step not in kept:
            if path.resolve().parent!=checkpoints.resolve():
                raise ValueError('拒绝删除越界checkpoint')
            shutil.rmtree(path);removed.append(step)
    write_json(run/'checkpoint_retention.json',{'top_k':top_k,'kept_steps':sorted(kept),'removed_steps':sorted(removed),
               'criterion':'val_prior_normalized_L1','timing':'after_training_before_evaluation'})


def run(plan_path):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path)
    config=TrainingConfig.from_dict(plan['config']);output=Path(plan['output']);dataset=Path(plan['dataset'])
    if output.exists():raise FileExistsError('训练目录已存在，拒绝覆盖')
    child=None;started=time.time();stage='starting'
    spawning=False;pending_signal=None
    def status(state,**extra):
        data={'status':state,'stage':stage,'supervisor_pid':os.getpid(),'elapsed_seconds':time.time()-started,
              'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),**extra}
        write_json(plan_path.parent/'status.json',data)
        if output.is_dir():write_json(output/'status.json',data)
    def stop(signum,frame):
        nonlocal pending_signal
        if spawning:
            pending_signal=signum
            return
        raise InterruptedError(f'收到停止信号{signum}')
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous:signal.signal(sig,stop)
    try:
        for stage,command in (('training',training_command(config,dataset,output)),
                              ('evaluation',evaluation_command(config,dataset,output))):
            status('running')
            if stage=='evaluation':
                write_json(output/'config.json',{'training':config.to_dict(),'dataset_manifest_sha256':plan['manifest_sha256']})
                retain_checkpoints(output,config.keep_top_k)
            spawning=True;pending_signal=None
            try:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True,
                                       env={**os.environ,'PYTHONUNBUFFERED':'1'})
            finally:
                spawning=False
            if pending_signal is not None:stop(pending_signal,None)
            code=child.wait();child=None
            if code:raise RuntimeError(f'{stage}退出码{code}，请查看日志')
        stage='report';status('running')
        from acttrain.evaluation.report import build_report
        archive=build_report(output,light=config.light_report)
        stage='completed';status('completed',archive=archive.name)
    except BaseException as error:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        if output.is_dir():
            write_json(output/'config.json',{'training':config.to_dict(),'dataset_manifest_sha256':plan['manifest_sha256']})
        status('failed',error=str(error));raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    import argparse
    p=argparse.ArgumentParser(description='内部后台监督进程');p.add_argument('--plan',required=True,type=Path)
    run(p.parse_args().plan)

if __name__=='__main__':main()
