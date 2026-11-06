"""最终服务器验收入口；只在新工程路径写产物，旧数据只读。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def call(args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--review-root',type=Path,required=True);p.add_argument('--devices',default='1,2')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];os.chdir(root)
    # The sole centralized acceptance suite. Repeat only failed checks after a repair.
    call(['-m','unittest','discover','-s','tests','-v'])
    for cli in ('preprocess','train','evaluate','report','doctor','dataset'):
        call(['-m','acttrain.cli.'+cli,'--help'])
    from acttrain.data.core import read_json,write_json
    review=read_json(a.review_root/'human_review.json');approved=read_json(a.review_root/'approved_episodes.json')
    names=approved['approved_episodes'][:2]
    records=[e for e in review['episodes'] if e['source_relative'] in names]
    approved={**approved,'approved_episodes':names,'pending_count':0,'acceptance_subset':True}
    review={**review,'episodes':records,'acceptance_subset':True}
    provenance=root/'raw_data/acceptance_review';provenance.mkdir(parents=True,exist_ok=False)
    write_json(provenance/'subset_approved.json',approved);write_json(provenance/'subset_review.json',review)
    cfg=read_json(root/'configs/preprocess.example.json')
    cfg.update(approved_file=str(provenance/'subset_approved.json'),review_file=str(provenance/'subset_review.json'),
               task_id=approved['task'],task_description='真实数据最终流程验收子集',validation_episodes=[names[-1]],
               output=str(root/'dataset/acceptance_real'))
    write_json(provenance/'config.json',cfg)
    call(['-m','acttrain.cli.preprocess','--raw-root',a.source_root,'--config',provenance/'config.json','--yes'])
    dataset=root/'dataset/acceptance_real'
    from acttrain.training.config import TrainingConfig
    from acttrain.training.pipeline import run
    from acttrain.data.core import file_hash
    for gpus,velocity,devices in ((1,True,a.devices.split(',')[0]),(2,False,a.devices)):
        os.environ['CUDA_VISIBLE_DEVICES']=devices
        config=TrainingConfig(gpus=gpus,use_velocity=velocity,steps=2,batch_size_per_gpu=1,workers_per_gpu=0,
            save_freq=1,log_freq=1,val_stride=256,eval_batch_size=16,eval_max_episodes=8,
            policy_steplr=True,policy_steplr_step_size=1,policy_steplr_gamma=.5,
            backbone_steplr=False)
        output=root/'runs'/f'acceptance_state{21 if velocity else 14}_{gpus}gpu'
        job=root/'runs/.jobs'/output.name;job.mkdir(parents=True,exist_ok=False)
        write_json(job/'plan.json',{'config':config.to_dict(),'dataset':str(dataset),'output':str(output),
                   'manifest_sha256':file_hash(dataset/'manifest.json')})
        # A new process tests installed package resolution and fresh CUDA environment.
        call(['-m','acttrain.training.pipeline','--plan',job/'plan.json'])
        if read_json(output/'status.json')['status']!='completed':raise RuntimeError('流水线状态不是completed')
        from zipfile import ZipFile
        with ZipFile(output/(output.name+'_statistics.zip')) as bundle:
            if bundle.testzip() is not None or 'report.html' not in bundle.namelist():raise RuntimeError('ZIP损坏')
            if any(n.endswith('.safetensors') for n in bundle.namelist()):raise RuntimeError('ZIP误含权重')
    write_json(root/'runs/acceptance_result.json',{'status':'passed','real_preprocessing':True,
        'real_act_single_gpu':'state21','real_act_dual_gpu':'state14','four_gpu':'not_run_gpu0_reserved',
        'steps_per_run':2,'closed_loop_robot_test':False})

if __name__=='__main__':main()
