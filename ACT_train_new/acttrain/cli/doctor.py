"""检查当前环境与设备，不训练、不修改环境。"""
import argparse
import importlib.metadata
import sys
from pathlib import Path
from acttrain.cli.common import console,entry

def inspect_environment(gpus=1,dataset=None,require_cuda=True):
    if sys.version_info[:2]!=(3,12):raise ValueError('训练环境需要Python3.12')
    versions={name:importlib.metadata.version(name) for name in ('lerobot','torch','torchvision','accelerate','mcap','protobuf','rich','tensorboard','scikit-learn')}
    if versions['lerobot']!='0.4.4':raise ValueError('需要lerobot==0.4.4')
    import torch
    if require_cuda and (sys.platform!='linux' or not torch.cuda.is_available() or torch.cuda.device_count()<gpus):
        raise ValueError(f'需要Linux和至少{gpus}张可见CUDA卡')
    if dataset:
        from acttrain.data.registry import resolve_dataset
        resolve_dataset(dataset)
    return {'python':sys.version.split()[0],'versions':versions,'visible_gpu_count':torch.cuda.device_count(),
            'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

@entry
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,choices=(1,2,4),default=1)
    p.add_argument('--dataset',type=Path);p.add_argument('--cpu',action='store_true')
    a=p.parse_args(argv);console.print(inspect_environment(a.gpus,a.dataset,not a.cpu))

if __name__=='__main__':main()
