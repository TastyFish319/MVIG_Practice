import argparse
from pathlib import Path
from acttrain.cli.common import console,entry
from acttrain.evaluation.report import build_report

@entry
def main(argv=None):
    p=argparse.ArgumentParser(description='将完整评测整理为离线HTML和statistics.zip')
    p.add_argument('run',type=Path);p.add_argument('--evaluation',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--full',action='store_true',help='额外包含NPZ和SVG')
    a=p.parse_args(argv);console.print(build_report(a.run,a.evaluation,not a.full,a.output))

if __name__=='__main__':main()
