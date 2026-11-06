import argparse
from pathlib import Path
from acttrain.cli.common import entry,console
from acttrain.data.registry import register

@entry
def main(argv=None):
    p=argparse.ArgumentParser(description='只读引用旧leader数据集，保留原始划分与指纹')
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args(argv);console.print(register(a.source,a.output))

if __name__=='__main__':main()
