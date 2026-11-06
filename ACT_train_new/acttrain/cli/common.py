from pathlib import Path
import functools
import os
import re
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console=Console()

def project_root():
    return Path(os.environ.get("ACTTRAIN_ROOT",Path(__file__).resolve().parents[2])).expanduser().resolve()

def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}",value) or value in ('.','..'):
        raise ValueError('名称须为1~100位英文字母/数字/下划线/短横线/点，首位为字母或数字')
    return value

def ask(label,default,cast=str):
    while True:
        raw=Prompt.ask(label,default=str(default))
        try:
            if cast is bool:
                if raw.lower() in ('y','yes','true','1'):return True
                if raw.lower() in ('n','no','false','0'):return False
                raise ValueError('请输入y或n')
            return cast(raw)
        except (ValueError,OverflowError) as error:console.print(f'[red]输入无效：{error}[/red]')

def select(title,items):
    if not items:raise ValueError(title+'：没有可用条目')
    table=Table(title=title);table.add_column('编号');table.add_column('条目')
    for i,(_,label) in enumerate(items,1):table.add_row(str(i),label)
    console.print(table)
    while True:
        choice=ask('选择编号',1,int)
        if 1<=choice<=len(items):return items[choice-1][0]
        console.print('[red]编号超出范围[/red]')

def entry(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        try:return function(*args,**kwargs)
        except KeyboardInterrupt:
            console.print('[yellow]已取消[/yellow]');raise SystemExit(130)
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
            console.print(f'[red]{type(error).__name__}: {error}[/red]',markup=False)
            raise SystemExit(2) from error
    return wrapped
