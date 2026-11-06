"""终端界面工具（基于 rich）：统一的标题、表格、带默认值的输入。"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

console = Console(theme=Theme({
    "ok": "bold green", "warn": "bold yellow", "err": "bold red",
    "key": "bold cyan", "dim": "grey58", "title": "bold magenta",
}), highlight=False)


def banner(title: str, subtitle: str = "") -> None:
    console.print(Panel.fit(f"[title]{title}[/title]" + (f"\n[dim]{subtitle}[/dim]" if subtitle else ""),
                            border_style="magenta", box=box.DOUBLE))


def step(index: int, total: int, text: str) -> None:
    console.rule(f"[key]步骤 {index}/{total}[/key]  {text}", style="cyan")


def info(text: str) -> None:
    console.print(f"[dim]›[/dim] {text}")


def ok(text: str) -> None:
    console.print(f"[ok]✔[/ok] {text}")


def warn(text: str) -> None:
    console.print(f"[warn]⚠[/warn] {text}")


def error(text: str) -> None:
    console.print(f"[err]✘[/err] {text}")


def table(title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], caption: str | None = None) -> None:
    t = Table(title=title, box=box.ROUNDED, header_style="bold cyan", caption=caption, caption_style="dim")
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


def kv_table(title: str, items: Sequence[tuple[str, Any]] | dict) -> None:
    items = list(items.items()) if isinstance(items, dict) else list(items)
    t = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False)
    t.add_column(style="key")
    t.add_column()
    for k, v in items:
        t.add_row(str(k), str(v))
    console.print(t)


def choose(title: str, options: Sequence[str], default: int = 1) -> int:
    """展示编号列表让用户选择，返回 0-based 下标。"""
    for i, opt in enumerate(options, 1):
        console.print(f"  [key]{i:>2}[/key]. {opt}")
    choices = [str(i) for i in range(1, len(options) + 1)]
    return int(Prompt.ask(f"[bold]{title}[/bold]", choices=choices, default=str(default), console=console)) - 1


def ask_value(label: str, default: Any, convert: Callable[[str], Any] = str,
              valid: Callable[[Any], bool] = lambda v: True, hint: str = "") -> Any:
    """带默认值的输入；回车即采用默认值，非法输入会重新询问。"""
    shown = f"{label}" + (f" [dim]({hint})[/dim]" if hint else "")
    while True:
        raw = Prompt.ask(shown, default=str(default), console=console)
        try:
            value = convert(raw.strip())
            if valid(value):
                return value
        except (ValueError, OverflowError):
            pass
        error("输入无效，请重新输入")


def ask_float(label: str, default: float, lo: float = -math.inf, hi: float = math.inf, hint: str = "") -> float:
    return ask_value(label, default, float, lambda v: math.isfinite(v) and lo <= v <= hi, hint)


def ask_int(label: str, default: int, lo: int = -10**12, hi: int = 10**12, hint: str = "") -> int:
    return ask_value(label, default, int, lambda v: lo <= v <= hi, hint)


def confirm(label: str, default: bool = True) -> bool:
    return Confirm.ask(label, default=default, console=console)
