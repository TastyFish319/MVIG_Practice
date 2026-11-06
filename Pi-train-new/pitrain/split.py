"""训练集 / 验证集划分：分层均匀抽样。

为什么不用纯随机？采集是按时间顺序进行的，操作员手法、光照、物体摆放会随时间漂移。
把按编号排序的 episode 均分成 k 段，每段随机抽 1 条作验证集，可以保证验证集
"均匀覆盖"整个采集过程，而不会扎堆在某一段。

    排序后的 80 条:  |----段1----|----段2----| ... |----段16----|
    每段抽 1 条:          ^            ^                  ^
"""
from __future__ import annotations

import random


def val_count(n: int, ratio: float) -> int:
    if n < 2:
        raise ValueError("至少需要 2 条 episode 才能划分训练/验证集")
    return min(n - 1, max(1, round(n * ratio)))


def stratified_split(names: list[str], ratio: float, seed: int) -> list[str]:
    """返回验证集 episode 名称列表（已排序）。"""
    names = sorted(names)
    k = val_count(len(names), ratio)
    rng = random.Random(seed)
    chosen = []
    n = len(names)
    for i in range(k):
        lo, hi = round(i * n / k), round((i + 1) * n / k)
        chosen.append(names[rng.randrange(lo, max(lo + 1, hi))])
    return sorted(chosen)


def parse_manual(text: str, names: list[str]) -> list[str]:
    """解析用户手动输入的验证集编号，如 "3, 15, 27" 或 "episode_0003 episode_0015"。"""
    by_number = {int(n.split("_")[-1]): n for n in names}
    picked = []
    for token in text.replace(",", " ").replace("，", " ").split():
        token = token.strip()
        if token in names:
            picked.append(token)
        elif token.isdigit() and int(token) in by_number:
            picked.append(by_number[int(token)])
        else:
            raise ValueError(f"无法识别或不在通过列表中: {token}")
    picked = sorted(set(picked))
    if not picked or len(picked) >= len(names):
        raise ValueError("验证集不能为空，也不能包含全部 episode")
    return picked
