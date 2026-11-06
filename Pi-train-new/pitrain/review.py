"""读取人工审核记录，筛选出 approved 的 episode，并做一致性校验。

审核记录有两份（由审核工具导出）::

    <task>_approved.json   # 只列出通过的 episode（相对路径）
    <task>_review.json     # 每条 episode 的决定 / 技术状态 / 源文件 sha256

两份必须属于同一个任务（task 与 source_fingerprint 相同），且 approved 列表要与
review 中 decision == "approved" 的集合完全一致，否则拒绝继续。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .nedf3 import read_json


@dataclass
class ApprovedEpisode:
    name: str          # episode_0000
    path: Path         # 绝对路径
    review: dict       # review.json 中这一条的完整记录


@dataclass
class ReviewPair:
    approved_file: Path
    review_file: Path


def find_review_files(folder: Path, raw_root: Path | None = None) -> list[ReviewPair]:
    """在数据文件夹（及 raw_data 根目录）中寻找成对的审核文件。"""
    candidates = sorted(folder.glob("*_approved.json"))
    if raw_root is not None and not candidates:
        candidates = sorted(raw_root.glob(f"{folder.name}*_approved.json"))
    pairs = []
    for approved in candidates:
        review = approved.with_name(approved.name.replace("_approved.json", "_review.json"))
        if review.is_file():
            pairs.append(ReviewPair(approved, review))
    return pairs


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_approved(folder: Path, pair: ReviewPair, verify_hash: bool = True, progress=None) -> tuple[str, list[ApprovedEpisode], dict]:
    """返回 (任务名, 通过的 episode 列表, 统计信息)。"""
    approved = read_json(pair.approved_file)
    review = read_json(pair.review_file)
    if approved.get("task") != review.get("task"):
        raise ValueError("approved 与 review 的 task 不一致")
    if approved.get("source_fingerprint") != review.get("source_fingerprint"):
        raise ValueError("approved 与 review 的 source_fingerprint 不一致（不是同一次审核导出）")
    if approved.get("pending_count", 0):
        raise ValueError(f"审核尚未完成：还有 {approved['pending_count']} 条待定")

    records = review.get("episodes", [])
    by_name = {r["episode"]: r for r in records}
    if len(by_name) != len(records):
        raise ValueError("review.json 中存在重复 episode")
    approved_names = sorted(Path(p).name for p in approved.get("approved_episodes", []))
    decided = sorted(n for n, r in by_name.items() if r.get("decision") == "approved")
    if approved_names != decided:
        diff = set(approved_names) ^ set(decided)
        raise ValueError(f"approved 列表与 review 决定不一致: {sorted(diff)[:10]}")

    result: list[ApprovedEpisode] = []
    missing = []
    for name in approved_names:
        path = (folder / name).resolve()
        rec = by_name[name]
        if not path.is_dir():
            missing.append(name)
            continue
        if rec.get("technical_status") == "fail":
            raise ValueError(f"{name}: 技术检查为 fail 却被人工通过，请复核")
        if verify_hash:
            for item in rec.get("source_files", []):
                f = path / "data" / item["name"]
                if not f.is_file() or sha256_of(f) != item["sha256"]:
                    raise ValueError(f"{name}: 源文件 {item['name']} 缺失或 sha256 不匹配（数据被改动过？）")
        result.append(ApprovedEpisode(name=name, path=path, review=rec))
        if progress is not None:
            progress()
    if missing:
        raise FileNotFoundError(f"以下 approved episode 在 {folder} 中不存在: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    stats = {
        "total": len(records),
        "approved": len(result),
        "rejected": sum(r.get("decision") == "rejected" for r in records),
        "warning_approved": sum(by_name[e.name].get("technical_status") == "warning" for e in result),
    }
    return approved["task"], result, stats
