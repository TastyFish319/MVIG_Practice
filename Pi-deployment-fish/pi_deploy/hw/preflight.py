"""Passive (read-only) hardware checks before anything is powered."""

from __future__ import annotations

import json
import shutil
import subprocess


def check_can(channels: dict[str, str]) -> list[str]:
    """Problems with the follower CAN links (empty list = OK).  Never changes anything."""
    if shutil.which("ip") is None:
        return ["找不到 ip 命令，无法检查 CAN"]
    try:
        out = subprocess.run(["ip", "-j", "-details", "link", "show", "type", "can"],
                             capture_output=True, text=True, timeout=5)
        links = {v["ifname"]: v for v in json.loads(out.stdout or "[]")}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return [f"读取 CAN 状态失败: {exc}"]
    problems = []
    for side, ch in channels.items():
        link = links.get(ch)
        if link is None:
            problems.append(f"{side} 从臂 {ch}: 不存在（USB-CAN 未插？）")
            continue
        info = link.get("linkinfo", {}).get("info_data", {})
        if "UP" not in link.get("flags", ()):
            problems.append(f"{side} 从臂 {ch}: 未 UP（可运行 SI third_party/i2rt/scripts/reset_all_can.sh）")
        rate = info.get("bittiming", {}).get("bitrate")
        if rate not in (None, 1_000_000):
            problems.append(f"{side} 从臂 {ch}: 波特率 {rate}，应为 1000000")
        if info.get("state") in ("BUS-OFF", "STOPPED"):
            problems.append(f"{side} 从臂 {ch}: 状态 {info.get('state')}")
    return problems


def list_realsense() -> list[str] | None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
