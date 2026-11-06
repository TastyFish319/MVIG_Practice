"""Terminal UI: start-up menus, live dashboard, single-key control (no Enter needed)."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time

import numpy as np

from act_deploy.app.engine import PHASES

if os.name == "nt":
    os.system("")  # enable ANSI escape codes in the Windows console

C = {"reset": "\x1b[0m", "bold": "\x1b[1m", "dim": "\x1b[2m", "red": "\x1b[31m", "green": "\x1b[32m",
     "yellow": "\x1b[33m", "blue": "\x1b[34m", "magenta": "\x1b[35m", "cyan": "\x1b[36m", "grey": "\x1b[90m",
     "inv": "\x1b[7m"}


def c(text, *styles) -> str:
    return "".join(C[s] for s in styles) + str(text) + C["reset"]


def rule(title: str = "") -> str:
    w = min(shutil.get_terminal_size((100, 30)).columns, 110)
    if not title:
        return c("─" * w, "grey")
    return c("── ", "grey") + c(title, "bold") + " " + c("─" * max(4, w - len(title) * 2 - 4), "grey")


def banner(lines: list[str]) -> None:
    print()
    print(c("  ╔═══════════════════════════════════════════════╗", "cyan"))
    print(c("  ║        ACT 推理部署控制台 · I2RT YAM 双臂        ║", "cyan", "bold"))
    print(c("  ╚═══════════════════════════════════════════════╝", "cyan"))
    for ln in lines:
        print("   " + ln)
    print()


def choose(title: str, options: list[tuple[str, str, bool]], default: int = 0) -> int:
    """Numbered menu.  options = [(label, detail, enabled)].  Enter = default.  q = quit."""
    print(rule(title))
    for i, (label, detail, enabled) in enumerate(options):
        mark = c("➤", "green", "bold") if i == default else " "
        num = c(f"[{i + 1}]", "bold" if enabled else "grey")
        lab = label if enabled else c(label + "（不可用）", "grey")
        print(f"  {mark} {num} {lab}")
        if detail:
            print(f"        {c(detail, 'grey')}")
    while True:
        try:
            raw = input(c(f"  选择 1-{len(options)}（Enter = {default + 1}，q 退出）: ", "cyan")).strip().lower()
        except EOFError:
            raw = "q"
        if raw == "q":
            raise SystemExit(0)
        if raw == "":
            idx = default
        elif raw.isdigit() and 1 <= int(raw) <= len(options):
            idx = int(raw) - 1
        else:
            print(c("  输入无效", "red"))
            continue
        if not options[idx][2]:
            print(c("  该选项不可用", "red"))
            continue
        print(c(f"  ✔ {options[idx][0]}", "green"))
        print()
        return idx


# ---------------------------------------------------------------------- keys
class KeyReader:
    """Reads single key presses in a background thread (cbreak mode, no echo)."""

    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._old = None
        if os.name == "posix" and sys.stdin.isatty():
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # keeps ISIG: Ctrl+C still raises SIGINT
        threading.Thread(target=self._loop, daemon=True, name="keys").start()

    def _loop(self) -> None:
        if os.name == "nt":
            import msvcrt

            while not self._stop.is_set():
                if msvcrt.kbhit():
                    self._put(msvcrt.getwch())
                else:
                    time.sleep(0.02)
            return
        import select

        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
                if ch:
                    self._put(ch)

    def _put(self, ch: str) -> None:
        if ch in ("\r", "\n"):
            self.q.put("enter")
        elif ch == " ":
            self.q.put("space")
        elif ch == "\x1b":
            self.q.put("esc")
        else:
            self.q.put(ch.lower())

    def get(self, timeout: float) -> str | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


KEYMAP = {"enter": "next", "space": "estop", "f": "finish", "p": "pause", "z": "zero", "i": "init",
          "r": "release", "q": "release"}
HELP = [
    ("Enter", "下一步：移动到初始姿态 / 开始执行 / 继续"),
    ("f", "结束本次执行 → 平滑回零位"),
    ("p", "暂停（原地保持）/ 继续"),
    ("空格", "急停：立即原地保持（推理停止）"),
    ("z", "回零位（急停/故障后为紧急回零，故障会先复位）"),
    ("i", "回初始姿态"),
    ("r / q", "回零位后释放电机并退出程序"),
    ("h", "显示/隐藏本帮助"),
    ("Ctrl+C", "1次=急停  2次=回零释放退出  3次=立即退出（worker 保持通电，start.sh 自动救援）"),
]

MODE_NAMES = {"real": "真机执行", "shadow": "影子模式(只推理不动)", "sim": "模拟演练(无硬件)"}


def _vec(v, idx, fmt="{:6.2f}") -> str:
    if v is None:
        return "  —"
    return " ".join(fmt.format(v[i]) for i in idx)


class Dashboard:
    def __init__(self, engine, header: dict, server_alive):
        self.e, self.header, self.server_alive = engine, header, server_alive
        self.show_help = False
        self.extra: list[str] = []  # messages from main (e.g. Ctrl+C hints)

    def render(self) -> str:
        s = self.e.snapshot()
        h = self.header
        name, colour, hint = PHASES.get(s["phase"], (s["phase"], "grey", ""))
        L = []
        L.append(c(" ACT 部署控制台 ", "inv", "bold") + "  "
                 + f"任务 {c(h['task'], 'bold')}  │  权重 {c(h['ckpt'], 'bold')}  │  "
                 + f"模式 {c(MODE_NAMES[s['mode']], 'red' if s['mode'] == 'real' else 'cyan', 'bold')}  │  {h['preset']}")
        L.append(rule())
        line = f" 状态   {c('●', colour)} {c(name, colour, 'bold')}"
        if s["elapsed"] is not None:
            line += f"   第 {s['episode']} 次  {s['elapsed']:5.1f} / {s['max_episode_s']:.0f} s"
        if s["route_remaining"] is not None:
            line += f"   {s['route_label']} 剩余 {s['route_remaining']:.1f}s"
        if s["release_requested"] and s["phase"] not in ("RELEASED",):
            line += c("   [结束后释放退出]", "yellow")
        L.append(line)
        L.append(f" 下一步 {c(hint, 'cyan')}")
        if s["fault"]:
            L.append(f" 故障   {c(s['fault'], 'red')}")
        L.append(rule("运行指标"))
        lat = s["lat_ms"]
        lat_txt = f"中位 {lat[0]:.0f}ms / 最大 {lat[1]:.0f}ms" if lat else "—"
        inf_txt = f"{s['infer_ms']:.1f}ms" if s["infer_ms"] is not None else "—"
        sched = s["sched"] or {}
        age = sched.get("plan_age_s")
        plan_txt = f"计划已用 {age:.2f}s" if age is not None and s["phase"] == "RUNNING" else ""
        starve = sched.get("starved_s", 0.0)
        if starve and s["phase"] == "RUNNING":
            plan_txt += c(f" 计划耗尽 {starve:.2f}s!", "red")
        vel = c("补零(开局)", "yellow") if s["velocity_padded"] else "正常"
        L.append(f" 推理   次数 {s['n_infer']:<5} 往返 {lat_txt}  模型 {inf_txt}  {plan_txt}  速度输入 {vel}")
        if s["policy_fail"]:
            msg = "连续失败 %d: %s" % (s["policy_fail"], s["policy_error"])
            L.append("        " + c(msg, "red"))
        cams = "  ".join(
            f"{n} " + (c(f"{v['fps']:.0f}fps", 'green') if v["age_s"] is not None and v["age_s"] < 0.2
                       else c("无画面", 'red')) for n, v in s["cams"].items())
        srv = c("● 正常", "green") if self.server_alive() else c("● 已退出", "red")
        L.append(f" 硬件   相机 {cams}   模型服务 {srv}   臂反馈 {s['arm_age_ms']:.0f}ms   "
                 f"控制环最大延迟 {s['loop_late_ms']:.1f}ms")
        sc = s["safety"]
        L.append(f" 安全   范围裁剪 {sc['range']}  偏差牵引 {sc['leash']}  限速 {sc['rate']}"
                 + (f"   预热: {s['warmup']}" if s["warmup"] else "   预热: 进行中…"))
        L.append(rule("关节 (rad) / 夹爪 (0闭-1开)"))
        st, cmd, pt = s["state"], s["cmd"], s["policy_target"]
        L.append(c("        j0     j1     j2     j3     j4     j5  │ 夹爪", "grey"))
        L.append(f" 右实际 {_vec(st, range(7, 13))} │ {_vec(st, [13])}")
        L.append(f" 右指令 {_vec(cmd, range(7, 13))} │ {_vec(cmd, [13])}")
        if s["mode"] == "shadow" and pt is not None:
            L.append(c(f" 模型想要 {_vec(pt, range(7, 13))} │ {_vec(pt, [13])}", "magenta"))
        L.append(f" 左实际 {_vec(st, range(0, 6))} │ {_vec(st, [6])}")
        if s["results"]:
            L.append(rule("历次执行"))
            for r in s["results"][-3:]:
                L.append(f"  #{r['episode']}  {r['duration']:5.1f}s  {r['reason']}")
        L.append(rule("最近事件"))
        icons = {"info": c("·", "grey"), "ok": c("✔", "green"), "warn": c("!", "yellow"), "error": c("✖", "red")}
        for t, level, msg in s["events"]:
            col = {"error": "red", "warn": "yellow"}.get(level)
            L.append(f"  {c(t, 'grey')} {icons.get(level, '·')} " + (c(msg, col) if col else msg))
        for m in self.extra[-2:]:
            L.append(c(f"  ▶ {m}", "yellow", "bold"))
        if self.show_help:
            L.append(rule("按键帮助"))
            for k, d in HELP:
                L.append(f"  {c(k.ljust(7), 'bold')} {d}")
        L.append(rule())
        L.append(" " + "  ".join(f"{c(k, 'inv')} {d}" for k, d in
                                 (("Enter", "下一步"), ("f", "结束"), ("p", "暂停"), ("空格", "急停"),
                                  ("z", "回零"), ("i", "初始位"), ("r", "释放退出"), ("h", "帮助"))))
        return "\x1b[H" + "\n".join(x + "\x1b[K" for x in L) + "\n\x1b[J"


def enter_screen() -> None:
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    sys.stdout.flush()


def leave_screen() -> None:
    sys.stdout.write("\x1b[?25h\x1b[?1049l")
    sys.stdout.flush()


def fmt_pose(v) -> str:
    v = np.asarray(v)
    return f"左[{_vec(v, range(7), '{:.2f}')}]  右[{_vec(v, range(7, 14), '{:.2f}')}]"
