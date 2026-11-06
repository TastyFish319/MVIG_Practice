"""Reusable local review for finalized I2RT NEDF3-v2 tasks under data/."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import sys
import threading
import time
import webbrowser

TOOL = Path(__file__).resolve().parents[1]
DATA = TOOL / "raw_data"
DEFAULT_CRITERIA = ["是否按当前任务要求完成了操作？", "目标选择、抓取和放置是否正确？",
                    "是否没有掉落或需要人工帮助？", "关键过程是否可见，相机视角是否可用？"]
SKIP = {"_review", "quarantine", "__pycache__"}


def finalized(path):
    return path.is_dir() and bool(re.fullmatch(r"episode_\d+", path.name))


def discover_tasks(data_root):
    root = Path(data_root).resolve()
    tasks = set()
    for path in root.rglob("episode_*"):
        if any(part in SKIP or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if finalized(path) and path.resolve().is_relative_to(root):
            tasks.add(path.parent.resolve())
    return sorted(tasks, key=lambda p: p.relative_to(root).as_posix())


def resolve_task(data_root, selection):
    root = Path(data_root).resolve()
    candidate = (root / selection).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Task must stay inside data/ and name a task directory")
    tasks = discover_tasks(root)
    if candidate in tasks:
        return candidate
    nested = [p for p in tasks if p.is_relative_to(candidate)]
    if len(nested) == 1:
        return nested[0]
    if not nested:
        raise ValueError(f"No finalized Episode task found: {selection}")
    raise ValueError("Multiple tasks found; choose the exact relative directory")


def task_key(relative_path):
    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(relative_path).name).strip("_")[:40] or "task"
    return prefix + "_" + hashlib.sha256(relative_path.encode()).hexdigest()[:10]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_signature(episode):
    episode = Path(episode)
    files = [episode / name for name in ("metadata.json", "episode_stats.json")]
    files.extend(sorted((episode / "data").glob("*.mcap")))
    index = episode / "data/mcap_index.json"
    if index.exists():
        files.append(index)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(episode).as_posix().encode())
        digest.update((sha256(path) if path.is_file() else "MISSING").encode())
    return digest.hexdigest()


def dataset_fingerprint(episodes):
    identities = [(ep["episode"], ep.get("source_signature", ""), ep.get("source_files", [])) for ep in episodes]
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def validate_decisions(report, incoming):
    if not isinstance(incoming, dict):
        raise ValueError("审核记录必须是 JSON 对象")
    if incoming.get("task") != report["dataset"]:
        raise ValueError("审核记录属于其他 task")
    first = next(iter(report.get("episodes", [])), {})
    legacy = next(iter(first.get("source_files", [])), {}).get("sha256")
    if incoming.get("source_fingerprint") not in (report.get("dataset_fingerprint"), legacy):
        raise ValueError("源数据版本发生变化，请重新审核")
    valid = {ep["episode"]: ep for ep in report["episodes"]}
    items = incoming.get("episodes")
    if not isinstance(items, list):
        raise ValueError("审核记录缺少 episodes 列表")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("无效 Episode 记录")
        key = item.get("episode")
        if key not in valid or key in seen:
            raise ValueError("审核记录包含未知或重复 Episode")
        seen.add(key)
        if item.get("decision") not in ("pending", "approved", "rejected", "unsure"):
            raise ValueError("无效审核决定")
        if item.get("source_files") != valid[key].get("source_files"):
            raise ValueError("Episode 文件指纹不匹配")
        if item["decision"] == "approved" and valid[key]["technical_status"] == "fail":
            raise ValueError("技术不通过的 Episode 不能直接保留")
        if not isinstance(item.get("notes", ""), str) or len(item.get("notes", "")) > 10000:
            raise ValueError("备注无效或过长")
    return incoming


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def can_adopt(result, shards):
    """Migration may reuse only the complete, unchanged MCAP inventory."""
    recorded = result.get("source_files", [])
    actual = {path.name: path for path in shards}
    if not recorded or sorted(entry["name"] for entry in recorded) != sorted(actual):
        return False
    return all(sha256(actual[entry["name"]]) == entry["sha256"] for entry in recorded)


def archive_previous_decisions(output, report):
    path, old_report = output / "decisions.json", output / "report.json"
    if not path.exists() or not old_report.exists():
        return
    previous = json.loads(old_report.read_text(encoding="utf-8"))
    old_fingerprint = previous.get("dataset_fingerprint")
    if old_fingerprint == report["dataset_fingerprint"]:
        return
    history = output / "history"
    history.mkdir(exist_ok=True)
    # Preserve old human work; a new dataset version starts a fresh save file.
    destination = history / f"decisions-{old_fingerprint or 'legacy'}-{time.time_ns()}.json"
    path.replace(destination)


def catalog_entries():
    catalog = []
    for path in sorted((TOOL / "tasks").glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        decisions = path.parent / "decisions.json"
        saved = json.loads(decisions.read_text(encoding="utf-8")) if decisions.exists() else {"episodes":[]}
        if decisions.exists():
            validate_decisions(report,saved)
        by_name = {item["episode"]:item for item in saved["episodes"]}
        items = [by_name.get(ep["episode"],{}) for ep in report["episodes"]]
        catalog.append({"id":path.parent.name,"title":report.get("task_title",report["dataset"]),
            "source":report.get("source_directory_alias",""),"episodes":len(report["episodes"]),
            "summary":report["summary"],"reviewed":sum(i.get("decision","pending")!="pending" for i in items),
            "approved":sum(i.get("decision")=="approved" for i in items)})
    return catalog


def update_catalog():
    (TOOL / "tasks").mkdir(exist_ok=True)
    return catalog_entries()


def import_task(source):
    """Copy generated assets and human records; never move or edit their source."""
    import shutil
    source=Path(source).expanduser().resolve()
    report=json.loads((source/"report.json").read_text(encoding="utf-8"))
    task_id=report["dataset"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",task_id):
        raise ValueError("已有任务的标识格式无效")
    if (source/"decisions.json").is_file():
        validate_decisions(report,json.loads((source/"decisions.json").read_text(encoding="utf-8")))
    output=TOOL/"tasks"/task_id
    if output.exists():
        raise FileExistsError("同名审核任务已存在，拒绝覆盖")
    allowed_dirs={"episodes","media","numeric","history"}
    allowed_files={"report.json","report.js","decisions.json","verification.json"}
    files=[p for p in source.rglob("*") if p.is_file() and (p.relative_to(source).parts[0] in allowed_dirs or p.relative_to(source).as_posix() in allowed_files)]
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source) or path.suffix==".mcap":
            raise ValueError("迁移仅接受任务目录内部的审核产物，不接受原始MCAP或外部链接")
    stage=output.with_name(output.name+".importing")
    stage.mkdir(parents=True,exist_ok=False)
    for path in files:
        target=stage/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    report.setdefault("source_root",str(DATA.resolve()))
    write_json(stage/"report.json",report)
    (stage/"report.js").write_text("window.AUDIT_REPORT="+json.dumps(report,ensure_ascii=False).replace("<","\\u003c")+";",encoding="utf-8")
    (stage/"index.html").write_text((TOOL/"frontend/review.html").read_text(encoding="utf-8"),encoding="utf-8")
    stage.rename(output)
    update_catalog()
    return output


def generate(task, task_id, title, criteria, adopt=False):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id):
        raise ValueError("task-id 只能使用 ASCII 字母、数字、下划线或连字符")
    from backend import audit
    from backend import nedf3_reader as reader
    relative = task.relative_to(DATA).as_posix()
    output = TOOL / "tasks" / task_id
    output.mkdir(parents=True, exist_ok=True)
    old_report = output / "report.json"
    if old_report.exists():
        previous = json.loads(old_report.read_text(encoding="utf-8"))
        if previous.get("source_directory_alias") != relative:
            raise ValueError("该 task-id 已用于其他目录，请使用不同的 task-id")
    for folder in ("episodes", "media", "numeric"):
        (output / folder).mkdir(exist_ok=True)
    results = []
    episodes = sorted(p for p in task.iterdir() if finalized(p))
    for number, episode in enumerate(episodes, 1):
        cache = output / "episodes" / f"{episode.name}.json"
        signature = source_signature(episode)
        result = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        reuse = result is not None and result.get("source_signature") == signature
        if adopt and result and not result.get("source_signature") and result.get("source_files"):
            reuse = can_adopt(result, reader.episode_mcap_paths(episode))
        if reuse:
            preview = result.get("preview", {})
            reuse = bool(preview.get("path") and preview.get("poster") and result.get("source_files"))
            if reuse:
                reuse = (output / preview["path"]).is_file() and (output / preview["poster"]).is_file()
        if not reuse:
            try:
                result = audit.inspect(episode, output, reader, task_id=task_id, source_relative=f"{relative}/{episode.name}")
            except Exception as error:
                result = {"episode": episode.name, "technical_status": "fail", "failures": [f"{type(error).__name__}: {error}"],
                          "warnings": [], "human_decision": "pending", "task_success": "not_assessed"}
        result.update(source_relative=f"{relative}/{episode.name}", canonical_task=task_id, source_signature=signature)
        write_json(cache, result)
        results.append(result)
        print(f"{number:02}/{len(episodes)} {episode.name}: {result['technical_status']} {'(cached)' if reuse else ''}", flush=True)
    report = {"schema_version": 2, "dataset": task_id, "task_title": title,
              "review_criteria": criteria or DEFAULT_CRITERIA, "source_directory_alias": relative,
              "dataset_fingerprint": dataset_fingerprint(results), "source_root": str(DATA.resolve()), "episodes": results,
              "summary": dict(Counter(ep["technical_status"] for ep in results)),
              "limitations": ["支持当前 I2RT NEDF3 v2，技术通过不等于任务成功", "回放仅供审核，训练使用原始 MCAP"]}
    archive_previous_decisions(output, report)
    write_json(output / "report.json", report)
    (output / "report.js").write_text("window.AUDIT_REPORT=" + json.dumps(report, ensure_ascii=False).replace("<", "\\u003c") + ";", encoding="utf-8")
    (output / "index.html").write_text((TOOL / "frontend/review.html").read_text(encoding="utf-8"), encoding="utf-8")
    update_catalog()
    return output


class ReviewHandler(SimpleHTTPRequestHandler):
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(TOOL), **kwargs)

    def decision_paths(self):
        match = re.fullmatch(r"/api/decisions/([a-zA-Z0-9_-]+)", self.path)
        if not match:
            raise ValueError("无效的任务地址")
        folder = (TOOL / "tasks" / match.group(1)).resolve()
        if not folder.is_relative_to((TOOL / "tasks").resolve()):
            raise ValueError("无效任务目录")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        return folder / "decisions.json", report

    def json_response(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path):
        from urllib.parse import unquote, urlsplit
        requested = unquote(urlsplit(path).path)
        aliases = {"/":"frontend/index.html", "/index.html":"frontend/index.html",
                   "/app.js":"frontend/app.js", "/styles.css":"frontend/styles.css", "/home.js":"frontend/home.js"}
        return str(TOOL / aliases.get(requested, requested.lstrip("/")))

    def do_GET(self):
        if self.path == "/api/catalog":
            try:
                self.json_response(200,{"tasks":catalog_entries()})
            except (ValueError,OSError,KeyError,TypeError) as error:
                self.json_response(409,{"error":str(error)})
            return
        if self.path == "/api/health":
            self.json_response(200, {"tool":"nedf3-data-review", "instance":hashlib.sha256(str(TOOL).encode()).hexdigest()[:16]})
            return
        if self.path.startswith("/api/decisions/"):
            try:
                path, report = self.decision_paths()
                payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                if payload is not None:
                    validate_decisions(report, payload)
                self.json_response(200, {"saved": payload})
            except (ValueError, OSError, IndexError) as error:
                self.json_response(409, {"error": str(error)})
            return
        # Serve only generated pages/media, never source episodes, dependencies or local settings.
        from urllib.parse import unquote, urlsplit
        requested = unquote(urlsplit(self.path).path)
        if requested == "/catalog.js":
            body = ("window.REVIEW_TASKS=" + json.dumps(catalog_entries(),ensure_ascii=False) + ";").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","text/javascript; charset=utf-8")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(body)
            return
        static = {"/", "/index.html", "/app.js", "/styles.css", "/home.js",
                  "/frontend/index.html", "/frontend/review.html", "/frontend/app.js", "/frontend/home.js", "/frontend/styles.css"}
        if requested not in static and not requested.startswith("/tasks/"):
            self.send_error(404)
            return
        resolved = Path(self.translate_path(self.path)).resolve()
        if not resolved.is_relative_to(TOOL.resolve()) or any(part.startswith(".") for part in Path(requested).parts):
            self.send_error(403)
            return
        if requested.startswith("/tasks/") and requested.endswith("/index.html") and resolved.is_file():
            body = (TOOL / "frontend/review.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin") != origin or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json_response(403, {"error": "仅接受本机审核页的同源 JSON 请求"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("审核记录为空或过大")
            payload = json.loads(self.rfile.read(length))
            with self.lock:
                path, report = self.decision_paths()
                validate_decisions(report, payload)
                # A late save from another tab must not undo newer per-Episode decisions.
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    validate_decisions(report, previous)
                    merged = {entry["episode"]: entry for entry in previous["episodes"]}
                    for entry in payload["episodes"]:
                        old = merged.get(entry["episode"], {})
                        if entry.get("updated_at", "") >= old.get("updated_at", ""):
                            merged[entry["episode"]] = entry
                    payload["episodes"] = list(merged.values())
                write_json(path, payload)
            self.json_response(200, {"ok": True})
        except (ValueError, OSError, IndexError, TypeError) as error:
            self.json_response(400, {"error": str(error)})


class ReviewServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main(argv=None):
    global DATA
    settings_path = TOOL / "review.local.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    if not isinstance(settings,dict) or set(settings) - {"data_root","port"}:
        raise ValueError("review.local.json仅支持data_root与port")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-task", type=Path, help="复制旧工具中已生成的单个任务，保留审核记录和回放")
    parser.add_argument("--data-root", type=Path, help="原始数据根目录，内含各task；不修改源数据")
    parser.add_argument("--save-config", action="store_true", help="保存当前数据路径与端口到review.local.json")
    parser.add_argument("--list", action="store_true", help="列出可审核的任务目录")
    parser.add_argument("--choose", action="store_true", help="交互选择任务并生成")
    parser.add_argument("--task", help="相对于 data 的任务目录，可自动识别单层嵌套")
    parser.add_argument("--task-id", help="独立审核任务标识；默认由源路径生成")
    parser.add_argument("--title", help="任务真实名称，不必与原始目录同名")
    parser.add_argument("--criterion", action="append", help="审核标准，可重复指定")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true", help="核对原始 MCAP 指纹后接纳已有审核产物")
    parser.add_argument("--port", type=int, default=settings.get("port",8776))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在1~65535之间")
    source = (args.data_root or Path(settings.get("data_root", "raw_data"))).expanduser()
    DATA = (TOOL / source).resolve()
    if args.save_config:
        write_json(settings_path, {"data_root":str(DATA),"port":args.port})
    if args.import_task:
        imported=import_task(args.import_task)
        print(f"已复制已有审核任务：{imported}",flush=True)
    if args.list:
        for task in discover_tasks(DATA):
            print(f"{task.relative_to(DATA).as_posix()}\t{sum(finalized(p) for p in task.iterdir())} episodes")
        return
    if args.choose:
        tasks = discover_tasks(DATA)
        if not tasks:
            raise ValueError(f"{DATA} 下没有可审核的episode，请用--data-root指定原始数据根目录")
        for number, task in enumerate(tasks, 1):
            print(f"{number}. {task.relative_to(DATA).as_posix()}")
        choice = int(input("选择任务序号：")) - 1
        if not 0 <= choice < len(tasks):
            raise ValueError("无效任务序号")
        args.task = tasks[choice].relative_to(DATA).as_posix()
        # Reuse the same destination for a known source, including custom display names.
        for path in (TOOL / "tasks").glob("*/report.json"):
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("source_directory_alias") == args.task:
                args.task_id = previous["dataset"]
                args.title = previous.get("task_title")
                if not args.criterion:
                    args.criterion = previous.get("review_criteria")
                break
        entered_title = input(f"真实任务名称（回车使用 {args.title or tasks[choice].name}）：").strip()
        args.title = entered_title or args.title or tasks[choice].name
    if args.task:
        task = resolve_task(DATA, args.task)
        relative = task.relative_to(DATA).as_posix()
        generate(task, args.task_id or task_key(relative), args.title or task.name, args.criterion, args.adopt_existing)
    if args.generate_only:
        return
    update_catalog()
    try:
        server = ReviewServer(("127.0.0.1", args.port), ReviewHandler)
    except OSError:
        from urllib.request import urlopen
        url = f"http://127.0.0.1:{args.port}/"
        try:
            with urlopen(url + "api/health", timeout=2) as response:
                health = json.load(response)
                is_review = health.get("instance") == hashlib.sha256(str(TOOL).encode()).hexdigest()[:16]
        except OSError:
            is_review = False
        if not is_review:
            raise SystemExit("端口已被其他程序占用，请用 --port 选择其他端口")
        print(f"审核服务已运行：{url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"审核工作台：{url}\n仅在本机运行，Ctrl+C 停止；审核记录自动保存到各 task 的 decisions.json。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError,OSError,ImportError) as error:
        raise SystemExit(f"启动失败：{error}\n依赖缺失时请在当前Python环境安装requirements.txt。") from error
