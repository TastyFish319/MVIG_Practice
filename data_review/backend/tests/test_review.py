"""Contract checks for task isolation and source-safe reusable review."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


class ReviewContracts(unittest.TestCase):
    def module(self):
        path = ROOT / "backend/server.py"
        self.assertTrue(path.exists(), "Reusable review entrypoint must exist")
        spec = importlib.util.spec_from_file_location("review_tool", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_discovers_nested_tasks_and_ignores_quarantine(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for name in ("a/a/episode_0000", "b/episode_0001", "b/quarantine/episode_0002", "_review/episode_0003"):
                (data / name).mkdir(parents=True)
            self.assertEqual([p.relative_to(data.resolve()).as_posix() for p in module.discover_tasks(data)], ["a/a", "b"])

    def test_source_cannot_escape_data_root(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            with self.assertRaises(ValueError):
                module.resolve_task(root, "../other")

    def test_task_id_distinguishes_equal_folder_names(self):
        module = self.module()
        self.assertNotEqual(module.task_key("one/demo"), module.task_key("two/demo"))
        self.assertRegex(module.task_key("中文任务"), r"^[a-zA-Z0-9_-]+$")

    def test_cached_audit_invalidated_by_source_change(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "metadata.json").write_text("{}")
            (episode / "episode_stats.json").write_text("{}")
            (episode / "data").mkdir()
            (episode / "data" / "a.mcap").write_bytes(b"original")
            stamp = module.source_signature(episode)
            (episode / "data" / "a.mcap").write_bytes(b"modified")
            self.assertNotEqual(stamp, module.source_signature(episode))

    def test_import_rejects_wrong_dataset_and_fail_approval(self):
        module = self.module()
        report = {"dataset": "a", "dataset_fingerprint": "hash-a", "episodes": [
            {"episode": "episode_0000", "technical_status": "fail", "source_files": [{"sha256": "same"}]}]}
        incoming = {"task": "b", "source_fingerprint": "hash-b", "episodes": []}
        with self.assertRaises(ValueError):
            module.validate_decisions(report, incoming)

    def test_technical_fail_cannot_be_approved(self):
        module = self.module()
        report = {"dataset": "a", "dataset_fingerprint": "hash-a", "episodes": [
            {"episode": "episode_0000", "technical_status": "fail", "source_files": [{"sha256": "same"}]}]}
        incoming = {"task": "a", "source_fingerprint": "hash-a", "episodes": [
            {"episode": "episode_0000", "decision": "approved", "source_files": [{"sha256": "same"}]}]}
        with self.assertRaisesRegex(ValueError, "技术不通过"):
            module.validate_decisions(report, incoming)

    def test_adopt_rejects_added_or_missing_shard(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory) / "one.mcap", Path(directory) / "two.mcap"
            one.write_bytes(b"one")
            two.write_bytes(b"two")
            result = {"source_files": [{"name": one.name, "sha256": module.sha256(one)}]}
            self.assertTrue(module.can_adopt(result, [one]))
            self.assertFalse(module.can_adopt(result, [one, two]))
            self.assertFalse(module.can_adopt(result, [two]))

    def test_changed_dataset_archives_decisions_without_losing_them(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            module.write_json(folder / "report.json", {"dataset_fingerprint": "before"})
            original = {"task": "a", "episodes": [{"decision": "approved"}]}
            module.write_json(folder / "decisions.json", original)
            module.archive_previous_decisions(folder, {"dataset_fingerprint": "after"})
            self.assertFalse((folder / "decisions.json").exists())
            archived = list((folder / "history").glob("*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(json.loads(archived[0].read_text()), original)
    def test_http_saves_locally_and_rejects_cross_origin(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            module.TOOL = Path(directory)
            folder = module.TOOL / "tasks" / "demo"
            folder.mkdir(parents=True)
            report = {"dataset": "demo", "dataset_fingerprint": "dataset-hash", "episodes": [
                {"episode": "episode_0000", "technical_status": "pass", "source_files": [{"sha256": "file-hash"}]}]}
            module.write_json(folder / "report.json", report)
            payload = {"task": "demo", "source_fingerprint": "dataset-hash", "episodes": [
                {"episode": "episode_0000", "decision": "unsure", "notes": "review test", "updated_at": "2026-10-25T12:00:00Z", "source_files": [{"sha256": "file-hash"}]}]}
            server = ThreadingHTTPServer(("127.0.0.1", 0), module.ReviewHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            url = origin + "/api/decisions/demo"
            try:
                request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Origin": origin})
                with urlopen(request) as response:
                    self.assertEqual(response.status, 200)
                self.assertTrue((folder / "decisions.json").exists())
                with urlopen(url) as response:
                    saved = json.load(response)["saved"]
                self.assertEqual(saved["episodes"][0]["decision"], "unsure")
                # Older browser tabs cannot undo a newer decision.
                payload["episodes"][0].update(decision="pending", updated_at="2026-10-25T11:00:00Z")
                request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Origin": origin})
                with urlopen(request):
                    pass
                self.assertEqual(json.loads((folder / "decisions.json").read_text())["episodes"][0]["decision"], "unsure")
                request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Origin": "http://other.example"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request)
                self.assertEqual(raised.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_independent_reader_rejects_missing_index_shard(self):
        from backend import nedf3_reader
        with tempfile.TemporaryDirectory() as directory:
            episode=Path(directory);(episode/"data").mkdir();(episode/"metadata.json").write_text("{}")
            (episode/"data/a.mcap").touch();(episode/"data/b.mcap").touch()
            (episode/"data/mcap_index.json").write_text('[{"filename":"a.mcap"}]')
            with self.assertRaises(ValueError):nedf3_reader.episode_mcap_paths(episode)

    def test_import_task_keeps_original_record_and_refuses_overwrite(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/"old/demo";source.mkdir(parents=True)
            module.TOOL=base/"new";(module.TOOL/"frontend").mkdir(parents=True)
            (module.TOOL/"frontend/review.html").write_text("new template")
            report={"dataset":"demo","dataset_fingerprint":"fingerprint","summary":{"pass":1},"episodes":[
                {"episode":"episode_0000","technical_status":"pass","source_files":[{"sha256":"filehash"}]}]}
            decisions={"task":"demo","source_fingerprint":"fingerprint","episodes":[
                {"episode":"episode_0000","decision":"approved","notes":"keep me","source_files":[{"sha256":"filehash"}]}]}
            module.write_json(source/"report.json",report);module.write_json(source/"decisions.json",decisions)
            original=(source/"decisions.json").read_bytes()
            output=module.import_task(source)
            self.assertEqual((source/"decisions.json").read_bytes(),original)
            self.assertEqual((output/"decisions.json").read_bytes(),original)
            self.assertEqual((output/"index.html").read_text(),"new template")
            self.assertEqual(module.catalog_entries()[0]["approved"],1)
            with self.assertRaises(FileExistsError):module.import_task(source)

    def test_http_new_assets_template_and_source_exclusion(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            module.TOOL=Path(directory);(module.TOOL/"frontend").mkdir();(module.TOOL/"tasks/demo").mkdir(parents=True)
            (module.TOOL/"frontend/review.html").write_text("fresh review template")
            (module.TOOL/"tasks/demo/index.html").write_text("old template")
            (module.TOOL/"frontend/styles.css").write_text("body{color:green}")
            (module.TOOL/"review.py").write_text("private source")
            server=ThreadingHTTPServer(("127.0.0.1",0),module.ReviewHandler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            origin=f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(origin+"/tasks/demo/index.html") as response:self.assertEqual(response.read(),b"fresh review template")
                with urlopen(origin+"/styles.css") as response:self.assertEqual(response.status,200)
                with urlopen(origin+"/frontend/styles.css") as response:self.assertEqual(response.status,200)
                with self.assertRaises(HTTPError):urlopen(origin+"/backend/server.py")
                with urlopen(origin+"/api/health") as response:self.assertEqual(json.load(response)["tool"],"nedf3-data-review")
                with self.assertRaises(HTTPError):urlopen(origin+"/review.py")
            finally:
                server.shutdown();server.server_close();thread.join()

if __name__ == "__main__":
    unittest.main()
