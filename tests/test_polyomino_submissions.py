from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(data):
    result = Mock(status_code=200)
    result.json.return_value = data
    return result


class SubmissionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))
        self.submit = load_module(
            "polyomino_submit", "benchmarks/polyomino/environment/submit.py",
        )
        self.verifier = load_module(
            "polyomino_evaluate", "benchmarks/polyomino/tests/evaluate.py",
        )
        self.log = self.root / "agent/submissions.jsonl"
        self.source = self.root / "candidate.cpp"
        self.source.write_bytes(b"int main() { return 0; }\n")
        verifier_dir = self.root / "verifier"
        self.enterContext(patch.object(self.submit, "SUBMISSIONS_LOG", self.log))
        self.enterContext(patch.object(sys, "argv", ["submit.py", str(self.source)]))
        self.enterContext(patch.multiple(
            self.verifier,
            SOLUTION_PATH=self.source,
            AGENT_SUBMISSIONS_LOG=self.log,
            VERIFIER_SUBMISSIONS_LOG=verifier_dir / "submissions.jsonl",
            REWARD_TXT=verifier_dir / "reward.txt",
            REWARD_JSON=verifier_dir / "reward.json",
            JUDGE_RESULT_JSON=verifier_dir / "judge_result.json",
            FINAL_SOURCE_ARCHIVE=verifier_dir / "final_solution.cpp",
        ))

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_exact_snapshot_before_request_and_repeated_path(self):
        snapshots = [b"// caf\xc3\xa9\r\nint main() { return 0; }\r\n", b"int main() {}\n"]
        for source in snapshots:
            self.source.write_bytes(source)

            def post(*args, **kwargs):
                archived = self.records()[-1]
                self.assertEqual(archived["status"], "archived")
                path = self.log.parent / archived["source_archive"]
                self.assertEqual(path.read_bytes(), source)
                self.assertEqual(kwargs["files"]["code"][1], source)
                self.source.write_bytes(b"changed while request was in flight")
                return response({"sid": "test-sid"})

            with patch.object(self.submit.requests, "post", side_effect=post), \
                 patch.object(self.submit.requests, "get", return_value=response({
                     "status": "done", "score": 65,
                 })):
                self.assertEqual(self.submit.main(), 0)

        archived = [r for r in self.records() if r["status"] == "archived"]
        self.assertEqual(len({r["submission_uuid"] for r in archived}), 2)
        for record, source in zip(archived, snapshots):
            self.assertEqual((self.log.parent / record["source_archive"]).read_bytes(), source)
            self.assertEqual(record["source_sha256"], hashlib.sha256(source).hexdigest())
            done = next(r for r in self.records()
                        if r["status"] == "done"
                        and r["submission_uuid"] == record["submission_uuid"])
            self.assertEqual(done["source_archive"], record["source_archive"])
            self.assertEqual(done["score"], 0.65)

    def test_failed_request_keeps_archive(self):
        source = self.source.read_bytes()
        with patch.object(
            self.submit.requests, "post", side_effect=requests.ConnectionError("offline"),
        ):
            self.assertEqual(self.submit.main(), 3)
        record = self.records()[-1]
        self.assertEqual(record["status"], "error")
        self.assertEqual((self.log.parent / record["source_archive"]).read_bytes(), source)

    def test_poll_timeout_keeps_archive(self):
        with patch.object(self.submit, "MAX_POLL_TIME", 0), \
             patch.object(self.submit.requests, "post", return_value=response({"sid": "test-sid"})):
            self.assertEqual(self.submit.main(), 4)
        record = self.records()[-1]
        self.assertEqual(record["status"], "error")
        self.assertTrue((self.log.parent / record["source_archive"]).is_file())

    def test_archive_failure_prevents_submission(self):
        with patch.object(Path, "write_bytes", side_effect=PermissionError("read-only")), \
             patch.object(self.submit.requests, "post") as post:
            self.assertEqual(self.submit.main(), 5)
        post.assert_not_called()
        self.assertIn("Source archiving failed", self.records()[-1]["error"])

    def test_missing_and_empty_sources_are_not_submitted(self):
        for source in (None, b" \n"):
            with self.subTest(source=source):
                if source is None:
                    self.source.unlink()
                else:
                    self.source.write_bytes(source)
                with patch.object(self.submit.requests, "post") as post:
                    self.assertEqual(self.submit.main(), 2)
                post.assert_not_called()
        self.assertFalse((self.log.parent / "submitted_sources").exists())

    def test_final_source_is_archived_before_submission(self):
        source = b"// final\r\nint main() {}\r\n"
        self.source.write_bytes(source)

        def post(*args, **kwargs):
            self.assertEqual(self.verifier.FINAL_SOURCE_ARCHIVE.read_bytes(), source)
            self.assertEqual(kwargs["files"]["code"][1], source)
            return response({"sid": "final-sid"})

        with patch.object(self.verifier.requests, "post", side_effect=post), \
             patch.object(self.verifier.requests, "get", side_effect=[
                 response({}), response({"status": "done", "score": 42}),
             ]):
            self.verifier.main()
        reward = json.loads(self.verifier.REWARD_JSON.read_text())
        self.assertEqual(reward["reward"], 0.42)

    def test_missing_final_source_mirrors_archives_and_preserves_best_reward(self):
        with patch.object(self.submit.requests, "post", return_value=response({"sid": "test-sid"})), \
             patch.object(self.submit.requests, "get", return_value=response({
                 "status": "done", "score": 65,
             })):
            self.assertEqual(self.submit.main(), 0)
        self.source.unlink()
        with patch.object(self.verifier.requests, "post") as post:
            self.verifier.main()
        post.assert_not_called()
        self.assertEqual(self.verifier.VERIFIER_SUBMISSIONS_LOG.read_bytes(), self.log.read_bytes())
        record = self.records()[-1]
        mirrored = self.verifier.VERIFIER_SUBMISSIONS_LOG.parent / record["source_archive"]
        self.assertEqual(mirrored.read_bytes(), (self.log.parent / record["source_archive"]).read_bytes())
        self.assertEqual(json.loads(self.verifier.REWARD_JSON.read_text())["reward"], 0.65)


if __name__ == "__main__":
    unittest.main()
