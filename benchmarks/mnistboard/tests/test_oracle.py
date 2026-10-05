#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ENVIRONMENT = BENCHMARK / "environment"
TOOLS = BENCHMARK / "tools"

sys.path.insert(0, str(ENVIRONMENT))
from model_spec import gzip_archive  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load_module("mnistboard_check_test", ENVIRONMENT / "check_test.py")
worker = load_module("mnistboard_test_oracle_worker", TOOLS / "test_oracle_worker.py")


class OracleTest(unittest.TestCase):
    def make_submission(self, root: Path) -> Path:
        submission = root / "submission"
        submission.mkdir()
        (submission / "predict.py").write_text(
            "def predict(model, images):\n"
            "    return [0] * len(images)\n"
        )
        (submission / "weights.bin").write_bytes(b"\x00\x01\x02")
        return submission

    def test_client_queues_canonical_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = self.make_submission(root)
            request = client.create_request(submission, root / "oracle")
            self.assertTrue((request / "READY").is_file())
            self.assertEqual(request.stat().st_mode & 0o777, 0o777)
            self.assertEqual(
                (request / "submission.tar.gz").read_bytes(),
                gzip_archive(submission),
            )

    def test_test_request_eligibility_uses_99_4_full_validation(self) -> None:
        self.assertTrue(
            client.eligible_for_test(
                {
                    "val_accuracy": 99.4,
                    "val_images": 5000,
                    "qualified_on_val": True,
                }
            )
        )
        self.assertFalse(
            client.eligible_for_test(
                {
                    "val_accuracy": 99.3999,
                    "val_images": 5000,
                    "qualified_on_val": False,
                }
            )
        )
        self.assertFalse(
            client.eligible_for_test(
                {
                    "val_accuracy": 99.9,
                    "val_images": 2000,
                    "qualified_on_val": True,
                }
            )
        )

    def test_worker_extracts_canonical_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = self.make_submission(root)
            archive = gzip_archive(submission)
            destination = root / "private" / "submission"
            worker.extract_snapshot(archive, destination)
            self.assertEqual(gzip_archive(destination), archive)

    def test_worker_binds_request_id_to_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = self.make_submission(root)
            request = client.create_request(submission, root / "oracle")
            renamed = request.with_name(f"{'0' * 16}-{request.name[-8:]}")
            request.rename(renamed)
            with self.assertRaises(worker.OracleError):
                worker.open_request(renamed)

    def test_worker_rejects_path_traversal(self) -> None:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        malicious = gzip.compress(payload.getvalue(), compresslevel=9, mtime=0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(worker.OracleError):
                worker.extract_snapshot(malicious, Path(tmp) / "submission")

    def test_public_result_contains_test_accuracy(self) -> None:
        self.assertEqual(
            worker.public_result(
                {
                    "status": "ok",
                    "val_accuracy": 99.7,
                    "test_accuracy": 99.5,
                    "passed_99_4": True,
                }
            ),
            {"test_accuracy": 99.5},
        )
        self.assertEqual(
            worker.public_result(
                {"status": "ok", "test_accuracy": 0.0, "passed_99_4": False}
            ),
            {"test_accuracy": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
