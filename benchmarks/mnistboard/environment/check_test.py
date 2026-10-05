#!/usr/bin/env python3
"""Request sealed-test accuracy for the current submission."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(os.environ.get("MNISTBOARD_APP_DIR", "/app"))
SUBMISSION_DIR = Path(
    os.environ.get("MNISTBOARD_SUBMISSION_DIR", str(APP_DIR / "submission"))
)
ORACLE_DIR = Path(
    os.environ.get("MNISTBOARD_TEST_ORACLE_DIR", "/logs/agent/test_oracle")
)
WAIT_SECONDS = int(os.environ.get("MNISTBOARD_TEST_ORACLE_TIMEOUT_SEC", "21600"))
MIN_VALIDATION_ACCURACY = 99.4

sys.path.insert(0, str(APP_DIR))
from model_spec import SpecError, check_layout, gzip_archive  # noqa: E402


def full_validation_result() -> dict:
    environment = {
        **os.environ,
        "MNISTBOARD_SUBMISSION_DIR": str(SUBMISSION_DIR),
    }
    result = subprocess.run(
        [sys.executable, str(APP_DIR / "check_submission.py"), "--json"],
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"full validation failed: {result.stderr[-1000:]}"
        )
    try:
        start = result.stdout.index("{")
        return json.loads(result.stdout[start : result.stdout.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("full validation returned invalid JSON") from exc


def eligible_for_test(result: dict) -> bool:
    return (
        int(result.get("val_images") or 0) == 5000
        and float(result.get("val_accuracy") or 0) >= MIN_VALIDATION_ACCURACY
        and result.get("qualified_on_val") is True
    )


def create_request(submission_dir: Path, oracle_dir: Path) -> Path:
    check_layout(submission_dir)
    archive = gzip_archive(submission_dir)
    digest = hashlib.sha256(archive).hexdigest()
    request_id = f"{digest[:16]}-{secrets.token_hex(4)}"
    requests = oracle_dir / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    partial = requests / f".{request_id}.tmp"
    request = requests / request_id
    partial.mkdir()
    (partial / "submission.tar.gz").write_bytes(archive)
    os.replace(partial, request)
    os.chmod(request, 0o777)
    (request / "READY").write_text("\n")
    return request


def main() -> int:
    try:
        validation = full_validation_result()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not eligible_for_test(validation):
        print(
            "test request rejected: full validation must reach "
            f"{MIN_VALIDATION_ACCURACY:.1f}% "
            f"(got {float(validation.get('val_accuracy') or 0):.4f}%)",
            file=sys.stderr,
        )
        return 1

    try:
        request = create_request(SUBMISSION_DIR, ORACLE_DIR)
    except (OSError, SpecError) as exc:
        print(f"test request rejected: {exc}", file=sys.stderr)
        return 1

    result_path = request / "result.json"
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            result = json.loads(result_path.read_text())
        except FileNotFoundError:
            time.sleep(1)
            continue
        except (OSError, json.JSONDecodeError) as exc:
            print(f"invalid oracle response: {exc}", file=sys.stderr)
            return 2
        for name in ("READY", "submission.tar.gz"):
            try:
                (request / name).unlink()
            except FileNotFoundError:
                pass
        print(json.dumps(result, sort_keys=True))
        return 0 if "test_accuracy" in result else 1

    print("test oracle timed out", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
