#!/usr/bin/env python3
"""Serve MNISTBoard sealed-test requests from host-owned isolation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ENVIRONMENT = BENCHMARK / "environment"
TESTS = BENCHMARK / "tests"

sys.path.insert(0, str(ENVIRONMENT))
from model_spec import (  # noqa: E402
    MAX_FILES,
    MAX_NAME_LEN,
    MAX_UNCOMPRESSED_BYTES,
    SpecError,
    check_layout,
    gzip_archive,
)

TEST_GATE = 99.4
DOCKER_TIMEOUT_SECONDS = 900
REQUEST_ID_RE = re.compile(r"^[a-f0-9]{16}-[a-f0-9]{8}$")
MAX_REQUEST_BYTES = MAX_UNCOMPRESSED_BYTES + 1024 * 1024
WATCH_PID = 0


class OracleError(RuntimeError):
    """A request cannot be evaluated safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_json_at(directory_fd: int, name: str, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    temporary = f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    file_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(file_fd, view) :]
    finally:
        os.close(file_fd)
    try:
        os.rename(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except Exception:
        os.unlink(temporary, dir_fd=directory_fd)
        raise


def public_result(record: dict) -> dict:
    if record["status"] == "ok":
        return {"test_accuracy": float(record["test_accuracy"])}
    return {"error": "oracle_evaluation_failed"}


def extract_snapshot(archive: bytes, destination: Path) -> str:
    if len(archive) > MAX_REQUEST_BYTES:
        raise OracleError("compressed request exceeds the size limit")

    digest = hashlib.sha256(archive).hexdigest()
    if destination.is_dir():
        if gzip_archive(destination) != archive:
            raise OracleError("snapshot hash collision")
        return digest

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as payload:
            for member in payload:
                name = member.name
                pure = PurePosixPath(name)
                if (
                    not member.isreg()
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.as_posix() != name
                    or name in seen
                ):
                    raise OracleError(f"unsafe archive member: {name!r}")
                if len(name.encode()) > MAX_NAME_LEN:
                    raise OracleError(f"archive path exceeds {MAX_NAME_LEN} bytes")
                if len(seen) >= MAX_FILES:
                    raise OracleError(f"archive contains more than {MAX_FILES} files")
                total += member.size
                if member.size < 0 or total > MAX_UNCOMPRESSED_BYTES:
                    raise OracleError("archive exceeds the uncompressed size limit")
                source = payload.extractfile(member)
                if source is None:
                    raise OracleError(f"cannot read archive member: {name!r}")
                data = source.read(member.size + 1)
                if len(data) != member.size:
                    raise OracleError(f"truncated archive member: {name!r}")
                target = temporary.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                seen.add(name)
        check_layout(temporary)
        if gzip_archive(temporary) != archive:
            raise OracleError("request is not a canonical submission archive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except (OSError, SpecError, tarfile.TarError) as exc:
        raise OracleError(str(exc)) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return digest


def run_container(
    name: str,
    image: str,
    options: list[str],
    command_args: list[str],
    work: Path,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    stdout_path = work / "stdout.log"
    stderr_path = work / "stderr.log"
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        "all",
        "--name",
        name,
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--pids-limit",
        "512",
        "--cpus",
        "4",
        "--memory",
        "8g",
        "--memory-swap",
        "8g",
        "--ulimit",
        f"fsize={MAX_UNCOMPRESSED_BYTES}:{MAX_UNCOMPRESSED_BYTES}",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
        "-e",
        "HOME=/tmp",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "MNISTBOARD_GPU_MEMORY_LIMIT_MIB=1000",
        "-e",
        "MNISTBOARD_GPU_ALLOCATOR_MIB=400",
        *options,
        "--entrypoint",
        "python",
        image,
        *command_args,
    ]
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        deadline = time.monotonic() + DOCKER_TIMEOUT_SECONDS
        failure = None
        while process.poll() is None:
            if time.monotonic() >= deadline:
                failure = "isolated evaluation timed out"
                break
            if WATCH_PID and not pid_alive(WATCH_PID):
                failure = "campaign ended during isolated evaluation"
                break
            time.sleep(0.5)
        if failure:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise OracleError(failure)
        returncode = process.returncode
    if returncode:
        raise OracleError(
            f"isolated evaluation failed with exit code {returncode}"
        )


def prepare_test_bundle(private: Path) -> Path:
    bundle = private / "sealed-tests"
    if bundle.is_dir():
        return bundle
    required = ("mnist_test.npz", "model_spec.py", "run_infer.py", "verify_mnist.py")
    missing = [name for name in required if not (TESTS / name).is_file()]
    if missing:
        raise OracleError(f"missing sealed-test files: {', '.join(missing)}")
    temporary = bundle.with_name(f".{bundle.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    for name in required:
        shutil.copyfile(TESTS / name, temporary / name)
    os.replace(temporary, bundle)
    return bundle


def evaluate(snapshot: Path, archive_size: int, image: str, private: Path) -> dict:
    digest = snapshot.parent.name
    cache_path = private / "cache" / f"{digest}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text())

    record = {
        "candidate_sha256": digest,
        "model_bytes": archive_size,
        "observed_at": utc_now(),
    }
    bundle = prepare_test_bundle(private)
    work = private / "evaluations" / f"{digest}-{uuid.uuid4().hex[:8]}"
    verifier_output = work / "verifier"
    verifier_output.mkdir(parents=True)
    test = work / "test"
    test_name = f"mnist-oracle-{digest[:12]}-{os.getpid()}-test"
    try:
        run_container(
            test_name,
            image,
            [
                "-e",
                "MNISTBOARD_SUBMISSION_DIR=/app/submission",
                "-e",
                "MNISTBOARD_TEST_NPZ=/tests/mnist_test.npz",
                "-e",
                "MNISTBOARD_REWARD_PATH=/logs/verifier/reward.json",
                "-e",
                f"MNISTBOARD_REWARD_UID={os.getuid()}",
                "-e",
                "MNISTBOARD_ORACLE_MODE=1",
                "--tmpfs",
                "/data:rw,nosuid,nodev,noexec,size=1m",
                "--mount",
                f"type=bind,src={snapshot},dst=/app/submission,readonly",
                "--mount",
                f"type=bind,src={bundle},dst=/tests",
                "--mount",
                f"type=bind,src={verifier_output},dst=/logs/verifier",
            ],
            ["/tests/verify_mnist.py"],
            test,
        )
        reward_path = verifier_output / "reward.json"
        if not reward_path.is_file():
            raise OracleError("sealed verifier produced no result")
        sealed = json.loads(reward_path.read_text())
        if int(sealed.get("model_bytes", -1)) != archive_size:
            raise OracleError("sealed verifier used a different submission")
        if int(sealed.get("total", -1)) != 10000:
            raise OracleError("sealed verifier did not use all 10,000 images")
        if float(sealed.get("flag_isolation_failed", 0)) != 0.0:
            raise OracleError("sealed verifier isolation failed")
        if float(sealed.get("verifier_error", 0)) != 0.0:
            raise OracleError("sealed verifier did not complete")

        test_accuracy = float(sealed.get("accuracy", 0.0))
        guardrails_passed = float(sealed.get("guardrails_passed", 0)) == 1.0
        record.update(
            status="ok",
            test_accuracy=test_accuracy,
            passed_99_4=guardrails_passed and test_accuracy >= TEST_GATE,
        )
        atomic_json(cache_path, record)
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)


def append_audit(private: Path, request_id: str, record: dict) -> None:
    row = {"request_id": request_id, **record, "recorded_at": utc_now()}
    audit = private / "audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")


def validate_request_path(ready: Path, agent_dir: Path) -> Path:
    request = ready.parent
    if not REQUEST_ID_RE.fullmatch(request.name):
        raise OracleError("invalid request id")
    queue = agent_dir / "test_oracle" / "requests"
    if (
        request.parent != queue
        or not stat.S_ISREG(ready.lstat().st_mode)
        or not stat.S_ISDIR(request.lstat().st_mode)
        or request.is_symlink()
        or queue.is_symlink()
        or queue.parent.is_symlink()
    ):
        raise OracleError("invalid request path")
    return request


def open_request(request: Path) -> tuple[int, bytes, str]:
    directory_fd = os.open(
        request,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        payload_fd = os.open(
            "submission.tar.gz",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            info = os.fstat(payload_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_REQUEST_BYTES:
                raise OracleError("request payload is invalid or too large")
            chunks = []
            remaining = info.st_size
            while remaining:
                chunk = os.read(payload_fd, min(1024 * 1024, remaining))
                if not chunk:
                    raise OracleError("request payload changed while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(payload_fd, 1):
                raise OracleError("request payload changed while reading")
        finally:
            os.close(payload_fd)
        archive = b"".join(chunks)
        digest = hashlib.sha256(archive).hexdigest()
        if request.name.split("-", 1)[0] != digest[:16]:
            raise OracleError("request id does not match the submission")
        return directory_fd, archive, digest
    except Exception:
        os.close(directory_fd)
        raise


def process_request(
    ready: Path,
    agent_dir: Path,
    image: str,
    private: Path,
    global_lock: Path,
) -> None:
    request = validate_request_path(ready, agent_dir)
    completed = private / "requests" / f"{request.name}.json"
    directory_fd, archive, digest = open_request(request)
    snapshot = private / "snapshots" / digest / "submission"
    try:
        if completed.is_file():
            saved = json.loads(completed.read_text())
            if saved.get("candidate_sha256") != digest:
                raise OracleError("completed request payload changed")
            atomic_json_at(directory_fd, "result.json", saved["public"])
            return

        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with global_lock.open("a+") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if WATCH_PID and not pid_alive(WATCH_PID):
                        raise OracleError("campaign ended while waiting to evaluate")
                    time.sleep(0.5)
            cache_path = private / "cache" / f"{digest}.json"
            if cache_path.is_file():
                record = json.loads(cache_path.read_text())
            else:
                extract_snapshot(archive, snapshot)
                try:
                    record = evaluate(snapshot, len(archive), image, private)
                finally:
                    shutil.rmtree(snapshot.parent, ignore_errors=True)

        public = public_result(record)
        append_audit(private, request.name, record)
        atomic_json(
            completed,
            {"candidate_sha256": digest, "public": public},
        )
        atomic_json_at(directory_fd, "result.json", public)
    finally:
        os.close(directory_fd)


def live_agent_mounts(run_root: Path) -> list[tuple[Path, str, Path]]:
    containers = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not containers:
        return []
    inspection = subprocess.run(
        ["docker", "inspect", *containers],
        capture_output=True,
        text=True,
    )
    if inspection.returncode == 0:
        inspected = json.loads(inspection.stdout)
    else:
        inspected = []
        for container_id in containers:
            result = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                inspected.extend(json.loads(result.stdout))
    mounts = []
    for container in inspected:
        name = str(container.get("Name", "")).lstrip("/")
        image = str(container.get("Config", {}).get("Image", ""))
        if not name.startswith("mnistboard__") or "-main-" not in name:
            continue
        for mount in container.get("Mounts", []):
            if mount.get("Destination") != "/logs/agent":
                continue
            agent_dir = Path(mount["Source"]).resolve()
            if not agent_dir.is_relative_to(run_root):
                continue
            private = run_root / ".test-oracle-private" / name
            mounts.append((agent_dir, image, private))
    return mounts


def find_requests(
    run_root: Path,
) -> list[tuple[Path, Path, str, Path]]:
    requests = []
    for agent_dir, image, private in live_agent_mounts(run_root):
        queue = agent_dir / "test_oracle" / "requests"
        if not queue.is_dir() or queue.is_symlink() or queue.parent.is_symlink():
            continue
        for ready in queue.glob("*/READY"):
            request = ready.parent
            if (request / "result.json").is_file():
                continue
            try:
                validate_request_path(ready, agent_dir)
                requests.append((ready, agent_dir, image, private))
            except (OSError, OracleError):
                continue
    return sorted(requests, key=lambda item: str(item[0]))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    global WATCH_PID

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int, default=0)
    args = parser.parse_args()
    WATCH_PID = args.watch_pid

    run_root = args.run_root.resolve()
    global_lock = run_root.parent / ".mnistboard-test-oracle.lock"
    while True:
        if WATCH_PID and not pid_alive(WATCH_PID):
            return 0
        found = False
        try:
            pending = find_requests(run_root)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            traceback.print_exc()
            time.sleep(1.0)
            continue
        for ready, agent_dir, image, private in pending:
            found = True
            try:
                process_request(ready, agent_dir, image, private, global_lock)
            except Exception as exc:
                traceback.print_exc()
                try:
                    request = validate_request_path(ready, agent_dir)
                    public = {"error": "oracle_evaluation_failed"}
                    directory_fd = os.open(
                        request,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    )
                    try:
                        atomic_json_at(directory_fd, "result.json", public)
                    finally:
                        os.close(directory_fd)
                except Exception:
                    traceback.print_exc()
                print(f"request failed: {exc}", file=sys.stderr, flush=True)
        if args.watch_pid and not pid_alive(args.watch_pid) and not found:
            return 0
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
