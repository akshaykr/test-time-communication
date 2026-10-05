#!/usr/bin/env python3
"""Trusted grader for mnistboard.

Objective: the smallest SELF-CONTAINED solver -- everything shipped in the
submission directory -- that reaches >= 99.4% accuracy on the hidden MNIST test
set. Size is the deterministic gzip byte count of the whole submission:

    model_bytes = len( gzip -9 of a normalised tar of every file in the dir )

Code and data are both counted, so the byte is charged wherever it lives.
Inference runs over unseen images in a separate, unprivileged process from a
root-owned read-only snapshot, with /tests and /data made unreadable to it, over
a privately shuffled test set. The shared GPU is visible because the benchmark
scores compressed size rather than inference hardware.

Harbor requires a numeric ordering field. It is zero when unqualified and the
reciprocal of `model_bytes` when qualified, so it adds no target or shaping.
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_spec import (  # noqa: E402
    SpecError,
    check_layout,
    format_report,
    gzip_size,
)

SUBMISSION_DIR = Path(os.environ.get("MNISTBOARD_SUBMISSION_DIR", "/app/submission"))
GRADED_DIR = Path("/tmp/mnistboard_graded")
EVAL_DIR = Path("/tmp/mnistboard_eval")
TEST_NPZ = Path(os.environ.get("MNISTBOARD_TEST_NPZ", "/tests/mnist_test.npz"))
RUN_INFER = Path(__file__).resolve().parent / "run_infer.py"
REWARD_PATH = Path(os.environ.get("MNISTBOARD_REWARD_PATH", "/logs/verifier/reward.json"))
ORACLE_MODE = os.environ.get("MNISTBOARD_ORACLE_MODE") == "1"

QUALIFY_ACCURACY = 99.4
SHUFFLE_SEED = int.from_bytes(os.urandom(4), "big")
INFER_TIMEOUT_SEC = 600
MAX_PREDICTION_BYTES = 1024 * 1024

def framework_score(qualified: bool, model_bytes: int) -> float:
    if not qualified or model_bytes <= 0:
        return 0.0
    return 1.0 / model_bytes


def force_rmtree(path: Path) -> None:
    if not path.exists():
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=True):
        try:
            os.chmod(dirpath, 0o700)
        except OSError:
            pass
        for name in filenames:
            try:
                os.chmod(os.path.join(dirpath, name), 0o600)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def snapshot() -> None:
    force_rmtree(GRADED_DIR)
    GRADED_DIR.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(SUBMISSION_DIR, followlinks=False):
        rel_root = Path(dirpath).relative_to(SUBMISSION_DIR)
        for d in list(dirnames):
            if os.path.islink(os.path.join(dirpath, d)):
                dirnames.remove(d)
        (GRADED_DIR / rel_root).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = Path(dirpath) / name
            st = os.lstat(src)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            shutil.copyfile(src, GRADED_DIR / rel_root / name)
    for dirpath, dirnames, filenames in os.walk(GRADED_DIR):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), 0o444)
        os.chmod(dirpath, 0o555)


class IsolationError(RuntimeError):
    """The grader could not establish the sandbox it requires."""


def kill_container_uid(uid: int) -> None:
    if not Path("/.dockerenv").exists():
        return
    for _ in range(5):
        victims = []
        for status_path in Path("/proc").glob("[0-9]*/status"):
            try:
                uid_line = next(
                    line for line in status_path.read_text().splitlines()
                    if line.startswith("Uid:")
                )
                if int(uid_line.split()[1]) == uid:
                    victims.append(int(status_path.parent.name))
            except (OSError, StopIteration, ValueError):
                continue
        if not victims:
            return
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.01)


def harden_and_spawn(cmd: list[str], scratch: Path, timeout: int):
    import site

    lib_paths = [p for p in site.getsitepackages() if os.path.isdir(p)]
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str) and os.path.isdir(user_site):
            lib_paths.append(user_site)
    except (AttributeError, TypeError):
        pass

    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "PYTHONPATH": os.pathsep.join(lib_paths),
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",
        "MNISTBOARD_AGENTS",
        "MNISTBOARD_CPUS",
        "MNISTBOARD_THREADS",
        "MNISTBOARD_GPU_MEMORY_LIMIT_MIB",
        "MNISTBOARD_GPU_ALLOCATOR_MIB",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env.setdefault("OMP_NUM_THREADS", "3")
    env.setdefault("MKL_NUM_THREADS", "3")

    preexec = None
    child_uid = None
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        for secret in (TEST_NPZ.parent, Path("/data")):
            if not secret.exists():
                continue
            try:
                os.chmod(secret, 0o700)
            except OSError as exc:
                raise IsolationError(f"cannot restrict {secret}: {exc}") from exc
        try:
            import pwd

            nobody = pwd.getpwnam("nobody")
        except (KeyError, ImportError) as exc:
            raise IsolationError(f"no unprivileged user: {exc}") from exc
        for path in (EVAL_DIR, scratch):
            for dirpath, dirnames, filenames in os.walk(path):
                for entry in [dirpath] + [
                    os.path.join(dirpath, n) for n in dirnames + filenames
                ]:
                    try:
                        os.chown(entry, nobody.pw_uid, nobody.pw_gid)
                    except OSError as exc:
                        raise IsolationError(f"cannot hand over {entry}: {exc}") from exc
        child_uid = nobody.pw_uid

        def preexec() -> None:  # type: ignore[misc]
            os.setsid()
            os.setgid(nobody.pw_gid)
            os.setuid(nobody.pw_uid)
    else:
        preexec = os.setsid

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=str(scratch),
        preexec_fn=preexec,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if child_uid is not None:
            kill_container_uid(child_uid)
    return subprocess.CompletedProcess(cmd, returncode, "", "")


def base_metrics() -> dict[str, float]:
    return {
        "reward": 0.0,
        "accuracy": 0.0,
        "correct": 0.0,
        "total": 0.0,
        "qualified": 0.0,
        "model_bytes": 0.0,
        "num_files": 0.0,
        "guardrails_passed": 0.0,
        "flag_bad_layout": 0.0,
        "flag_invalid_predictions": 0.0,
        "flag_subprocess_failed": 0.0,
        "flag_isolation_failed": 0.0,
        "verifier_error": 0.0,
        "shuffle_seed": 0.0,
    }


def finish(m: dict[str, float], msg: str) -> int:
    print(msg)
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(json.dumps(m, sort_keys=True) + "\n")
    if ORACLE_MODE:
        os.chmod(REWARD_PATH.parent, 0o700)
        os.chmod(REWARD_PATH, 0o600)
        reward_uid = os.environ.get("MNISTBOARD_REWARD_UID")
        if reward_uid:
            os.chown(REWARD_PATH, int(reward_uid), -1)
    return 0


def archive_submission() -> None:
    try:
        dest = REWARD_PATH.parent / "submission"
        force_rmtree(dest)
        shutil.copytree(GRADED_DIR, dest, symlinks=False)
        for dirpath, dirnames, filenames in os.walk(dest):
            os.chmod(dirpath, 0o755)
            for name in filenames:
                os.chmod(os.path.join(dirpath, name), 0o644)
    except OSError as exc:
        print(f"(could not archive submission: {exc})")


def run_inference(images, tag: str, timeout: int):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    scratch = EVAL_DIR / f"run-{tag}"
    force_rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    runner = scratch / "run_infer.py"
    shutil.copyfile(RUN_INFER, runner)

    images_path = EVAL_DIR / f"images-{tag}.npz"
    preds_path = EVAL_DIR / f"preds-{tag}.npy"
    if preds_path.exists():
        preds_path.unlink()
    np.savez(images_path, images=images)

    cmd = [
        sys.executable, str(runner),
        "--submission", str(GRADED_DIR),
        "--images", str(images_path),
        "--out", str(preds_path),
    ]
    proc = harden_and_spawn(cmd, scratch, timeout)
    return proc, preds_path


def main() -> int:
    m = base_metrics()
    try:
        if not SUBMISSION_DIR.is_dir():
            return finish(m, f"No submission directory at {SUBMISSION_DIR}")

        try:
            files = check_layout(SUBMISSION_DIR)
            model_bytes = gzip_size(SUBMISSION_DIR)
        except SpecError as exc:
            m["flag_bad_layout"] = 1.0
            return finish(m, f"Rejected: invalid submission layout: {exc}")

        m.update(
            model_bytes=float(model_bytes),
            num_files=float(len(files)),
            shuffle_seed=float(SHUFFLE_SEED),
        )
        print(format_report(SUBMISSION_DIR, model_bytes, files))
        print()

        snapshot()
        if not ORACLE_MODE:
            archive_submission()

        data = np.load(TEST_NPZ)
        images = data["images"].astype(np.uint8)
        labels = data["labels"].astype(np.int64)
        total = len(labels)
        m["total"] = float(total)
        rng = np.random.default_rng(SHUFFLE_SEED)
        perm = rng.permutation(total)
        images_shuf = np.ascontiguousarray(images[perm])
        labels_shuf = labels[perm]

        try:
            proc, preds_path = run_inference(images_shuf, "real", INFER_TIMEOUT_SEC)
        except IsolationError as exc:
            m["flag_isolation_failed"] = 1.0
            return finish(m, f"REFUSING TO GRADE: sandbox failed: {exc}")
        except subprocess.TimeoutExpired:
            m["flag_subprocess_failed"] = 1.0
            return finish(m, f"Inference timed out after {INFER_TIMEOUT_SEC}s")

        print("---- inference stdout ----")
        print(proc.stdout[-3000:])
        print("---- inference stderr ----")
        print(proc.stderr[-3000:])
        if proc.returncode != 0:
            m["flag_subprocess_failed"] = 1.0
            return finish(m, f"Inference failed (rc={proc.returncode})")

        try:
            prediction_fd = os.open(preds_path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            m["flag_invalid_predictions"] = 1.0
            return finish(m, "Invalid predictions: missing output file")
        try:
            prediction_stat = os.fstat(prediction_fd)
            if (
                not stat.S_ISREG(prediction_stat.st_mode)
                or prediction_stat.st_size > MAX_PREDICTION_BYTES
            ):
                m["flag_invalid_predictions"] = 1.0
                return finish(m, "Invalid predictions: unsafe output file")
            with os.fdopen(prediction_fd, "rb") as prediction_file:
                prediction_fd = -1
                preds = np.asarray(
                    np.load(prediction_file, allow_pickle=False)
                ).reshape(-1)
        except (OSError, ValueError):
            m["flag_invalid_predictions"] = 1.0
            return finish(m, "Invalid predictions: unreadable output file")
        finally:
            if prediction_fd >= 0:
                os.close(prediction_fd)
        if len(preds) != total:
            m["flag_invalid_predictions"] = 1.0
            return finish(m, f"Invalid predictions: len={len(preds)} != {total}")
        if preds.min() < 0 or preds.max() > 9:
            m["flag_invalid_predictions"] = 1.0
            return finish(m, f"Invalid predictions: range=[{preds.min()},{preds.max()}]")

        correct = int((preds.astype(np.int64) == labels_shuf).sum())
        accuracy = 100.0 * correct / total

        qualified = accuracy >= QUALIFY_ACCURACY
        m.update(
            reward=framework_score(qualified, model_bytes),
            accuracy=round(accuracy, 4),
            correct=float(correct),
            qualified=1.0 if qualified else 0.0,
            guardrails_passed=1.0,
        )
        print()
        print(f"Correct : {correct}/{total}  ({accuracy:.2f}%)")
        print(f"Model   : {model_bytes} bytes gzipped over {len(files)} files")
        return finish(m, f"Qualified: {'yes' if qualified else 'no'}")

    except Exception:
        traceback.print_exc()
        m["verifier_error"] = 1.0
        return finish(m, "Verifier error (see traceback above).")


if __name__ == "__main__":
    raise SystemExit(main())
