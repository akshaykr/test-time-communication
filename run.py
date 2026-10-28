from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parent
PROTOCOLS = ("antiherding", "antiherding_ml", "anti_plateau")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run MNIST compression, polyomino packing, or ARC-AGI-3.")
    result.add_argument("task", choices=("mnist", "polyomino", "arc-agi-3"))
    result.add_argument("--model", required=True)
    result.add_argument("--agents", type=positive, default=1)
    result.add_argument("--protocol", choices=PROTOCOLS)
    result.add_argument("--task-prompt-only", action="store_true",
                        help="MNIST solo: omit the dedicated solo protocol")
    result.add_argument("--trials", type=positive, default=1)
    result.add_argument("--concurrent", type=positive, default=1)
    result.add_argument("--timeout", type=positive, default=3600, help="agent seconds per trial")
    result.add_argument("--continuations", type=nonnegative, default=1000,
                        help="maximum additional sessions (default: 1000)")
    result.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="max")
    result.add_argument("--long-context", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--cpus", type=positive)
    result.add_argument("--memory-mb", type=positive)
    result.add_argument("--name")
    result.add_argument("--force-build", action="store_true")
    result.add_argument("--copilot-binary", default=os.environ.get("COPILOT_BINARY"))
    result.add_argument("--polyomino-data", type=Path,
                        default=ROOT / ".cache/frontier-cs/algorithmic/problems/0")
    result.add_argument("--arc-game", default="ls20")
    result.add_argument("--arc-budget", type=positive, help="ARC actions per level")
    result.add_argument("--arc-target-levels", type=positive)
    result.add_argument("--arc-sync-freq", type=nonnegative,
                        help="ARC synchronizations per level budget (default: 0; disabled)")
    result.add_argument("--dry-run", action="store_true", help="print the command without starting trials")
    return result


def prepare(args: argparse.Namespace) -> tuple[list[str], dict[str, str], Path]:
    if args.task_prompt_only and (args.task != "mnist" or args.agents != 1):
        raise ValueError("--task-prompt-only requires mnist --agents 1")
    name = args.name or f"{args.task}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("--name must contain only letters, digits, underscores, and hyphens")
    protocol = args.protocol or ("antiherding_ml" if args.task == "mnist" else "antiherding")
    if args.task != "mnist" and protocol == "antiherding_ml":
        raise ValueError("antiherding_ml is only supported for MNIST")
    if args.task != "arc-agi-3" and (args.arc_game != "ls20" or args.arc_budget is not None
                                   or args.arc_target_levels is not None or args.arc_sync_freq is not None):
        raise ValueError("ARC options require the arc-agi-3 task")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.arc_game):
        raise ValueError("--arc-game must be a valid ARC game identifier")
    task = ROOT / "benchmarks" / ("mnistboard" if args.task == "mnist" else args.task)
    settings = tomllib.loads((task / "task.toml").read_text())
    env = dict(os.environ)
    mounts = []
    cached_binary = ROOT / ".cache/copilot/copilot"
    binary = args.copilot_binary or (
        str(cached_binary) if cached_binary.is_file() else shutil.which("copilot"))
    if not binary and args.dry_run:
        binary = str(Path.home() / ".local/bin/copilot")
    if not binary:
        raise ValueError("install the standalone Linux Copilot CLI using README.md "
                         "or pass --copilot-binary")
    binary = Path(binary).expanduser().resolve()
    if not args.dry_run and (not binary.is_file() or not os.access(binary, os.X_OK)):
        raise ValueError("Copilot binary must be an executable file")
    if not args.dry_run:
        with binary.open("rb") as stream:
            signature = stream.read(4)
        if binary.name != "copilot" or signature != b"\x7fELF":
            raise ValueError("--copilot-binary must point to the standalone Linux copilot executable")
    token_file = Path.home() / ".copilot/pat"
    if not env.get("COPILOT_GITHUB_TOKEN") and env.get("GH_TOKEN"):
        env["COPILOT_GITHUB_TOKEN"] = env["GH_TOKEN"]
    if not env.get("COPILOT_GITHUB_TOKEN") and token_file.is_file():
        env["COPILOT_GITHUB_TOKEN"] = token_file.read_text().strip()
    if not env.get("COPILOT_GITHUB_TOKEN") and not args.dry_run:
        raise ValueError("set COPILOT_GITHUB_TOKEN or GH_TOKEN to a PAT with Copilot Requests permission")
    mounts.append({"type": "bind", "source": str(binary.parent),
                   "target": "/mnt/release-copilot", "read_only": True,
                   "bind": {"create_host_path": False}})
    continuation = (task / "continuation.txt").read_text().rstrip("\n")
    continuations = args.continuations
    if args.task == "polyomino":
        data = args.polyomino_data.expanduser().resolve()
        if not args.dry_run and not all((data / path).exists() for path in ("config.yaml", "chk.cc", "testdata")):
            raise ValueError("prepare Frontier-CS problem 0 as described in README.md")
        env["POLYOMINO_DATA"] = str(data)
    elif args.task == "mnist" and not args.dry_run and not (task / "tests/mnist_test.npz").is_file():
        raise ValueError("run python benchmarks/mnistboard/prepare_data.py first")
    env.update(MULTIAGENT_N_AGENT=str(args.agents),
               MULTIAGENT_PROTOCOL=f"multiagent_protocols.{protocol}:MultiAgentProtocol",
               RELEASE_CONTINUATION=continuation, RELEASE_CONTINUATIONS=str(continuations),
               MULTIAGENT_AGENT_EFFORT=args.effort or "", RELEASE_LONG_CONTEXT=str(int(args.long_context)),
               RELEASE_TASK=args.task)
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = ["harbor", "run", "-p", str(task), "--yes", "--env", "docker",
           "--force-build" if args.force_build else "--no-force-build",
           "--mounts-json", json.dumps(mounts), "--jobs-dir", str(ROOT / "jobs"),
           "--job-name", name, "-k", str(args.trials), "-n", str(args.concurrent),
           "--agent-import-path", "agent:CopilotSingleAgent" if args.task_prompt_only else "agent:CollaborationAgent", "-m", args.model,
           "--agent-timeout-multiplier", str(args.timeout / settings["agent"]["timeout_sec"])]
    cpus = args.cpus
    memory_mb = args.memory_mb
    if args.task == "mnist" and args.agents == 4:
        cpus = 12 if cpus is None else cpus
        memory_mb = 32768 if memory_mb is None else memory_mb
    if cpus is not None:
        cmd += ["--override-cpus", str(cpus)]
    if memory_mb is not None:
        cmd += ["--override-memory-mb", str(memory_mb)]
    if args.task == "mnist":
        cmd += ["--ae", f"MNISTBOARD_AGENTS={args.agents}",
                "--ve", f"MNISTBOARD_AGENTS={args.agents}"]
    if args.task == "arc-agi-3":
        arc_env = {"ARCCTL_DEFAULT_GAME": args.arc_game,
                   "ARCCTL_SYNC_FREQ": str(0 if args.arc_sync_freq is None else args.arc_sync_freq)}
        if args.arc_budget is not None:
            arc_env["ARCCTL_FIXED_LEVEL_BUDGET"] = str(args.arc_budget)
        if args.arc_target_levels is not None:
            arc_env["ARCCTL_TARGET_LEVELS"] = str(args.arc_target_levels)
        for key, value in arc_env.items():
            env[key] = value
            cmd += ["--ae", f"{key}={value}", "--ve", f"{key}={value}"]
    return cmd, env, ROOT / "jobs" / name


def main() -> int:
    args = parser().parse_args()
    try:
        cmd, env, job = prepare(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    if job.exists():
        raise SystemExit(f"job already exists: {job}; choose a new --name")
    worker = None
    log = None
    try:
        if args.task == "mnist":
            job.mkdir(parents=True)
            log = (job / "test-oracle.log").open("w")
            worker = subprocess.Popen([
                sys.executable, str(ROOT / "benchmarks/mnistboard/tools/test_oracle_worker.py"),
                "--run-root", str(job), "--watch-pid", str(os.getpid())],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        return subprocess.call(cmd, cwd=ROOT, env=env)
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
