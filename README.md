# Multi-Agent Test-Time Communication

Three tasks: MNIST solver compression (at least 99.4% sealed-test accuracy, then
minimize compressed bytes), polyomino packing (Frontier-CS problem 0), and
interactive ARC-AGI-3 games.
We run models through Copilot, but feel free to add your favorite, e.g., codex, claude
(the harness will affect accuracy).

## Shared setup

Requires Python 3.12+, Docker with Compose running Linux containers, and a
standalone Linux Copilot CLI executable. MNIST also requires a Linux host,
an NVIDIA GPU, and NVIDIA Container Toolkit.

```bash
cd test-time-communication
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
docker info
docker compose version
```

The requirements install Harbor 0.23.0.
Start Docker before proceeding. On macOS, run `open -a Docker` and wait for Docker
Desktop to finish starting. On Linux with systemd, run `sudo systemctl start docker`.
`docker info` must connect to a running daemon before launching experiments.

Define this helper to download each task's pinned
[official binary](https://github.com/github/copilot-cli/releases).
Select the container architecture (`x64` or `arm64`):

```bash
set -e
COPILOT_ARCH=arm64 # use x64 for x86-64 Linux containers
install_copilot() {
  COPILOT_VERSION="$1"
  COPILOT_DIR="$PWD/.cache/copilot-${COPILOT_VERSION}"
  mkdir -p "$COPILOT_DIR"
  curl -fL "https://github.com/github/copilot-cli/releases/download/v${COPILOT_VERSION}/copilot-linux-${COPILOT_ARCH}.tar.gz" -o "$COPILOT_DIR/copilot.tar.gz"
  tar -xzf "$COPILOT_DIR/copilot.tar.gz" -C "$COPILOT_DIR" copilot
  chmod 755 "$COPILOT_DIR/copilot"
  export COPILOT_BINARY="$COPILOT_DIR/copilot"
}
export COPILOT_GITHUB_TOKEN=your_copilot_pat
```

The PAT needs Copilot Requests permission. The launcher checks
`COPILOT_GITHUB_TOKEN`, then `GH_TOKEN`, then `~/.copilot/pat` (use mode 600).
It passes the selected token as `COPILOT_GITHUB_TOKEN` and removes `GH_TOKEN`
and `GITHUB_TOKEN` from the child environment.
On macOS, download the Linux binary matching
Docker's container architecture; it runs inside the container.

Replace model identifiers below with those available to your account.
`--copilot-binary` selects a standalone Linux executable explicitly.
`--timeout` bounds each trial in seconds. `--continuations` sets the maximum
additional sessions, resumed with the original continuation prompt. It defaults to
1,000 for all tasks. Maximum
reasoning effort and long context are enabled by default; override with `--effort` or `--no-long-context`.
Older CLI builds use their native context window when no context flag exists.
With `--model auto`, Copilot chooses reasoning effort.
Use `--dry-run` to inspect a launch without calling
a model. Results and agent logs are written under `jobs/`. Agents retain the original
`agent-*-output.log` trajectories and session `.otel` files for token accounting;
message-content capture is disabled.

## ARC-AGI-3

Uses Copilot CLI **1.0.54**. No host dataset preparation is required.

```bash
install_copilot 1.0.54
games=(lp85 ar25 dc22 sc25 ka59 ft09 wa30 tr87 m0r0 tn36 re86 sb26 tu93 s5i5 lf52 g50t cd82 sk48 su15 bp35 ls20 vc33 sp80 cn04 r11l)
for game in "${games[@]}"; do
  python run.py arc-agi-3 --model claude-sonnet-4.6 --arc-game "$game" --agents 1 --trials 64 --timeout 172800 --arc-sync-freq 2
  for agents in 3 5; do
    python run.py arc-agi-3 --protocol antiherding --model claude-sonnet-4.6 --arc-game "$game" --agents "$agents" --trials 20 --timeout 172800 --arc-sync-freq 2
  done
done
```

The image installs `arc-agi`, where the sidecar keeps game implementation and scoring
separate from agents. Each agent has its own authenticated session and action
budget. Teams use the `antiherding` communication protocol.
A single agent run receives only the original task instruction.

Omit `--arc-budget` and `--arc-target-levels` to use native per-level budgets and play the complete game.
Sync defaults to `0` (disabled). The commands above use `--arc-sync-freq 2`
to synchronize every half-level budget. Containers share 4 CPUs and 8 GiB. After a successful CLI
session, agents stop if any valid action was recorded. Otherwise they receive
up to 1,000 continuations by default, then fail if no action was recorded.

## Polyomino packing

Uses Copilot CLI **1.0.70** for long-horizon experiments. Download
[Frontier-CS problem 0](https://github.com/FrontierCS/Frontier-CS/tree/main/algorithmic/problems/0):

```bash
install_copilot 1.0.70
git clone --depth 1 --filter=blob:none --sparse https://github.com/FrontierCS/Frontier-CS.git .cache/frontier-cs
git -C .cache/frontier-cs sparse-checkout set algorithmic/problems/0
python run.py polyomino --model claude-sonnet-4.6 --agents 1 --trials 60 --timeout 10800
python run.py polyomino --model claude-sonnet-4.6 --agents 3 --protocol antiherding --trials 20 --timeout 10800
for model in claude-sonnet-4.6 claude-opus-4.6; do
  python run.py polyomino --model "$model" --agents 1 --trials 12 --timeout 259200
  python run.py polyomino --model "$model" --agents 4 --protocol antiherding --trials 2 --timeout 259200
done
```

Teams default to `antiherding`, preserving distinct approaches and sharing evidence
before converging. Single-agent runs receive only the task instruction.
Each agent defaults to 1,000 continuation attempts if it concludes before the time limit.
Each container shares 2 CPUs and 6 GiB.
The judge requires a privileged Linux container and grades 70 cases with a
2-second time limit and 256 MiB per case.

Every iterative submission archives the exact source sent to the judge under
`jobs/<job>/<trial>/agent/submitted_sources/<submission_uuid>.cpp` before the
request, including submissions that later fail or time out. Records in
`agent/submissions.jsonl` link the UUID, original path, archive path (relative to
that log), SHA-256, and grading outcome. The verifier mirrors this history and
the sources under `verifier/`; its final candidate is saved separately as
`verifier/final_solution.cpp`. These files survive container cleanup.

## MNIST compression

Uses Copilot CLI **1.0.78**. Requires a Linux host, an NVIDIA GPU, and NVIDIA
Container Toolkit. The paper uses an A100. Prepare the dataset before launching:

```bash
install_copilot 1.0.78
python benchmarks/mnistboard/prepare_data.py
python run.py mnist --protocol antiherding_ml --model gpt-5.6-sol --agents 1 --trials 4 --timeout 345600
python run.py mnist --protocol antiherding_ml --model gpt-5.6-sol --agents 4 --timeout 345600
```

The communication protocol is `antiherding_ml`, which is similar to `antiherding` but with
more information about sharing CPU/GPU resources.
MNIST solo uses the dedicated solo protocol; `--task-prompt-only` omits it.
Solo containers default to 3 CPUs and 8 GiB. Four-agent MNIST runs automatically
default to 12 CPUs and 32 GiB; explicit `--cpus` and `--memory-mb` flags override
these defaults independently. Other team sizes use the task defaults unless overridden.
Each agent has a 1,000 MiB GPU-memory limit. The host oracle starts automatically and returns
aggregate test accuracy. Final Harbor grading runs in the task container with
inference under an unprivileged user.

Use `--concurrent N` to run independent trials concurrently if resources permit.
For best@k, use separate one-agent trials; `--agents k` launches a communicating
team.

Checks: `pip install pytest && python -m pytest -q benchmarks/mnistboard/tests/test_oracle.py`.
