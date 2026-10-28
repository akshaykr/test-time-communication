# Copilot execution paths retained from multiagent-cli:agent-collab/meta_duo_agent.py.
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from multiagent_protocols.base import ProtocolPaths, build_protocol, protocol_to_dict
from arc_sessions import ArcAgi3Runtime

DEFAULT_PROTOCOL = "multiagent_protocols.antiherding:MultiAgentProtocol"
DEFAULT_CONTAINER_COPILOT_BINARY = "/mnt/release-copilot/copilot"
COPILOT_BINARY_ENV = "META_DUO_CONTAINER_COPILOT_BINARY"
COPILOT_SECRET_ENV_VARS = "COPILOT_GITHUB_TOKEN"
COMM_DIR = f"{EnvironmentPaths.agent_dir.as_posix()}/multiagent-collab"
REDACT_ARC_TOKEN_SED = (
    r's/ARCCTL_SESSION_TOKEN=[^[:space:]"\\]+/ARCCTL_SESSION_TOKEN=<redacted>/g'
)
COPILOT_AVAILABLE_TOOLS = (
    "bash",
    "read_bash",
    "write_bash",
    "stop_bash",
    "view",
    "glob",
    "grep",
    "rg",
    "apply_patch",
)
COPILOT_EXCLUDED_TOOLS = ("task", "read_agent", "write_agent", "list_agents")
COPILOT_TRAJECTORY_METRICS_SOURCE = (
    pathlib.Path(__file__).with_name("copilot_trajectory_metrics.py").read_text()
)

COPILOT_NO_ACTION_REPROMPTS = int(os.environ.get("RELEASE_CONTINUATIONS", "1000"))


def _no_action_feedback_body() -> str:
    return os.environ["RELEASE_CONTINUATION"]


def _non_arc_continuation_enabled() -> bool:
    return os.environ["RELEASE_TASK"] != "arc-agi-3" and COPILOT_NO_ACTION_REPROMPTS > 0


def copilot_tool_flags() -> str:
    available = " ".join(f"--available-tools={shlex.quote(tool)}" for tool in COPILOT_AVAILABLE_TOOLS)
    excluded = " ".join(f"--excluded-tools={shlex.quote(tool)}" for tool in COPILOT_EXCLUDED_TOOLS)
    return f"{available} {excluded}"


class CopilotAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "release-collaboration"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(environment, command=(
            "apt-get update && apt-get install -y --no-install-recommends "
            "ca-certificates curl git ripgrep"), env={"DEBIAN_FRONTEND": "noninteractive"})
        await self.exec_as_agent(environment, command=(
            "set -e; mkdir -p $HOME/.local/bin; "
            "printf '%s\n' '#!/usr/bin/env bash' "
            "'echo \"gh is disabled for Copilot auth isolation; run copilot login instead.\" >&2' "
            "'exit 127' > $HOME/.local/bin/gh; chmod 700 $HOME/.local/bin/gh; "
            f"test -x {DEFAULT_CONTAINER_COPILOT_BINARY}; "
            f"ln -sf {DEFAULT_CONTAINER_COPILOT_BINARY} $HOME/.local/bin/copilot"))

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; copilot --no-auto-update --version'

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    def _cli_model(self) -> str:
        return (self.model_name or "claude-sonnet-4.6").split("/")[-1]

    def _build_env(self, protocol_import: str) -> dict[str, str]:
        env = {
            "IS_SANDBOX": "1",
            "MULTIAGENT_PROTOCOL": protocol_import,
            "COPILOT_ALLOW_ALL": "true",
            "COPILOT_AUTO_UPDATE": "false",
            "RELEASE_LONG_CONTEXT": os.environ.get("RELEASE_LONG_CONTEXT", "1"),
        }
        token = os.environ.get("COPILOT_GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "COPILOT_GITHUB_TOKEN is not set in the harness environment; "
                "Copilot agents require a PAT."
            )
        env["COPILOT_GITHUB_TOKEN"] = token
        return env

    def _copilot_action_feedback_helpers(self) -> str:
        return r"""
COPILOT_API_RETRY_PATTERN='(api_error_status.?40(1|9)|status.?code.?40(1|9)|http[^[:alnum:]]*40(1|9)|40(1|9)[^[:alnum:]]*(unauthori[sz]ed|too many requests)|unauthori[sz]ed|rate.?limit|too many requests|throttl|quota|invalid_grant|invalid.*token|expired.*token|refresh.*token|access.*token|oauth|please log in|not logged in)'

configure_copilot_command() {
  copilot_command=("$copilot_binary")
  local copilot_help
  copilot_help=$("$copilot_binary" --no-auto-update --help) || return $?
  effort_option="--effort"
  if grep -Eq '^[[:space:]]*--reasoning-effort([[:space:]]|=|$)' <<< "$copilot_help"; then
    effort_option="--reasoning-effort"
  fi
  context_flag=""
  if [ "${RELEASE_LONG_CONTEXT:-1}" = "1" ] && \
     grep -Eq '^[[:space:]]*--context([[:space:]]|=|$)' <<< "$copilot_help"; then
    context_flag="--context long_context"
  fi
}

append_copilot_telemetry() {
  local attempt_log_file="$1"
  local otel_file="$2"
  python3 -c "$trajectory_metrics_source" "$attempt_log_file" "$otel_file" || {
    echo "Copilot telemetry capture incomplete for $attempt_log_file; see its telemetry_error event" >&2
  }
}

extract_copilot_session_id() {
  local source_log="$1"
  python3 - "$source_log" <<'PY'
import json
import sys

session_id = ""
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()

for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    value = event.get("sessionId")
    if isinstance(value, str) and value:
        session_id = value
    data = event.get("data")
    if isinstance(data, dict):
        value = data.get("sessionId")
        if isinstance(value, str) and value:
            session_id = value

if session_id:
    print(session_id)
PY
}

has_recorded_arc_action() {
  local state_file="$1"
  python3 - "$state_file" <<'PY'
import json
import pathlib
import sys

def positive(value):
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False

path = pathlib.Path(sys.argv[1])
if not path.exists():
    sys.exit(1)

try:
    data = json.loads(path.read_text(errors="replace"))
except (OSError, json.JSONDecodeError):
    sys.exit(1)

budget = data.get("action_budget")
if isinstance(budget, dict):
    for key in ("session_actions_used", "actions_used", "current_level_actions_used"):
        if positive(budget.get(key)):
            sys.exit(0)

level_clear_actions = data.get("level_clear_actions")
if isinstance(level_clear_actions, dict):
    for value in level_clear_actions.values():
        if positive(value):
            sys.exit(0)

last_action = data.get("last_action")
if isinstance(last_action, str) and last_action.startswith("ACTION"):
    sys.exit(0)

sys.exit(1)
PY
}

is_arc_runtime() {
  command -v arcctl >/dev/null 2>&1 \
    && arcctl --help 2>/dev/null | grep -q "ARC-AGI-3"
}

write_no_action_feedback() {
  local feedback_file="$1"
  cat > "$feedback_file" <<'EOF'
__NO_ACTION_FEEDBACK_BODY__
EOF
}

append_no_action_feedback_event() {
  local log_file="$1"
  local attempt="$2"
  local max_attempts="$3"
  printf '{"type":"environment.feedback","data":{"event":"no_valid_arc_action","message":"Previous turn produced no valid ARC action; continuing the same Copilot session.","attempt":%s,"maxAttempts":%s}}\n' "$attempt" "$max_attempts" >> "$log_file"
}

append_optimization_continuation_event() {
  local log_file="$1"
  local attempt="$2"
  local max_attempts="$3"
  printf '{"type":"environment.feedback","data":{"event":"optimization_continuation","message":"Previous turn ended; continuing the same Copilot session for another optimization pass.","attempt":%s,"maxAttempts":%s}}\n' "$attempt" "$max_attempts" >> "$log_file"
}

is_retryable_copilot_api_error() {
  local attempt_log_file="$1"
  if grep -Eiq "$COPILOT_API_RETRY_PATTERN" "$attempt_log_file"; then
    return 0
  fi
  if [ -n "${copilot_log_dir:-}" ] && [ -d "$copilot_log_dir" ]; then
    local copilot_process_log
    for copilot_process_log in "$copilot_log_dir"/*.log; do
      [ -e "$copilot_process_log" ] || continue
      if grep -Eiq "$COPILOT_API_RETRY_PATTERN" "$copilot_process_log"; then
        return 0
      fi
    done
  fi
  return 1
}

copilot_api_retry_sleep_seconds() {
  local retry="$1"
  local base="${META_DUO_COPILOT_API_RETRY_BASE_SLEEP:-30}"
  local cap="${META_DUO_COPILOT_API_RETRY_MAX_SLEEP:-600}"
  case "$retry" in ''|*[!0-9]*) retry=1 ;; esac
  case "$base" in ''|*[!0-9]*) base=30 ;; esac
  case "$cap" in ''|*[!0-9]*) cap=600 ;; esac
  if [ "$retry" -lt 1 ]; then
    retry=1
  fi
  if [ "$base" -lt 1 ]; then
    base=30
  fi
  if [ "$cap" -lt "$base" ]; then
    cap="$base"
  fi
  local delay="$base"
  local step=1
  while [ "$step" -lt "$retry" ] && [ "$delay" -lt "$cap" ]; do
    delay=$((delay * 2))
    if [ "$delay" -gt "$cap" ]; then
      delay="$cap"
    fi
    step=$((step + 1))
  done
  if [ "$delay" -gt "$cap" ]; then
    delay="$cap"
  fi
  if [ "$delay" -lt "$cap" ]; then
    delay=$((delay + (RANDOM % base)))
    if [ "$delay" -gt "$cap" ]; then
      delay="$cap"
    fi
  fi
  echo "$delay"
}

append_copilot_api_retry_event() {
  local log_file="$1"
  local retry="$2"
  local max_retries="$3"
  local code="$4"
  local delay="$5"
  printf '{"type":"environment.feedback","data":{"event":"copilot_api_retry","message":"Copilot CLI returned a retryable 401/429/API error; backing off and resuming the same session instead of failing the run.","attempt":%s,"maxAttempts":%s,"exitCode":%s,"sleepSeconds":%s}}\n' "$retry" "$max_retries" "$code" "$delay" >> "$log_file"
}

append_copilot_api_retry_exhausted_event() {
  local log_file="$1"
  local max_retries="$2"
  local code="$3"
  printf '{"type":"environment.feedback","data":{"event":"copilot_api_retries_exhausted","message":"Copilot CLI kept returning retryable 401/429/API errors after all retry attempts.","maxAttempts":%s,"exitCode":%s}}\n' "$max_retries" "$code" >> "$log_file"
}
""".strip().replace("__NO_ACTION_FEEDBACK_BODY__", _no_action_feedback_body())

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        protocol_import = self._protocol_import()
        model = self._cli_model()
        paths = ProtocolPaths.default(COMM_DIR)
        spec = build_protocol(protocol_import, instruction, paths, model)
        env = self._build_env(protocol_import)

        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        session_dirs: list[str] = []
        agent_logs: list[str] = []
        for idx, agent in enumerate(spec.agents):
            runtime_label = f"{idx}-{agent.id}"
            session_dirs.append(f"{agent_dir}/sessions-{runtime_label}")
            # Persist logs directly so timeouts do not lose them.
            agent_logs.append(f"{agent_dir}/agent-{runtime_label}-output.log")

        protocol_spec_record: dict[str, Any] = protocol_to_dict(spec)
        protocol_record: dict[str, Any] = {
            "protocol_import": protocol_import,
            "protocol_source_sha256": self._protocol_source_sha(protocol_import),
            "model": model,
            "shared_prompt_path": paths.shared_prompt,
            "spec": protocol_spec_record,
            "cli_backend": "copilot",
        }
        protocol_json = json.dumps(protocol_record, indent=2, sort_keys=True)
        shared_prompt = spec.agents[0].prompt
        runner_path = f"{paths.comm_dir}/run-agent.sh"
        copilot_binary = os.environ.get(
            COPILOT_BINARY_ENV, DEFAULT_CONTAINER_COPILOT_BINARY
        )
        benchmark_runtime = ArcAgi3Runtime.for_agent_ids(
            [agent.id for agent in spec.agents]
        )
        identity_env = (
            'AGENT_ID="$agent_id" AGENT_INDEX="$idx" AGENT_COUNT="$agent_count" '
        )
        argument_init = (
            '  idx="$1"\n'
            '  agent_id="$2"\n'
            '  agent_count="$3"\n'
            "  shift 3\n"
        )
        agent_effort = os.environ.get("MULTIAGENT_AGENT_EFFORT", "").strip()
        # Apply effort only to models that support it: Haiku rejects reasoning-effort
        # configuration, so skip it for haiku models.
        if agent_effort:
            _eff = shlex.quote(agent_effort)
            effort_flag = f'$(case "$model" in auto|*haiku*) ;; *) printf "%s %s" "$effort_option" {_eff} ;; esac) '
        else:
            effort_flag = ""
        context_flag = "$context_flag "
        non_arc_continuation = "1" if _non_arc_continuation_enabled() else "0"
        agent_role_intro = (
            "You are solving this task alone. Follow the solo work protocol exactly.\n\n"
            if len(spec.agents) == 1
            else (
                "You are one agent of many concurrent agents solving a task together. "
                "Follow this coordination protocol exactly.\n\n"
            )
        )
        tool_flags = copilot_tool_flags()
        runner_script = f"""#!/usr/bin/env bash
set +e
copilot_binary={shlex.quote(copilot_binary)}
redact_arc_token={shlex.quote(REDACT_ARC_TOKEN_SED)}
{self._copilot_action_feedback_helpers()}

run_agent() {{
{argument_init}  session_dir="$1"
  log_file="$2"
  model="$3"
  shared_prompt_path="$4"
  instruction="$5"
  : "${{ARCCTL_SESSION_TOKEN:?missing ARC session token}}"
  prompt_file="$session_dir/copilot-prompt.txt"
  copilot_log_dir="$session_dir/copilot-logs"
  trajectory_metrics_source={shlex.quote(COPILOT_TRAJECTORY_METRICS_SOURCE)}
  mkdir -p "$session_dir" "$copilot_log_dir"
  configure_copilot_command
  config_code=$?
  if [ "$config_code" -ne 0 ]; then
    exit "$config_code"
  fi
  state_file="{agent_dir}/arc_agi3_runs/$agent_id.state.json"
  is_arc_task=0
  if is_arc_runtime; then
    is_arc_task=1
  fi
  non_arc_continuation={non_arc_continuation}
  {{
    printf %s {shlex.quote(agent_role_intro)}
    cat "$shared_prompt_path"
    printf '\\n\\nTask instruction:\\n%s\\n' "$instruction"
  }} > "$prompt_file"
  # Startup jitter to spread Copilot model-load API hits across concurrent workers
  sleep "$(awk 'BEGIN{{srand(); print int(rand()*30)}}')"

  invoke_copilot() {{
    local attempt_prompt_file="$1"
    local attempt_log_file="$2"
    local otel_file="${{attempt_log_file%.jsonl}}.otel"
    shift 2
    : > "$otel_file"
    {identity_env}COPILOT_ALLOW_ALL=true COPILOT_AUTO_UPDATE=false \\
      COPILOT_OTEL_ENABLED=true COPILOT_OTEL_EXPORTER_TYPE=file \\
      COPILOT_OTEL_FILE_EXPORTER_PATH="$otel_file" \\
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false \\
      COPILOT_AGENT_ERROR_CODES_TO_RETRY="${{COPILOT_AGENT_ERROR_CODES_TO_RETRY:-401,429,500,502,503,504,529}}" \\
      "${{copilot_command[@]}}" --model "$model" {effort_flag}{context_flag}--output-format json --allow-all \\
      --allow-all-paths --allow-all-urls --no-custom-instructions \\
      {tool_flags} \\
      --disable-builtin-mcps --secret-env-vars={COPILOT_SECRET_ENV_VARS} \\
      --stream off --silent --no-auto-update \\
      --log-dir "$copilot_log_dir" "$@" -p "$(cat "$attempt_prompt_file")" 2>&1 | sed -E "$redact_arc_token; s|$ARCCTL_SESSION_TOKEN|<arc-token-redacted>|g" > "$attempt_log_file"
    local code=${{PIPESTATUS[0]}}
    append_copilot_telemetry "$attempt_log_file" "$otel_file"
    return "$code"
  }}

  run_copilot_with_api_retries() {{
    local attempt_prompt_file="$1"
    local attempt_log_base="$2"
    local max_api_retries="${{META_DUO_COPILOT_API_RETRIES:-288}}"
    case "$max_api_retries" in ''|*[!0-9]*) max_api_retries=288 ;; esac
    local retry=0
    local code=0
    while true; do
      local attempt_log_file="$attempt_log_base"
      if [ "$retry" -gt 0 ]; then
        attempt_log_file="${{attempt_log_base%.jsonl}}-api-retry-$retry.jsonl"
      fi
      if [ -n "$session_id" ]; then
        invoke_copilot "$attempt_prompt_file" "$attempt_log_file" --resume="$session_id"
      else
        invoke_copilot "$attempt_prompt_file" "$attempt_log_file"
      fi
      code=$?
      cat "$attempt_log_file" >> "$log_file"
      new_session_id="$(extract_copilot_session_id "$attempt_log_file")"
      if [ -n "$new_session_id" ]; then
        session_id="$new_session_id"
      fi
      if [ "$code" -eq 0 ]; then
        return 0
      fi
      if ! is_retryable_copilot_api_error "$attempt_log_file"; then
        return "$code"
      fi
      if [ "$retry" -ge "$max_api_retries" ]; then
        append_copilot_api_retry_exhausted_event "$log_file" "$max_api_retries" "$code"
        return "$code"
      fi
      retry=$((retry + 1))
      delay="$(copilot_api_retry_sleep_seconds "$retry")"
      append_copilot_api_retry_event "$log_file" "$retry" "$max_api_retries" "$code" "$delay"
      sleep "$delay"
    done
  }}

  : > "$log_file"
  session_id=""
  submissions_file="{agent_dir}/submissions.jsonl"
  attempt_log="$session_dir/copilot-attempt-0.jsonl"
  run_copilot_with_api_retries "$prompt_file" "$attempt_log"
  code=$?
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
  if [ "$is_arc_task" -eq 1 ]; then
    if has_recorded_arc_action "$state_file"; then
      exit 0
    fi
  elif [ "$non_arc_continuation" -ne 1 ]; then
    exit 0
  fi

  max_no_action_reprompts={COPILOT_NO_ACTION_REPROMPTS}
  # Retain the original submission-log probe and its shell output behavior.
  prev_submissions=$(wc -l < "$submissions_file" 2>/dev/null || echo 0)
  case "$prev_submissions" in ''|*[!0-9]*) prev_submissions=0 ;; esac
  no_action_attempt=1
  while [ "$no_action_attempt" -le "$max_no_action_reprompts" ]; do
    feedback_file="$session_dir/copilot-no-action-feedback-$no_action_attempt.txt"
    attempt_log="$session_dir/copilot-attempt-$no_action_attempt.jsonl"
    write_no_action_feedback "$feedback_file"
    if [ "$is_arc_task" -eq 1 ]; then
      append_no_action_feedback_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
    else
      append_optimization_continuation_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
    fi
    run_copilot_with_api_retries "$feedback_file" "$attempt_log"
    code=$?
    if [ "$code" -ne 0 ]; then
      exit "$code"
    fi
    if [ "$is_arc_task" -eq 1 ]; then
      if has_recorded_arc_action "$state_file"; then
        exit 0
      fi
    fi
    no_action_attempt=$((no_action_attempt + 1))
  done

  if [ "$is_arc_task" -eq 1 ]; then
    printf '{{"type":"environment.feedback","data":{{"event":"no_valid_arc_action_reprompts_exhausted","message":"No valid ARC action was recorded after all environment feedback attempts.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
    exit 1
  fi
  printf '{{"type":"environment.feedback","data":{{"event":"optimization_continuations_completed","message":"Completed all configured optimization continuation passes.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
  exit 0
}}

run_agent "$@"
"""

        setup_parts = [
            f"rm -rf {shlex.quote(COMM_DIR)}",
            f"mkdir -p {shlex.quote(COMM_DIR)}",
            f"printf %s {shlex.quote(protocol_json)} > {shlex.quote(paths.comm_dir + '/protocol.json')}",
            f"printf %s {shlex.quote(shared_prompt)} > {shlex.quote(paths.shared_prompt)}",
            f"printf %s {shlex.quote(runner_script)} > {shlex.quote(runner_path)}",
            f"chmod 700 {shlex.quote(runner_path)}",
            benchmark_runtime.setup_command(),
            f"[ -x {shlex.quote(copilot_binary)} ] || "
            f"(echo {shlex.quote(f'Mounted Copilot binary is not executable at {copilot_binary}')} >&2; exit 1)",
        ]
        for path in spec.shared_paths:
            if path.endswith("/"):
                setup_parts.append(f"mkdir -p {shlex.quote(path)}")
            else:
                setup_parts.append(
                    f"mkdir -p {shlex.quote(str(pathlib.PurePosixPath(path).parent))}"
                )
                setup_parts.append(f": > {shlex.quote(path)}")
        for session_dir in session_dirs:
            setup_parts.append(f"mkdir -p {session_dir}/copilot-logs")

        await self.exec_as_agent(
            environment,
            command="set -euo pipefail; " + " && ".join(setup_parts),
            env={**env, **benchmark_runtime.setup_env()},
        )

        escaped_instruction = shlex.quote(instruction + benchmark_runtime.instruction_suffix())
        run_parts = ['export PATH="$HOME/.local/bin:$PATH"', "set +e"]
        for idx, agent in enumerate(spec.agents):
            identity_args = (
                f"{idx} {shlex.quote(agent.id)} {len(spec.agents)} "
            )
            token_prefix = benchmark_runtime.shell_token_prefix(idx)
            run_parts.append(
                f"{token_prefix} bash {shlex.quote(runner_path)} "
                f"{identity_args}"
                f"{shlex.quote(session_dirs[idx])} {shlex.quote(agent_logs[idx])} "
                f"{shlex.quote(model)} {shlex.quote(paths.shared_prompt)} "
                f"{escaped_instruction} "
                f">{agent_logs[idx]} 2>&1 & PID_{idx}=$!"
            )
        for idx in range(len(spec.agents)):
            run_parts.append(f"wait $PID_{idx}; EXIT_{idx}=$?")
        exit_labels = [agent.id for agent in spec.agents]
        exit_summary = ", ".join(
            f"{label}=$EXIT_{idx}" for idx, label in enumerate(exit_labels)
        )
        run_parts.append(f'echo "Agent exits: {exit_summary}"')
        success_check = " || ".join(f"[ $EXIT_{idx} -eq 0 ]" for idx in range(len(spec.agents)))
        run_parts.append(success_check)
        run_command = "; ".join(run_parts)
        run_env = {**env, **benchmark_runtime.indexed_token_env()}

        await self.exec_as_agent(environment, command=run_command, env=run_env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        total_input = 0
        total_output = 0
        for usage_file in self.logs_dir.glob("agent-*-output.log"):
            try:
                # errors="replace": agent logs can contain truncated/invalid UTF-8
                # (e.g. an agent dumping binary-ish output); a strict decode here
                # raises UnicodeDecodeError mid trial-recovery and crashes harbor.
                text = usage_file.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = event.get("data", {})
                if isinstance(data, dict):
                    total_input += int(data.get("inputTokens", 0) or 0)
                    total_output += int(data.get("outputTokens", 0) or 0)
                    usage = data.get("usage", {})
                    if isinstance(usage, dict):
                        total_input += int(usage.get("input_tokens", 0) or 0)
                        total_output += int(usage.get("output_tokens", 0) or 0)

        context.n_input_tokens = total_input
        context.n_output_tokens = total_output

    def _protocol_import(self) -> str:
        return os.environ.get("MULTIAGENT_PROTOCOL", DEFAULT_PROTOCOL)

    def _protocol_source_sha(self, import_path: str) -> str | None:
        spec = importlib.util.find_spec(import_path.split(":", 1)[0])
        if not spec or not spec.origin:
            return None
        try:
            return hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest()
        except OSError:
            return None


class CopilotSingleAgent(CopilotAgent):
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        model = self._cli_model()
        env = self._build_env(DEFAULT_PROTOCOL)
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        session_dir = f"{agent_dir}/copilot-single-session"
        log_file = f"{agent_dir}/agent-single-output.log"
        prompt_file = f"{session_dir}/copilot-prompt.txt"
        copilot_log_dir = f"{session_dir}/copilot-logs"
        benchmark_runtime = ArcAgi3Runtime.for_agent_ids(["single"])
        copilot_binary = os.environ.get(
            COPILOT_BINARY_ENV, DEFAULT_CONTAINER_COPILOT_BINARY
        )

        setup_cmd = " && ".join(
            [
                f"rm -rf {shlex.quote(session_dir)}",
                f"mkdir -p {shlex.quote(copilot_log_dir)}",
                f"printf %s {shlex.quote(instruction)} > {shlex.quote(prompt_file)}",
                benchmark_runtime.setup_command(),
                f"[ -x {shlex.quote(copilot_binary)} ] || "
                f"(echo {shlex.quote(f'Mounted Copilot binary is not executable at {copilot_binary}')} >&2; exit 1)",
            ]
        )
        await self.exec_as_agent(
            environment,
            command=f"set -euo pipefail; {setup_cmd}",
            env={**env, **benchmark_runtime.setup_env()},
        )

        agent_effort = os.environ.get("MULTIAGENT_AGENT_EFFORT", "").strip()
        # Apply effort only to models that support it (Haiku rejects it); evaluated
        # at runtime against $model.
        if agent_effort:
            _eff = shlex.quote(agent_effort)
            effort_flag = f'$(case "$model" in auto|*haiku*) ;; *) printf "%s %s" "$effort_option" {_eff} ;; esac) '
        else:
            effort_flag = ""
        context_flag = "$context_flag "
        non_arc_continuation = "1" if _non_arc_continuation_enabled() else "0"
        tool_flags = copilot_tool_flags()
        run_cmd = f"""
export PATH="$HOME/.local/bin:$PATH"
set +e
: "${{ARCCTL_SESSION_TOKEN:?missing ARC session token}}"
redact_arc_token={shlex.quote(REDACT_ARC_TOKEN_SED)}
copilot_binary={shlex.quote(copilot_binary)}
model={shlex.quote(model)}
log_file={shlex.quote(log_file)}
prompt_file={shlex.quote(prompt_file)}
copilot_log_dir={shlex.quote(copilot_log_dir)}
agent_dir={shlex.quote(agent_dir)}
agent_id="${{ARCCTL_SESSION_ID:-single}}"
trajectory_metrics_source={shlex.quote(COPILOT_TRAJECTORY_METRICS_SOURCE)}
state_file="$agent_dir/arc_agi3_runs/$agent_id.state.json"
non_arc_continuation={non_arc_continuation}
export AGENT_ID="$agent_id" AGENT_INDEX=0 AGENT_COUNT=1
{self._copilot_action_feedback_helpers()}
configure_copilot_command
config_code=$?
if [ "$config_code" -ne 0 ]; then
  exit "$config_code"
fi
is_arc_task=0
if is_arc_runtime; then
  is_arc_task=1
fi

invoke_copilot() {{
  local attempt_prompt_file="$1"
  local attempt_log_file="$2"
  local otel_file="${{attempt_log_file%.jsonl}}.otel"
  shift 2
  : > "$otel_file"
  COPILOT_ALLOW_ALL=true COPILOT_AUTO_UPDATE=false \\
    COPILOT_OTEL_ENABLED=true COPILOT_OTEL_EXPORTER_TYPE=file \\
    COPILOT_OTEL_FILE_EXPORTER_PATH="$otel_file" \\
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false \\
    COPILOT_AGENT_ERROR_CODES_TO_RETRY="${{COPILOT_AGENT_ERROR_CODES_TO_RETRY:-401,429,500,502,503,504,529}}" \\
    "${{copilot_command[@]}}" --model "$model" {effort_flag}{context_flag}--output-format json --allow-all \\
    --allow-all-paths --allow-all-urls --no-custom-instructions \\
    {tool_flags} \\
    --disable-builtin-mcps --secret-env-vars={COPILOT_SECRET_ENV_VARS} \\
    --stream off --silent --no-auto-update \\
    --log-dir "$copilot_log_dir" "$@" -p "$(cat "$attempt_prompt_file")" 2>&1 | sed -E "$redact_arc_token; s|$ARCCTL_SESSION_TOKEN|<arc-token-redacted>|g" > "$attempt_log_file"
  code=${{PIPESTATUS[0]}}
  append_copilot_telemetry "$attempt_log_file" "$otel_file"
  return "$code"
}}

run_copilot_with_api_retries() {{
  local attempt_prompt_file="$1"
  local attempt_log_base="$2"
  local max_api_retries="${{META_DUO_COPILOT_API_RETRIES:-288}}"
  case "$max_api_retries" in ''|*[!0-9]*) max_api_retries=288 ;; esac
  local retry=0
  local code=0
  while true; do
    local attempt_log_file="$attempt_log_base"
    if [ "$retry" -gt 0 ]; then
      attempt_log_file="${{attempt_log_base%.jsonl}}-api-retry-$retry.jsonl"
    fi
    if [ -n "$session_id" ]; then
      invoke_copilot "$attempt_prompt_file" "$attempt_log_file" --resume="$session_id"
    else
      invoke_copilot "$attempt_prompt_file" "$attempt_log_file"
    fi
    code=$?
    cat "$attempt_log_file" >> "$log_file"
    new_session_id="$(extract_copilot_session_id "$attempt_log_file")"
    if [ -n "$new_session_id" ]; then
      session_id="$new_session_id"
    fi
    if [ "$code" -eq 0 ]; then
      return 0
    fi
    if ! is_retryable_copilot_api_error "$attempt_log_file"; then
      return "$code"
    fi
    if [ "$retry" -ge "$max_api_retries" ]; then
      append_copilot_api_retry_exhausted_event "$log_file" "$max_api_retries" "$code"
      return "$code"
    fi
    retry=$((retry + 1))
    delay="$(copilot_api_retry_sleep_seconds "$retry")"
    append_copilot_api_retry_event "$log_file" "$retry" "$max_api_retries" "$code" "$delay"
    sleep "$delay"
  done
}}

# Startup jitter to spread Copilot model-load API hits across concurrent workers
sleep "$(awk 'BEGIN{{srand(); print int(rand()*30)}}')"

: > "$log_file"
session_id=""
attempt_log="{shlex.quote(session_dir)}/copilot-attempt-0.jsonl"
run_copilot_with_api_retries "$prompt_file" "$attempt_log"
code=$?
if [ "$code" -ne 0 ]; then
  exit "$code"
fi
if [ "$is_arc_task" -eq 1 ]; then
  if has_recorded_arc_action "$state_file"; then
    exit 0
  fi
elif [ "$non_arc_continuation" -ne 1 ]; then
  exit 0
fi

max_no_action_reprompts={COPILOT_NO_ACTION_REPROMPTS}
no_action_attempt=1
while [ "$no_action_attempt" -le "$max_no_action_reprompts" ]; do
  feedback_file="{shlex.quote(session_dir)}/copilot-no-action-feedback-$no_action_attempt.txt"
  attempt_log="{shlex.quote(session_dir)}/copilot-attempt-$no_action_attempt.jsonl"
  write_no_action_feedback "$feedback_file"
  if [ "$is_arc_task" -eq 1 ]; then
    append_no_action_feedback_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
  else
    append_optimization_continuation_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
  fi
  run_copilot_with_api_retries "$feedback_file" "$attempt_log"
  code=$?
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
  if [ "$is_arc_task" -eq 1 ]; then
    if has_recorded_arc_action "$state_file"; then
      exit 0
    fi
  fi
  no_action_attempt=$((no_action_attempt + 1))
done

if [ "$is_arc_task" -eq 1 ]; then
  printf '{{"type":"environment.feedback","data":{{"event":"no_valid_arc_action_reprompts_exhausted","message":"No valid ARC action was recorded after all environment feedback attempts.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
  exit 1
fi
printf '{{"type":"environment.feedback","data":{{"event":"optimization_continuations_completed","message":"Completed all configured optimization continuation passes.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
exit 0
"""
        await self.exec_as_agent(
            environment,
            command=run_cmd,
            env={**env, **benchmark_runtime.token_env(0)},
        )



class CollaborationAgent(CopilotAgent):
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        if os.environ["MULTIAGENT_N_AGENT"] == "1" and os.environ["RELEASE_TASK"] != "mnist":
            await CopilotSingleAgent.run(self, instruction, environment, context)
        else:
            await super().run(instruction, environment, context)
