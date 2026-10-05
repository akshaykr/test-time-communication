"""ARC-AGI-3 runtime hook for Harbor CLI agents.

The generic agent adapters can include this hook safely for every benchmark:
it only activates inside task containers that provide the ARC-specific `arcctl`
command.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shlex


@dataclass(frozen=True)
class ArcAgi3Runtime:
    session_ids: tuple[str, ...]
    tokens: tuple[str, ...]
    session_bindings_json: str

    @classmethod
    def for_agent_ids(cls, agent_ids: list[str]) -> "ArcAgi3Runtime":
        tokens = []
        bindings = {}
        for agent_id in agent_ids:
            token = secrets.token_urlsafe(24)
            tokens.append(token)
            bindings[token] = agent_id
        return cls(
            session_ids=tuple(agent_ids),
            tokens=tuple(tokens),
            session_bindings_json=json.dumps(bindings, sort_keys=True),
        )

    def token_env(self, index: int) -> dict[str, str]:
        return {
            "ARCCTL_SESSION_ID": self.session_ids[index],
            "ARCCTL_SESSION_TOKEN": self.tokens[index],
        }

    def indexed_token_env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for index, (session_id, token) in enumerate(
            zip(self.session_ids, self.tokens, strict=True)
        ):
            values[self.indexed_session_id_var(index)] = session_id
            values[self.indexed_token_var(index)] = token
        return values

    def indexed_token_var(self, index: int) -> str:
        return f"ARCCTL_SESSION_TOKEN_{index}"

    def indexed_session_id_var(self, index: int) -> str:
        return f"ARCCTL_SESSION_ID_{index}"

    def shell_token_prefix(self, index: int) -> str:
        unsets = " ".join(
            f"-u {self.indexed_session_id_var(token_index)} "
            f"-u {self.indexed_token_var(token_index)}"
            for token_index in range(len(self.tokens))
        )
        return (
            f'ARCCTL_SESSION_ID="${{{self.indexed_session_id_var(index)}}}" '
            f'ARCCTL_SESSION_TOKEN="${{{self.indexed_token_var(index)}}}" '
            f"env {unsets}"
        )

    def setup_env(self) -> dict[str, str]:
        return {"ARCCTL_SESSION_BINDINGS": self.session_bindings_json}

    def instruction_suffix(self) -> str:
        """Barrier instructions appended to the task prompt when the sync barrier
        is armed (ARCCTL_SYNC_FREQ > 0). Empty otherwise, so non-barrier and
        non-ARC runs are byte-for-byte unaffected."""
        raw = os.environ.get("ARCCTL_SYNC_FREQ", "0").strip()
        try:
            freq = int(raw)
        except ValueError:
            freq = 0
        if freq <= 0 or len(self.session_ids) == 1:
            return ""
        path = (
            Path(__file__).resolve().parent
            / "benchmarks"
            / "arc-agi-3"
            / "instruction_barrier.md"
        )
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return "\n\n" + text + "\n"

    def setup_command(self) -> str:
        wait_for_server = """
import http.client
import json
import os
import time

host = os.environ.get("ARCCTL_HOST", "127.0.0.1")
port = int(os.environ.get("ARCCTL_PORT", "8765"))
deadline = time.time() + float(os.environ.get("ARCCTL_SERVER_TIMEOUT", "120"))
while time.time() < deadline:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=1)
        conn.request(
            "POST",
            "/",
            body='{"command":"ping"}',
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8") or "{}")
        conn.close()
        if data.get("ok"):
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.2)
raise SystemExit("arcctl daemon did not become ready")
"""
        commands = [
            'if command -v arcctl >/dev/null 2>&1 && '
            'arcctl --help 2>/dev/null | grep -q "ARC-AGI-3"; then',
            f"python3 -c {shlex.quote(wait_for_server)}",
            'arc_game="${ARCCTL_DEFAULT_GAME:-ls20}"',
            ': "${ARCCTL_SESSION_BINDINGS:?missing ARC session bindings}"',
            'extra_arc_config=""',
            'if [ -n "${ARCCTL_FIXED_LEVEL_BUDGET:-}" ]; then',
            '  extra_arc_config="$extra_arc_config --fixed-level-budget $ARCCTL_FIXED_LEVEL_BUDGET"',
            "fi",
            'if [ -n "${ARCCTL_TARGET_LEVELS:-}" ]; then',
            '  extra_arc_config="$extra_arc_config --target-levels $ARCCTL_TARGET_LEVELS"',
            "fi",
            'if [ -n "${ARCCTL_SYNC_FREQ:-}" ]; then',
            '  extra_arc_config="$extra_arc_config --sync-freq $ARCCTL_SYNC_FREQ"',
            "fi",
            "arcctl configure-sessions "
            '--bindings "$ARCCTL_SESSION_BINDINGS" '
            '--game "$arc_game" $extra_arc_config >/dev/null',
            "arcctl ping >/dev/null",
            "fi",
        ]
        return "(\nset -e\n" + "\n".join(commands) + "\n)"
