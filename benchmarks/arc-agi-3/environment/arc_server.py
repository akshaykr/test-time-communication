#!/usr/bin/env python3
"""ARC-AGI-3 sidecar controller for Harbor CLI agents.

The agent-visible `arcctl` command is a thin HTTP client. This sidecar owns the
stateful ARC environment objects and the `arc_agi` package so benchmark source
files are not present in the main agent container.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any


HOST = os.environ.get("ARCCTL_HOST", "0.0.0.0")
PORT = int(os.environ.get("ARCCTL_PORT", "8765"))
STATE_DIR = Path(os.environ.get("ARCCTL_STATE_DIR", "/logs/agent/arc_agi3_runs"))
DEFAULT_GAME = os.environ.get("ARCCTL_DEFAULT_GAME", "ls20")
DEFAULT_MODE = "normal"
ACTION_BUDGET_MULTIPLIER = float(
    os.environ.get("ARCCTL_ACTION_BUDGET_MULTIPLIER", "5.0")
)
FALLBACK_MAX_ACTIONS = int(os.environ.get("ARCCTL_FALLBACK_MAX_ACTIONS", "10"))
SESSION_TOKEN_ENV = "ARCCTL_SESSION_TOKEN"
SESSION_BINDINGS_ENV = "ARCCTL_SESSION_BINDINGS"
SAFE_GAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}")
    return value


def nonnegative_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def optional_positive_int_payload(payload: dict[str, Any], name: str) -> int | None:
    raw = payload.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def nonnegative_int_payload(
    payload: dict[str, Any], name: str, default: int = 0
) -> int:
    raw = payload.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


FIXED_LEVEL_BUDGET = positive_int_env("ARCCTL_FIXED_LEVEL_BUDGET")
TARGET_LEVELS = nonnegative_int_env("ARCCTL_TARGET_LEVELS")
# Number of sync barriers per level (0 disables the barrier entirely). When > 0,
# the per-level sync interval is ceil(level_budget / SYNC_FREQ); agents are paced
# in global (level, round) lockstep so none races ahead of its teammates.
SYNC_FREQ = nonnegative_int_env("ARCCTL_SYNC_FREQ", 0)
# Seconds of silence after which a session drops out of the barrier's live set, so an
# agent whose headless CLI has exited (it ends on a no-tool-call turn) with budget still
# left cannot pin the barrier closed for teammates forever. 0 disables the timeout.
BARRIER_IDLE_TIMEOUT = nonnegative_int_env("ARCCTL_BARRIER_IDLE_TIMEOUT", 600)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def sanitize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())
    return cleaned or "session0"


def load_session_bindings() -> dict[str, str]:
    raw = os.environ.get(SESSION_BINDINGS_ENV, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{SESSION_BINDINGS_ENV} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{SESSION_BINDINGS_ENV} must be a JSON object")
    bindings = {}
    for token, session_id in parsed.items():
        token_text = str(token).strip()
        if not token_text:
            raise RuntimeError(f"{SESSION_BINDINGS_ENV} contains an empty token")
        bindings[token_text] = sanitize_session_id(str(session_id))
    if not bindings:
        raise RuntimeError(f"{SESSION_BINDINGS_ENV} must contain at least one binding")
    return bindings


def normalize_session_bindings(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"session bindings must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("session bindings must be a JSON object")
    bindings = {}
    for token, session_id in value.items():
        token_text = str(token).strip()
        if not token_text:
            raise ValueError("session bindings contain an empty token")
        bindings[token_text] = sanitize_session_id(str(session_id))
    if not bindings:
        raise ValueError("session bindings must contain at least one binding")
    return bindings


def validate_game_id(game: str) -> str:
    game = game.strip()
    if not SAFE_GAME_RE.match(game):
        raise ValueError(f"Unsafe ARC game id: {game!r}")
    return game


@dataclass
class Session:
    session_id: str
    game: str
    mode: str
    env: Any
    observation: Any
    baseline_actions: list[int]
    level_action_budgets: list[int]
    max_actions: int
    action_budget_multiplier: float = ACTION_BUDGET_MULTIPLIER
    fixed_level_budget: int | None = FIXED_LEVEL_BUDGET
    target_levels: int = TARGET_LEVELS
    steps: int = 0
    level_action_count: int = 0
    last_levels_completed: int = 0
    level_clear_actions: dict[int, int] = field(default_factory=dict)
    target_clear_actions: int | None = None
    target_reached_at: str | None = None
    target_reached_steps: int | None = None
    last_seen: float = field(default_factory=time.monotonic)


class ArcController:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.arcades: dict[str, Any] = {}
        self.scorecard_ids: dict[str, str] = {}
        self.session_bindings: dict[str, str] | None = (
            load_session_bindings()
            if os.environ.get(SESSION_BINDINGS_ENV, "").strip()
            else None
        )
        self.default_game = validate_game_id(DEFAULT_GAME)
        self.fixed_level_budget = FIXED_LEVEL_BUDGET
        self.target_levels = TARGET_LEVELS
        self.sync_freq = SYNC_FREQ
        self.barrier_idle_timeout = BARRIER_IDLE_TIMEOUT
        self.lock = threading.Lock()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.persist_session_policy()

    def arcade(self, mode: str) -> Any:
        normalized = self.validate_mode(mode)
        if normalized not in self.arcades:
            import arc_agi
            from arc_agi import OperationMode

            self.arcades[normalized] = arc_agi.Arcade(
                operation_mode=OperationMode.NORMAL
            )
        return self.arcades[normalized]

    def scorecard_id(self, mode: str) -> str:
        normalized = self.validate_mode(mode)
        if normalized not in self.scorecard_ids:
            self.scorecard_ids[normalized] = self.arcade(normalized).open_scorecard(
                tags=["agent-collab", "arcctl", "arc-agi-3"]
            )
        return self.scorecard_ids[normalized]

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload.get("command")
        with self.lock:
            if "mode" in payload:
                self.validate_mode(str(payload.get("mode") or DEFAULT_MODE))
            if command == "ping":
                return {"ok": True, "time": utc_now()}
            if command == "configure-sessions":
                return self.configure_sessions(payload)
            if command == "export":
                raise ValueError("ARC sidecar export is disabled during agent runs")
            if command == "list-games":
                return self.list_games(payload)
            if command == "reset":
                return self.reset(payload)
            if command == "observe":
                return self.observe(payload)
            if command == "actions":
                return self.actions(payload)
            if command == "step":
                return self.step(payload)
            if command == "score":
                return self.score(payload)
            if command == "barrier":
                return self.barrier(payload)
        raise ValueError(f"Unknown command: {command}")

    def validate_mode(self, mode: str) -> str:
        normalized = mode.lower()
        if normalized != DEFAULT_MODE:
            raise ValueError(
                "ARC-AGI-3 benchmark runs only allow normal mode; "
                "offline and competition modes are disabled because they create "
                "separate ARC scorecards."
            )
        return normalized

    def session_id(self, payload: dict[str, Any]) -> str:
        if self.session_bindings is None:
            raise ValueError("ARC session bindings have not been configured")
        token = str(payload.get("session_token") or "")
        if not token:
            raise ValueError(f"{SESSION_TOKEN_ENV} is required for this ARC session")
        session_id = self.session_bindings.get(token)
        if session_id is None:
            raise ValueError("Unknown ARC session token")
        caller_session_id = payload.get("agent_id")
        if caller_session_id != session_id:
            raise ValueError(
                "ARCCTL_SESSION_ID/AGENT_ID does not match this ARC session token"
            )
        return session_id

    def session_policy(self) -> dict[str, Any]:
        expected_sessions = self.expected_session_ids()
        return {
            "updated_at": utc_now(),
            "configured": self.session_bindings is not None,
            "session_binding_required": True,
            "expected_session_ids": expected_sessions,
            "expected_session_count": len(expected_sessions),
            "expected_game": self.default_game,
            "expected_mode": DEFAULT_MODE,
            "action_budget_multiplier": ACTION_BUDGET_MULTIPLIER,
            "fixed_level_budget": self.fixed_level_budget,
            "target_levels": self.target_levels,
            "sync_freq": self.sync_freq,
        }

    def persist_session_policy(self) -> None:
        (STATE_DIR / "session_policy.json").write_text(
            json.dumps(self.session_policy(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def configure_sessions(self, payload: dict[str, Any]) -> dict[str, Any]:
        bindings = normalize_session_bindings(payload.get("bindings"))
        game = validate_game_id(str(payload.get("game") or self.default_game))
        fixed_level_budget = optional_positive_int_payload(
            payload, "fixed_level_budget"
        )
        if "fixed_level_budget" not in payload:
            fixed_level_budget = self.fixed_level_budget
        target_levels = nonnegative_int_payload(
            payload, "target_levels", default=self.target_levels
        )
        sync_freq = nonnegative_int_payload(
            payload, "sync_freq", default=self.sync_freq
        )
        if self.session_bindings is not None:
            if (
                self.session_bindings == bindings
                and self.default_game == game
                and self.fixed_level_budget == fixed_level_budget
                and self.target_levels == target_levels
                and self.sync_freq == sync_freq
            ):
                return {
                    "ok": True,
                    "configured": True,
                    "idempotent": True,
                    "session_policy": self.session_policy(),
                }
            raise ValueError("ARC session bindings are already configured")
        if self.sessions:
            raise ValueError("ARC sessions already exist; cannot configure bindings")
        self.session_bindings = bindings
        self.default_game = game
        self.fixed_level_budget = fixed_level_budget
        self.target_levels = target_levels
        self.sync_freq = sync_freq
        self.persist_session_policy()
        return {
            "ok": True,
            "configured": True,
            "idempotent": False,
            "session_policy": self.session_policy(),
        }

    def list_games(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = self.validate_mode(str(payload.get("mode") or DEFAULT_MODE))
        environments = self.arcade(mode).get_environments()
        games = []
        for environment in environments:
            if hasattr(environment, "model_dump"):
                raw = environment.model_dump()
            elif hasattr(environment, "dict"):
                raw = environment.dict()
            else:
                raw = {
                    name: getattr(environment, name)
                    for name in ("game_id", "id", "title", "description")
                    if hasattr(environment, name)
                }
            raw["action_budget"] = self.action_budget_from_info(environment)
            games.append(raw)
        totals = [
            int(game["action_budget"]["max_actions"])
            for game in games
            if game.get("action_budget", {}).get("max_actions") is not None
        ]
        return {
            "ok": True,
            "count": len(games),
            "action_budget_multiplier": ACTION_BUDGET_MULTIPLIER,
            "fixed_level_budget": self.fixed_level_budget,
            "target_levels": self.target_levels,
            "fallback_max_actions": FALLBACK_MAX_ACTIONS,
            "action_budget_range": {
                "min": min(totals) if totals else None,
                "max": max(totals) if totals else None,
            },
            "games": games,
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = self.session_id(payload)
        mode = self.validate_mode(str(payload.get("mode") or DEFAULT_MODE))
        session = self.sessions.get(session_id)
        game = str(payload.get("game") or (session.game if session is not None else self.default_game))
        if game != self.default_game:
            raise ValueError(
                f"This ARC-AGI-3 benchmark run is locked to game {self.default_game!r}; "
                f"got {game!r}"
            )
        if session is not None and session.game == game and session.mode == mode:
            result = self.reset_existing_session(session, payload)
            self.persist(session, "reset", result)
            return result

        env = self.arcade(mode).make(
            game,
            render_mode=None,
            scorecard_id=self.scorecard_id(mode),
        )
        observation = env.reset()
        if observation is None:
            raise RuntimeError(f"Reset returned no observation for game {game}")
        baseline_actions = self.baseline_actions(env)
        level_action_budgets = self.level_action_budgets(baseline_actions)
        session = Session(
            session_id=session_id,
            game=game,
            mode=mode,
            env=env,
            observation=observation,
            baseline_actions=baseline_actions,
            level_action_budgets=level_action_budgets,
            max_actions=self.max_actions_for(level_action_budgets),
            fixed_level_budget=self.fixed_level_budget,
            target_levels=self.target_levels,
            last_levels_completed=int(getattr(observation, "levels_completed", 0) or 0),
        )
        self.sessions[session_id] = session
        result = self.summarize(session, payload, event="reset")
        self.persist(session, "reset", result)
        return result

    def reset_existing_session(
        self, session: Session, payload: dict[str, Any]
    ) -> dict[str, Any]:
        from arcengine import GameAction

        if self.target_reached(session):
            return self.target_reached_response(session, payload)
        budget_error = self.action_budget_error(session)
        if budget_error is not None:
            result = self.summarize(session, payload, event="budget_exhausted")
            result.update({"ok": False, "error": budget_error})
            return result
        blocked = self._barrier_block_response(session, payload)
        if blocked is not None:
            return blocked
        observation = session.env.step(GameAction.RESET)
        if observation is None:
            raise RuntimeError(f"Reset returned no observation for game {session.game}")
        self.record_action(session)
        session.observation = observation
        self.sync_level_progress(session)
        return self.summarize(session, payload, event="reset")

    def existing_or_new_session(self, payload: dict[str, Any]) -> Session:
        self.validate_mode(str(payload.get("mode") or DEFAULT_MODE))
        session_id = self.session_id(payload)
        session = self.sessions.get(session_id)
        if session is None:
            self.reset({**payload, "command": "reset"})
            session = self.sessions[session_id]
        session.last_seen = time.monotonic()
        return session

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.existing_or_new_session(payload)
        result = self.summarize(session, payload, event="observe")
        self.persist(session, "observe", result)
        return result

    def actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.existing_or_new_session(payload)
        actions = [] if self.target_reached(session) else self.available_action_labels(session.observation)
        result = {
            "ok": True,
            "session_id": session.session_id,
            "game": session.game,
            "steps": session.steps,
            "action_budget": self.budget_status(session),
            "available_actions": actions,
            "target_reached": self.target_reached(session),
            "target_clear_actions": session.target_clear_actions,
        }
        barrier = self._barrier_info(session)
        if barrier is not None:
            result["barrier"] = barrier
        self.persist(session, "actions", result)
        return result

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        from arcengine import GameAction

        session = self.existing_or_new_session(payload)
        action_name = self.normalize_action(str(payload.get("action") or ""))
        if action_name == "RESET":
            result = self.reset_existing_session(session, payload)
            self.persist(session, "step", result)
            return result
        if not hasattr(GameAction, action_name):
            raise ValueError(f"Unknown action {action_name!r}")
        if self.target_reached(session):
            result = self.target_reached_response(session, payload)
            self.persist(session, "target_reached", result)
            return result
        budget_error = self.action_budget_error(session)
        if budget_error is not None:
            result = self.summarize(session, payload, event="budget_exhausted")
            result.update({"ok": False, "error": budget_error})
            self.persist(session, "budget_exhausted", result)
            return result
        blocked = self._barrier_block_response(session, payload)
        if blocked is not None:
            return blocked
        action = getattr(GameAction, action_name)
        data = payload.get("data")
        reasoning = payload.get("reasoning")
        observation = session.env.step(action, data=data, reasoning=reasoning)
        if observation is None:
            raise RuntimeError(f"Step {action_name} returned no observation")
        self.record_action(session)
        session.observation = observation
        self.sync_level_progress(session)
        result = self.summarize(session, payload, event="step")
        self.persist(session, "step", result)
        return result

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.existing_or_new_session(payload)
        result = self.snapshot(session)
        result.update({"ok": True, "event": "score"})
        self.persist(session, "score", result)
        return result

    def barrier(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Lightweight sync-barrier status poll: returns only the barrier object,
        # not the full snapshot (no grid frame, budget, or scorecard payload),
        # so a blocked agent can re-check cheaply without re-pulling observations.
        session = self.existing_or_new_session(payload)
        return {
            "ok": True,
            "event": "barrier",
            "session_id": session.session_id,
            "game": session.game,
            "steps": session.steps,
            "barrier": self._barrier_info(session),
        }

    def normalize_action(self, action: str) -> str:
        normalized = action.strip().upper()
        if normalized.isdigit():
            return f"ACTION{normalized}"
        if normalized.startswith("ACTION") or normalized == "RESET":
            return normalized
        raise ValueError(f"Action must be ACTIONn, n, or RESET; got {action!r}")

    def available_action_labels(self, observation: Any) -> list[dict[str, Any]]:
        raw_actions = getattr(observation, "available_actions", []) or []
        labels = []
        for action in raw_actions:
            value = int(action)
            labels.append({"id": value, "name": f"ACTION{value}"})
        return labels

    def baseline_actions(self, env: Any) -> list[int]:
        info = getattr(env, "info", None) or getattr(env, "environment_info", None)
        raw = getattr(info, "baseline_actions", None) or []
        return [int(action_count) for action_count in raw]

    def level_action_budgets(self, baseline_actions: list[int]) -> list[int]:
        if self.fixed_level_budget is not None:
            level_count = len(baseline_actions)
            if self.target_levels > 0:
                level_count = max(level_count, self.target_levels)
            return [self.fixed_level_budget for _ in range(max(1, level_count))]
        return [
            int(math.ceil(action_count * ACTION_BUDGET_MULTIPLIER))
            for action_count in baseline_actions
        ]

    def max_actions_for(self, level_action_budgets: list[int]) -> int:
        if not level_action_budgets:
            return self.fixed_level_budget or FALLBACK_MAX_ACTIONS
        if self.target_levels > 0:
            return sum(level_action_budgets[: self.target_levels])
        return sum(level_action_budgets)

    def action_budget_from_info(self, info: Any) -> dict[str, Any]:
        raw = getattr(info, "baseline_actions", None) or []
        baseline_actions = [int(action_count) for action_count in raw]
        level_budgets = self.level_action_budgets(baseline_actions)
        return {
            "baseline_actions": baseline_actions,
            "multiplier": ACTION_BUDGET_MULTIPLIER,
            "fixed_level_budget": self.fixed_level_budget,
            "target_levels": self.target_levels,
            "level_action_budgets": level_budgets,
            "max_actions": self.max_actions_for(level_budgets),
            "uses_fallback": not bool(level_budgets),
        }

    def budget_status(
        self, session: Session, scorecard: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        scorecard = scorecard or self.scorecard_summary(session)
        actions_used = self.cumulative_actions_used(session, scorecard)
        current_level = int(getattr(session.observation, "levels_completed", 0) or 0)
        current_level_budget = None
        if 0 <= current_level < len(session.level_action_budgets):
            current_level_budget = session.level_action_budgets[current_level]
        total_remaining = max(0, session.max_actions - actions_used)
        level_remaining = (
            max(0, current_level_budget - session.level_action_count)
            if current_level_budget is not None
            else total_remaining
        )
        target_reached = self.target_reached(session)
        if target_reached:
            total_remaining = 0
            level_remaining = 0
        exhausted = (not target_reached) and (
            total_remaining <= 0
            or (
                current_level_budget is not None
                and session.level_action_count >= current_level_budget
            )
        )
        return {
            "baseline_actions": session.baseline_actions,
            "multiplier": session.action_budget_multiplier,
            "fixed_level_budget": session.fixed_level_budget,
            "target_levels": session.target_levels,
            "level_action_budgets": session.level_action_budgets,
            "max_actions": session.max_actions,
            "actions_used": actions_used,
            "actions_remaining": total_remaining,
            "session_actions_used": session.steps,
            "current_level": current_level,
            "current_level_budget": current_level_budget,
            "current_level_actions_used": session.level_action_count,
            "current_level_actions_remaining": level_remaining,
            "uses_fallback": not bool(session.level_action_budgets),
            "target_reached": target_reached,
            "target_clear_actions": session.target_clear_actions,
            "target_reached_steps": session.target_reached_steps,
            "exhausted": exhausted,
        }

    def target_reached(self, session: Session) -> bool:
        return session.target_levels > 0 and session.target_clear_actions is not None

    def target_reached_response(
        self, session: Session, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = self.summarize(session, payload, event="target_reached")
        result.update(
            {
                "message": (
                    "ARC target already reached; no further ARC actions are charged "
                    "for this session."
                )
            }
        )
        return result

    def action_budget_error(self, session: Session) -> str | None:
        budget = self.budget_status(session)
        if not budget["exhausted"]:
            return None
        current_level_budget = budget["current_level_budget"]
        if current_level_budget is not None and budget["current_level_actions_remaining"] <= 0:
            return (
                "ARC-AGI-3 action budget exhausted for "
                f"level {budget['current_level']}: "
                f"{budget['current_level_actions_used']}/{current_level_budget} actions used"
            )
        return (
            "ARC-AGI-3 action budget exhausted: "
            f"{budget['actions_used']}/{budget['max_actions']} actions used"
        )

    def cumulative_actions_used(
        self, session: Session, scorecard: dict[str, Any]
    ) -> int:
        if len(self.expected_session_ids()) > 1:
            return session.steps
        card = (scorecard.get("cards") or {}).get(session.game)
        if not isinstance(card, dict):
            return session.steps
        return max(session.steps, int(card.get("total_actions", session.steps) or 0))

    def expected_session_ids(self) -> list[str]:
        if self.session_bindings is None:
            return []
        return sorted(set(self.session_bindings.values()))

    def record_action(self, session: Session) -> None:
        session.steps += 1
        session.level_action_count += 1

    def sync_level_progress(self, session: Session) -> None:
        current_level = int(getattr(session.observation, "levels_completed", 0) or 0)
        previous_level = session.last_levels_completed
        if current_level > previous_level:
            for completed_level in range(previous_level + 1, current_level + 1):
                session.level_clear_actions.setdefault(completed_level, session.steps)
            if (
                session.target_levels > 0
                and current_level >= session.target_levels
                and session.target_clear_actions is None
            ):
                session.target_clear_actions = session.steps
                session.target_reached_steps = session.steps
                session.target_reached_at = utc_now()
        if current_level != session.last_levels_completed:
            session.level_action_count = 0
            session.last_levels_completed = current_level

    # --- Sync barrier (multi-agent pacing) -------------------------------
    def _session_level(self, session: Session) -> int:
        return int(getattr(session.observation, "levels_completed", 0) or 0)

    def _session_state(self, session: Session) -> str:
        state = getattr(session.observation, "state", None)
        return getattr(state, "value", str(state))

    def _level_budget(self, session: Session, level: int) -> int | None:
        budgets = session.level_action_budgets
        if 0 <= level < len(budgets):
            return int(budgets[level])
        return None

    def _sync_interval(self, session: Session, level: int) -> int | None:
        if self.sync_freq <= 0:
            return None
        budget = self._level_budget(session, level)
        if not budget:
            return None
        return max(1, math.ceil(budget / self.sync_freq))

    def _phase(self, session: Session) -> tuple[int, int]:
        level = self._session_level(session)
        interval = self._sync_interval(session, level)
        rnd = session.level_action_count // interval if interval else 0
        return (level, rnd)

    def _is_live(self, session: Session) -> bool:
        # An agent still able to make progress toward the next phase. Finished,
        # dead, or level-budget-exhausted agents drop out so they never block.
        if self.target_reached(session):
            return False
        if self._session_state(session) == "GAME_OVER":
            return False
        level = self._session_level(session)
        budget = self._level_budget(session, level)
        if budget is not None and session.level_action_count >= budget:
            return False
        # An agent whose CLI has exited (headless copilot ends on a no-tool-call turn)
        # can sit non-terminal with budget left; if it stayed "live" it would pin the
        # barrier closed forever for teammates still ahead of it. Drop it once it has
        # been silent past the idle timeout — its next request re-marks it live.
        if (
            self.barrier_idle_timeout > 0
            and time.monotonic() - session.last_seen > self.barrier_idle_timeout
        ):
            return False
        return True

    def _live_sessions(self) -> list[Session]:
        return [s for s in self.sessions.values() if self._is_live(s)]

    def _barrier_info(self, session: Session) -> dict[str, Any] | None:
        if self.sync_freq <= 0:
            return None
        my_level = self._session_level(session)
        interval = self._sync_interval(session, my_level)
        my_phase = self._phase(session)
        others = [
            s for s in self._live_sessions() if s.session_id != session.session_id
        ]
        min_phase = min([my_phase] + [self._phase(s) for s in others])
        is_open = (not others) or (my_phase <= min_phase)
        return {
            "sync_freq": self.sync_freq,
            "sync_every": interval,
            "open": is_open,
            "your_level": my_phase[0],
            "your_level_actions": session.level_action_count,
            "your_round": my_phase[1],
            "blocking_level": min_phase[0],
            "blocking_round": min_phase[1],
            "live_agents": len(others) + (1 if self._is_live(session) else 0),
            "peers": [
                {
                    "level": self._session_level(s),
                    "level_actions": s.level_action_count,
                }
                for s in others
            ],
        }

    def _barrier_message(self, barrier: dict[str, Any]) -> str:
        return (
            f"SYNC BARRIER CLOSED. Your phase is level {barrier['your_level']} / "
            f"round {barrier['your_round']}, but a teammate is behind (team is at "
            f"level {barrier['blocking_level']} / round {barrier['blocking_round']}). "
            "No action was taken and no budget charged — your budget is intact, "
            "and this is not an error or a stopping condition. Only `arcctl step` and "
            "`arcctl RESET` are blocked; everything else still works — any shell "
            "command, all file I/O, and the read-only `arcctl observe` / `actions` / "
            "`score`. Coordinate with your team, then re-check with `arcctl barrier` "
            "(lightweight status poll — no grid frame). Your next `arcctl step` succeeds "
            "automatically once teammates reach your phase (or leave the game)."
        )

    def _barrier_block_response(
        self, session: Session, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        # Returns a no-charge refusal response when the barrier is closed for this
        # session, else None (the caller proceeds to take the action normally).
        barrier = self._barrier_info(session)
        if barrier is None or barrier["open"]:
            return None
        result = self.summarize(session, payload, event="barrier_wait")
        result.update(
            {
                "ok": False,
                "barrier_wait": True,
                "message": self._barrier_message(barrier),
            }
        )
        return result

    def summarize(
        self, session: Session, payload: dict[str, Any], *, event: str
    ) -> dict[str, Any]:
        snapshot = self.snapshot(session)
        snapshot.update(
            {
                "ok": True,
                "event": event,
                "frames": self.frame_summary(session.observation, payload),
            }
        )
        return snapshot

    def snapshot(self, session: Session) -> dict[str, Any]:
        observation = session.observation
        state = getattr(observation, "state", None)
        state_value = getattr(state, "value", str(state))
        levels_completed = int(getattr(observation, "levels_completed", 0) or 0)
        win_levels = int(getattr(observation, "win_levels", 0) or 0)
        scorecard = self.scorecard_summary(session)
        official_score = float(scorecard.get("official_score", 0.0) or 0.0)
        official_reward = float(scorecard.get("official_reward", official_score / 100.0))
        scorecard_levels = int(scorecard.get("levels_completed", levels_completed) or 0)
        total_levels = int(scorecard.get("total_levels", win_levels) or 0)
        progress_reward = scorecard_levels / total_levels if total_levels else 0.0
        action_input = getattr(observation, "action_input", None)
        action = None
        data = None
        if action_input is not None:
            action_id = getattr(action_input, "id", None)
            action = getattr(action_id, "name", str(action_id))
            data = getattr(action_input, "data", None)
        snapshot = {
            "session_id": session.session_id,
            "game": session.game,
            "mode": session.mode,
            "steps": session.steps,
            "state": state_value,
            "levels_completed": levels_completed,
            "scorecard_levels_completed": scorecard_levels,
            "win_levels": win_levels,
            "target_levels": session.target_levels,
            "target_reached": self.target_reached(session),
            "target_clear_actions": session.target_clear_actions,
            "target_reached_at": session.target_reached_at,
            "target_reached_steps": session.target_reached_steps,
            "level_clear_actions": {
                str(level): actions
                for level, actions in sorted(session.level_clear_actions.items())
            },
            "reward": official_reward,
            "official_score": official_score,
            "official_reward": official_reward,
            "progress_reward": progress_reward,
            "action_budget": self.budget_status(session, scorecard),
            "scorecard": scorecard,
            "available_actions": self.available_action_labels(observation),
            "last_action": action,
            "last_action_data": data,
            "updated_at": utc_now(),
        }
        barrier = self._barrier_info(session)
        if barrier is not None:
            snapshot["barrier"] = barrier
        return snapshot

    def scorecard_summary(self, session: Session) -> dict[str, Any]:
        scorecard_id = getattr(session.env, "scorecard_id", None)
        manager = getattr(session.env, "scorecard_manager", None)
        scorecards = getattr(manager, "scorecards", {}) if manager is not None else {}
        scorecard = scorecards.get(scorecard_id)
        if scorecard is None:
            return {
                "card_id": scorecard_id,
                "won": 0,
                "played": 0,
                "win_rate": 0.0,
                "levels_completed": 0,
                "total_actions": 0,
                "cards": {},
            }

        dump = scorecard.model_dump()
        dump.pop("api_key", None)
        dump.pop("opaque", None)
        won = int(dump.get("won", 0) or 0)
        played = int(dump.get("played", 0) or 0)
        official = self.official_scorecard_summary(scorecard)
        official_score = float(official.get("score", 0.0) or 0.0)
        total_levels_completed = int(
            official.get("total_levels_completed", dump.get("levels_completed", 0)) or 0
        )
        total_levels = int(official.get("total_levels", 0) or 0)
        cards = {}
        for game_id, card in (dump.get("cards") or {}).items():
            cards[game_id] = {
                "total_plays": card.get("total_plays", 0),
                "states": card.get("states", []),
                "levels_completed": card.get("levels_completed", []),
                "actions": card.get("actions", []),
                "actions_by_level": card.get("actions_by_level", []),
                "resets": card.get("resets", []),
                "total_actions": card.get("total_actions", 0),
            }
        return {
            "card_id": dump.get("card_id", scorecard_id),
            "won": won,
            "played": played,
            "official_score": official_score,
            "official_reward": official_score / 100.0,
            "win_rate": won / played if played else 0.0,
            "levels_completed": total_levels_completed,
            "total_levels": total_levels,
            "total_environments_completed": int(
                official.get("total_environments_completed", won) or 0
            ),
            "total_environments": int(official.get("total_environments", played) or 0),
            "total_actions": int(official.get("total_actions", dump.get("total_actions", 0)) or 0),
            "competition_mode": dump.get("competition_mode"),
            "environment_scorecard": official,
            "cards": cards,
        }

    def official_scorecard_summary(self, scorecard: Any) -> dict[str, Any]:
        from arc_agi.scorecard import EnvironmentScorecard

        environment_infos = []
        seen: set[str] = set()
        for session in self.sessions.values():
            info = getattr(session.env, "info", None) or getattr(
                session.env, "environment_info", None
            )
            game_id = getattr(info, "game_id", None)
            if info is not None and game_id not in seen:
                environment_infos.append(info)
                seen.add(game_id)

        official = EnvironmentScorecard.from_scorecard(scorecard, environment_infos)
        dump = official.model_dump()
        dump.pop("api_key", None)
        dump.pop("opaque", None)
        return dump

    def frame_summary(self, observation: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not payload.get("include_frame", True):
            return []
        frames = getattr(observation, "frame", []) or []
        max_frames = int(payload.get("max_frames") or 1)
        max_rows = int(payload.get("max_rows") or 64)
        max_cols = int(payload.get("max_cols") or 64)
        summaries = []
        for index, frame in enumerate(frames[:max_frames]):
            if hasattr(frame, "shape"):
                shape = tuple(frame.shape)
            else:
                frame_rows = list(frame)
                shape = (len(frame_rows), len(frame_rows[0]) if frame_rows else 0)
            height = int(shape[0]) if shape else 0
            width = int(shape[1]) if len(shape) > 1 else 0
            try:
                unique_colors = [int(value) for value in sorted(set(frame.flatten().tolist()))]
                rows = frame[:max_rows, :max_cols].tolist()
            except AttributeError:
                rows = [list(row)[:max_cols] for row in list(frame)[:max_rows]]
                unique_colors = sorted({int(value) for row in rows for value in row})
            summaries.append(
                {
                    "index": index,
                    "height": height,
                    "width": width,
                    "unique_colors": unique_colors,
                    "rows": rows,
                    "truncated_rows": max(0, height - min(height, max_rows)),
                    "truncated_cols": max(0, width - min(width, max_cols)),
                }
            )
        if len(frames) > max_frames:
            summaries.append({"truncated_frames": len(frames) - max_frames})
        return summaries

    def shared_summary(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        expected_session_ids = self.expected_session_ids()
        if len(expected_session_ids) <= 1:
            return self.single_session_summary(snapshot)
        return self.multi_agent_summary(snapshot, expected_session_ids)

    def single_session_summary(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "updated_at": snapshot["updated_at"],
            "session_id": snapshot["session_id"],
            "game": snapshot["game"],
            "mode": snapshot["mode"],
            "reward": snapshot["official_reward"],
            "official_score": snapshot["official_score"],
            "official_reward": snapshot["official_reward"],
            "progress_reward": snapshot["progress_reward"],
            "win_levels": snapshot["win_levels"],
            "target_levels": snapshot["target_levels"],
            "target_reached": snapshot["target_reached"],
            "target_clear_actions": snapshot["target_clear_actions"],
            "target_reached_at": snapshot["target_reached_at"],
            "target_reached_steps": snapshot["target_reached_steps"],
            "level_clear_actions": snapshot["level_clear_actions"],
            "action_budget": snapshot["action_budget"],
            "scorecard": snapshot["scorecard"],
        }

    def target_clear_actions_for_snapshot(
        self, snapshot: dict[str, Any], target_levels: int
    ) -> int | None:
        direct = int_or_zero(snapshot.get("target_clear_actions"))
        if direct:
            return direct
        action_budget = snapshot.get("action_budget") or {}
        direct = int_or_zero(action_budget.get("target_clear_actions"))
        if direct:
            return direct
        level_clear_actions = snapshot.get("level_clear_actions") or {}
        if isinstance(level_clear_actions, dict):
            direct = int_or_zero(level_clear_actions.get(str(target_levels)))
            if direct:
                return direct
        if int_or_zero(snapshot.get("levels_completed")) >= target_levels:
            return int_or_zero(snapshot.get("steps")) or None
        if snapshot.get("target_reached") is True:
            return int_or_zero(snapshot.get("steps")) or None
        return None

    def session_summary_details(
        self, snapshot: dict[str, Any], target_levels: int
    ) -> dict[str, Any]:
        clear_actions = self.target_clear_actions_for_snapshot(snapshot, target_levels)
        return {
            "present": True,
            "session_id": snapshot["session_id"],
            "state": snapshot["state"],
            "steps": snapshot["steps"],
            "levels_completed": snapshot["levels_completed"],
            "scorecard_levels_completed": snapshot["scorecard_levels_completed"],
            "target_reached": clear_actions is not None,
            "target_clear_actions": clear_actions,
            "target_reached_at": snapshot["target_reached_at"],
            "target_reached_steps": snapshot["target_reached_steps"],
            "level_clear_actions": snapshot["level_clear_actions"],
            "reward": snapshot["official_reward"],
            "official_score": snapshot["official_score"],
            "official_reward": snapshot["official_reward"],
            "progress_reward": snapshot["progress_reward"],
            "action_budget": snapshot["action_budget"],
        }

    def multi_agent_summary(
        self, current_snapshot: dict[str, Any], expected_session_ids: list[str]
    ) -> dict[str, Any]:
        snapshots_by_session = {
            session_id: self.snapshot(session)
            for session_id, session in sorted(self.sessions.items())
        }
        target_levels = (
            int_or_zero(current_snapshot.get("target_levels")) or self.target_levels
        )
        session_details: dict[str, dict[str, Any]] = {}
        successful_sessions: dict[str, int] = {}
        failed_sessions: dict[str, dict[str, Any]] = {}
        team_level_clear_actions: dict[str, int] = {}
        observed_snapshots = list(snapshots_by_session.values())

        for session_id in expected_session_ids:
            snapshot = snapshots_by_session.get(session_id)
            if snapshot is None:
                session_details[session_id] = {
                    "present": False,
                    "session_id": session_id,
                    "target_reached": False,
                    "target_clear_actions": None,
                    "reason": "missing_session",
                }
                failed_sessions[session_id] = {"reason": "missing_session"}
                continue

            details = self.session_summary_details(snapshot, target_levels)
            session_details[session_id] = details
            if details["target_clear_actions"] is None:
                failed_sessions[session_id] = {
                    "reason": "target_not_reached",
                    "levels_completed": details["levels_completed"],
                    "steps": details["steps"],
                }
            else:
                successful_sessions[session_id] = int(details["target_clear_actions"])

            level_clear_actions = snapshot.get("level_clear_actions") or {}
            if isinstance(level_clear_actions, dict):
                for level, actions in level_clear_actions.items():
                    action_count = int_or_zero(actions)
                    if not action_count:
                        continue
                    level_key = str(level)
                    previous = team_level_clear_actions.get(level_key)
                    if previous is None or action_count < previous:
                        team_level_clear_actions[level_key] = action_count

        target_reached = bool(successful_sessions)
        target_clear_actions = (
            min(successful_sessions.values()) if successful_sessions else None
        )
        if target_reached and target_levels > 0 and target_clear_actions is not None:
            team_level_clear_actions.setdefault(str(target_levels), target_clear_actions)

        sorted_level_clear_actions = {
            level: team_level_clear_actions[level]
            for level in sorted(team_level_clear_actions, key=lambda value: int_or_zero(value))
        }
        best_snapshot = max(
            observed_snapshots or [current_snapshot],
            key=lambda item: (
                self.target_clear_actions_for_snapshot(item, target_levels) is not None,
                int_or_zero(item.get("levels_completed")),
                float(item.get("official_reward", 0.0) or 0.0),
                -int_or_zero(item.get("steps")),
            ),
        )
        target_reached_times = [
            str(details["target_reached_at"])
            for details in session_details.values()
            if details.get("target_reached_at")
        ]
        target_reached_steps = [
            int_or_zero(details["target_reached_steps"])
            for details in session_details.values()
            if int_or_zero(details.get("target_reached_steps"))
        ]
        team_levels_completed = max(
            [int_or_zero(snapshot.get("levels_completed")) for snapshot in observed_snapshots]
            + [max((int_or_zero(level) for level in sorted_level_clear_actions), default=0)]
        )
        team_scorecard_levels_completed = max(
            [int_or_zero(snapshot.get("scorecard_levels_completed")) for snapshot in observed_snapshots]
            + [team_levels_completed]
        )
        win_levels = max(
            [int_or_zero(snapshot.get("win_levels")) for snapshot in observed_snapshots]
            + [int_or_zero(current_snapshot.get("win_levels"))]
        )
        progress_denominator = target_levels or win_levels
        progress_reward = (
            min(team_levels_completed, progress_denominator) / progress_denominator
            if progress_denominator
            else 0.0
        )
        best_scorecard = best_snapshot.get("scorecard") or {}

        return {
            "updated_at": utc_now(),
            "summary_type": "multi_agent_team",
            "target_success_condition": "any_session_target_reached",
            "source_session_id": current_snapshot["session_id"],
            "session_id": "team",
            "session_ids": expected_session_ids,
            "session_count": len(observed_snapshots),
            "expected_session_count": len(expected_session_ids),
            "game": current_snapshot["game"],
            "mode": current_snapshot["mode"],
            "reward": float(best_snapshot.get("official_reward", 0.0) or 0.0),
            "official_score": float(best_snapshot.get("official_score", 0.0) or 0.0),
            "official_reward": float(best_snapshot.get("official_reward", 0.0) or 0.0),
            "progress_reward": progress_reward,
            "levels_completed": team_levels_completed,
            "scorecard_levels_completed": team_scorecard_levels_completed,
            "win_levels": win_levels,
            "target_levels": target_levels,
            "target_reached": target_reached,
            "target_any_reached": target_reached,
            "target_all_reached": (
                bool(expected_session_ids)
                and len(successful_sessions) == len(expected_session_ids)
                and not failed_sessions
            ),
            "target_success_count": len(successful_sessions),
            "target_expected_count": len(expected_session_ids),
            "target_success_session_ids": sorted(successful_sessions),
            "target_clear_actions": target_clear_actions,
            "target_clear_actions_min": target_clear_actions,
            "target_clear_actions_by_session": successful_sessions,
            "target_reached_at": min(target_reached_times) if target_reached_times else None,
            "target_reached_steps": min(target_reached_steps) if target_reached_steps else None,
            "level_clear_actions": sorted_level_clear_actions,
            "failed_sessions": failed_sessions,
            "sessions": session_details,
            "action_budget": {
                "target_reached": target_reached,
                "target_clear_actions": target_clear_actions,
                "target_reached_steps": min(target_reached_steps) if target_reached_steps else None,
                "target_levels": target_levels,
                "expected_session_count": len(expected_session_ids),
                "session_count": len(observed_snapshots),
                "success_condition": "any_session_target_reached",
                "per_session": {
                    session_id: details.get("action_budget")
                    for session_id, details in session_details.items()
                    if details.get("present")
                },
            },
            "scorecard": best_scorecard,
        }

    def persist(self, session: Session, event: str, result: dict[str, Any]) -> None:
        snapshot = self.snapshot(session)
        agent_file = STATE_DIR / f"{sanitize_session_id(session.session_id)}.jsonl"
        state_file = STATE_DIR / f"{sanitize_session_id(session.session_id)}.state.json"
        scorecard_file = STATE_DIR / "scorecard.json"
        record = {
            "time": utc_now(),
            "event": event,
            "snapshot": snapshot,
        }
        with agent_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")
        state_file.write_text(
            json.dumps(snapshot, default=json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scorecard_file.write_text(
            json.dumps(
                self.shared_summary(snapshot),
                default=json_default,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class Handler(BaseHTTPRequestHandler):
    controller: ArcController

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = self.controller.handle(payload)
            self.write_json(200, result)
        except Exception as exc:
            self.write_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def write_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, default=json_default, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server() -> None:
    Handler.controller = ArcController()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI-3 sidecar")
    parser.add_argument("command", nargs="?", choices=["server"], default="server")
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
