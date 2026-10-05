#!/usr/bin/env python3
"""Verifier for the ARC-AGI-3 interactive benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any


STATE_DIR = Path(os.environ.get("ARCCTL_STATE_DIR", "/logs/agent/arc_agi3_runs"))
VERIFIER_DIR = Path(os.environ.get("ARCCTL_VERIFIER_DIR", "/logs/verifier"))
REWARD_TXT = VERIFIER_DIR / "reward.txt"
REWARD_JSON = VERIFIER_DIR / "reward.json"
SIDE_CAR = VERIFIER_DIR / "arc_agi3_result.json"
TRACE_COPY_DIR = VERIFIER_DIR / "arc_agi3_runs"
SCORECARD_PATH = STATE_DIR / "scorecard.json"
SESSION_POLICY_PATH = STATE_DIR / "session_policy.json"


def load_snapshots() -> list[dict[str, Any]]:
    snapshots = []
    if not STATE_DIR.exists():
        return snapshots
    for path in sorted(STATE_DIR.glob("*.state.json")):
        try:
            snapshot = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: failed to read {path}: {exc}")
            continue
        snapshot["_path"] = str(path)
        snapshots.append(snapshot)
    return snapshots


def load_shared_scorecard() -> dict[str, Any] | None:
    if not SCORECARD_PATH.exists():
        return None
    try:
        snapshot = json.loads(SCORECARD_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: failed to read {SCORECARD_PATH}: {exc}")
        return None
    snapshot["_path"] = str(SCORECARD_PATH)
    return snapshot


def load_session_policy() -> dict[str, Any]:
    if not SESSION_POLICY_PATH.exists():
        return {}
    try:
        policy = json.loads(SESSION_POLICY_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: failed to read {SESSION_POLICY_PATH}: {exc}")
        return {}
    return policy if isinstance(policy, dict) else {}


def official_reward_for(snapshot: dict[str, Any]) -> float:
    """Return the official ARC score normalized to Harbor's 0-1 reward."""
    scorecard = snapshot.get("scorecard") or {}
    try:
        return float(
            snapshot.get(
                "official_reward",
                scorecard.get("official_reward", scorecard.get("official_score", 0.0) / 100.0),
            )
        )
    except (TypeError, ValueError):
        return 0.0


def progress_reward_for(snapshot: dict[str, Any]) -> float:
    """Side metric only: scorecard levels completed divided by this game's levels."""
    try:
        return float(snapshot.get("progress_reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def total_actions_for(snapshot: dict[str, Any]) -> int:
    scorecard = snapshot.get("scorecard") or {}
    try:
        return int(scorecard.get("total_actions", 0))
    except (TypeError, ValueError):
        return 0


def optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def reward_metrics_for(
    snapshot: dict[str, Any],
    official_reward: float,
    progress_reward: float,
    session_count: int,
) -> dict[str, Any]:
    scorecard = snapshot.get("scorecard") or {}
    return {
        "official_reward": official_reward,
        "official_score": float(scorecard.get("official_score", 0.0) or 0.0),
        "progress_reward": progress_reward,
        "won": int(scorecard.get("won", 0) or 0),
        "played": int(scorecard.get("played", 0) or 0),
        "win_rate": float(scorecard.get("win_rate", 0.0) or 0.0),
        "scorecard_levels_completed": int(scorecard.get("levels_completed", 0) or 0),
        "scorecard_total_levels": int(scorecard.get("total_levels", 0) or 0),
        "total_environments_completed": int(
            scorecard.get("total_environments_completed", 0) or 0
        ),
        "total_environments": int(scorecard.get("total_environments", 0) or 0),
        "total_actions": int(scorecard.get("total_actions", 0) or 0),
        "session_count": session_count,
    }


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def session_actions_used(snapshot: dict[str, Any]) -> int:
    action_budget = snapshot.get("action_budget") or {}
    for key in ("actions_used", "session_actions_used"):
        actions = int_or_zero(action_budget.get(key))
        if actions:
            return actions
    return int_or_zero(snapshot.get("steps"))


def target_levels_for(policy: dict[str, Any], snapshots: list[dict[str, Any]]) -> int:
    expected = optional_positive_int_env("ARCCTL_TARGET_LEVELS")
    if expected is not None:
        return expected
    policy_target = int_or_zero(policy.get("target_levels"))
    if policy_target > 0:
        return policy_target
    for snapshot in snapshots:
        target = int_or_zero(snapshot.get("target_levels"))
        if target > 0:
            return target
        target = int_or_zero((snapshot.get("action_budget") or {}).get("target_levels"))
        if target > 0:
            return target
    return 0


def target_clear_actions(snapshot: dict[str, Any], target_levels: int) -> int | None:
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
        return session_actions_used(snapshot)
    if snapshot.get("target_reached") is True:
        return session_actions_used(snapshot)
    return None


def target_config_violation(policy: dict[str, Any], target_levels: int) -> str | None:
    expected_budget = optional_positive_int_env("ARCCTL_FIXED_LEVEL_BUDGET")
    if expected_budget is not None:
        observed_budget = int_or_zero(policy.get("fixed_level_budget"))
        if observed_budget != expected_budget:
            return (
                "ARC fixed level budget mismatch: "
                f"expected {expected_budget}, sidecar has {observed_budget}"
            )
    expected_target = optional_positive_int_env("ARCCTL_TARGET_LEVELS")
    if expected_target is not None and target_levels != expected_target:
        return (
            "ARC target level mismatch: "
            f"expected {expected_target}, sidecar has {target_levels}"
        )
    return None


def latest_snapshot_by_session(
    snapshots: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_session: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        session_id = snapshot.get("session_id")
        if session_id is not None:
            by_session[str(session_id)] = snapshot
    return by_session


def write_target_reward(
    snapshots: list[dict[str, Any]],
    session_policy: dict[str, Any],
    target_levels: int,
) -> int:
    by_session = latest_snapshot_by_session(snapshots)
    expected_session_ids = [
        str(session_id)
        for session_id in (session_policy.get("expected_session_ids") or [])
        if str(session_id)
    ]
    if not expected_session_ids:
        expected_session_ids = sorted(by_session)

    policy_check_snapshot = next(iter(by_session.values()), {})
    violation = policy_violation(session_policy, snapshots, policy_check_snapshot)
    if violation is not None:
        print(f"Invalid ARC target run: {violation}")
        write_reward(
            0.0,
            violation,
            {
                "target_levels": target_levels,
                "target_team_success": 0,
                "target_any_cleared": 0,
                "target_all_cleared": 0,
                "target_success_count": 0,
                "target_expected_count": len(expected_session_ids),
                "target_success_condition": "any_session_target_reached",
            },
            {
                "session_count": len(snapshots),
                "session_policy": session_policy,
                "invalid_reason": violation,
            },
        )
        return 0

    config_violation = target_config_violation(session_policy, target_levels)
    if config_violation is not None:
        print(f"Invalid ARC target run: {config_violation}")
        write_reward(
            0.0,
            config_violation,
            {
                "target_levels": target_levels,
                "target_team_success": 0,
                "target_any_cleared": 0,
                "target_all_cleared": 0,
                "target_success_count": 0,
                "target_expected_count": len(expected_session_ids),
                "target_success_condition": "any_session_target_reached",
            },
            {
                "session_count": len(snapshots),
                "session_policy": session_policy,
                "invalid_reason": config_violation,
            },
        )
        return 0

    session_clear_actions: dict[str, int] = {}
    failed_sessions: dict[str, dict[str, Any]] = {}
    for session_id in expected_session_ids:
        snapshot = by_session.get(session_id)
        if snapshot is None:
            failed_sessions[session_id] = {"reason": "missing_session"}
            continue
        clear_actions = target_clear_actions(snapshot, target_levels)
        if clear_actions is None:
            failed_sessions[session_id] = {
                "reason": "target_not_reached",
                "levels_completed": int_or_zero(snapshot.get("levels_completed")),
                "steps": int_or_zero(snapshot.get("steps")),
            }
            continue
        session_clear_actions[session_id] = clear_actions

    all_cleared = (
        bool(expected_session_ids)
        and len(session_clear_actions) == len(expected_session_ids)
        and not failed_sessions
    )
    team_success = bool(session_clear_actions)
    min_clear_actions = min(session_clear_actions.values()) if session_clear_actions else 0
    max_clear_actions = max(session_clear_actions.values()) if session_clear_actions else 0
    mean_clear_actions = (
        sum(session_clear_actions.values()) / len(session_clear_actions)
        if session_clear_actions
        else 0.0
    )
    budget = (
        optional_positive_int_env("ARCCTL_FIXED_LEVEL_BUDGET")
        or int_or_zero(session_policy.get("fixed_level_budget"))
        or int_or_zero(
            ((next(iter(by_session.values()), {}).get("action_budget") or {})).get(
                "current_level_budget"
            )
        )
    )
    reward = 0.0
    if team_success:
        reward = 1.0
        if budget:
            reward = max(0.0, (budget - min_clear_actions) / budget)

    print(
        "ARC target-level result: "
        f"target_levels={target_levels} team_success={team_success} cleared="
        f"{len(session_clear_actions)}/{len(expected_session_ids)} "
        f"min_clear_actions={min_clear_actions or None} "
        f"max_clear_actions={max_clear_actions or None} "
        f"actions={session_clear_actions}"
    )
    if failed_sessions:
        print(f"ARC target-level failures: {failed_sessions}")

    metrics: dict[str, Any] = {
        "target_levels": target_levels,
        "target_team_success": int(team_success),
        "target_any_cleared": int(team_success),
        "target_all_cleared": int(all_cleared),
        "target_success_count": len(session_clear_actions),
        "target_expected_count": len(expected_session_ids),
        "target_clear_actions_min": min_clear_actions,
        "target_clear_actions_max": max_clear_actions,
        "target_clear_actions_mean": mean_clear_actions,
        "target_fixed_level_budget": budget,
        "target_success_condition": "any_session_target_reached",
        "session_clear_actions": session_clear_actions,
    }
    if failed_sessions:
        metrics["failed_sessions"] = failed_sessions
    write_reward(
        reward,
        "ARC target-level action efficiency",
        metrics,
        {
            "target_levels": target_levels,
            "team_success": team_success,
            "all_cleared": all_cleared,
            "session_clear_actions": session_clear_actions,
            "failed_sessions": failed_sessions,
            "session_count": len(snapshots),
            "session_policy": session_policy,
        },
    )
    return 0


def policy_violation(
    policy: dict[str, Any],
    snapshots: list[dict[str, Any]],
    best: dict[str, Any],
) -> str | None:
    if not policy:
        return None
    scorecard = best.get("scorecard") or {}
    expected_session_ids = {
        str(session_id)
        for session_id in (policy.get("expected_session_ids") or [])
        if str(session_id)
    }
    observed_session_ids = {
        str(snapshot.get("session_id"))
        for snapshot in snapshots
        if snapshot.get("session_id") is not None
    }
    unexpected_sessions = sorted(observed_session_ids - expected_session_ids)
    if unexpected_sessions:
        return "unexpected ARC session ids: " + ", ".join(unexpected_sessions)
    expected_session_count = int_or_zero(policy.get("expected_session_count")) or len(
        expected_session_ids
    )
    if expected_session_count and len(observed_session_ids) > expected_session_count:
        return (
            "too many ARC sessions: "
            f"{len(observed_session_ids)}/{expected_session_count}"
        )

    expected_game = str(policy.get("expected_game") or "")
    cards = scorecard.get("cards") or {}
    if expected_game and isinstance(cards, dict):
        unexpected_games = sorted(str(game) for game in cards if str(game) != expected_game)
        if unexpected_games:
            return "unexpected ARC games: " + ", ".join(unexpected_games)
    expected_mode = str(policy.get("expected_mode") or "normal")
    if expected_mode:
        observed_modes = {
            str(snapshot.get("mode"))
            for snapshot in snapshots
            if snapshot.get("mode") is not None
        }
        if best.get("mode") is not None:
            observed_modes.add(str(best.get("mode")))
        unexpected_modes = sorted(mode for mode in observed_modes if mode != expected_mode)
        if unexpected_modes:
            return "unexpected ARC modes: " + ", ".join(unexpected_modes)
    total_environments = int_or_zero(scorecard.get("total_environments"))
    if total_environments > 1:
        return f"too many ARC environments: {total_environments}/1"

    default_max_actions = int_or_zero((best.get("action_budget") or {}).get("max_actions"))
    for snapshot in snapshots:
        action_budget = snapshot.get("action_budget") or {}
        max_actions = int_or_zero(action_budget.get("max_actions")) or default_max_actions
        actions_used = session_actions_used(snapshot)
        if max_actions and actions_used > max_actions:
            session_id = snapshot.get("session_id", "<unknown>")
            return (
                "ARC session action budget exceeded: "
                f"session {session_id} used {actions_used}/{max_actions} actions"
            )
    return None


def copy_traces() -> None:
    if not STATE_DIR.exists():
        return
    TRACE_COPY_DIR.mkdir(parents=True, exist_ok=True)
    for path in STATE_DIR.glob("*"):
        if path.is_file():
            try:
                shutil.copy2(path, TRACE_COPY_DIR / path.name)
            except OSError as exc:
                print(f"WARN: failed to copy {path}: {exc}")


def write_reward(
    reward: float,
    detail: str,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    REWARD_TXT.write_text(str(reward))
    reward_metrics: dict[str, int | float] = {"reward": reward}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, bool):
                reward_metrics[key] = int(value)
            elif isinstance(value, (int, float)):
                reward_metrics[key] = value
    REWARD_JSON.write_text(json.dumps(reward_metrics, indent=2, sort_keys=True) + "\n")
    sidecar = {"reward": reward, "detail": detail}
    if metrics:
        sidecar["metrics"] = metrics
    if extra:
        sidecar.update(extra)
    SIDE_CAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def main() -> int:
    copy_traces()
    snapshots = load_snapshots()
    shared_scorecard = load_shared_scorecard()
    session_policy = load_session_policy()
    target_levels = target_levels_for(session_policy, snapshots)
    if target_levels > 0:
        return write_target_reward(snapshots, session_policy, target_levels)

    candidates = snapshots or ([shared_scorecard] if shared_scorecard is not None else [])
    if not candidates:
        print(f"No ARC-AGI-3 sessions found in {STATE_DIR}")
        write_reward(0.0, "no sessions found")
        return 0

    scored = [
        (
            official_reward_for(snapshot),
            progress_reward_for(snapshot),
            -total_actions_for(snapshot),
            snapshot,
        )
        for snapshot in candidates
    ]
    scored.sort(reverse=True, key=lambda item: item[:3])
    best_reward, best_progress, _, best = scored[0]
    scorecard = best.get("scorecard") or {}
    violation = policy_violation(session_policy, snapshots, best)
    if violation is not None:
        print(f"Invalid ARC-AGI-3 run: {violation}")
        write_reward(
            0.0,
            violation,
            reward_metrics_for(best, 0.0, best_progress, len(snapshots)),
            {
                "best_session": best,
                "session_count": len(snapshots),
                "session_policy": session_policy,
                "invalid_reason": violation,
            },
        )
        return 0
    print(
        "Best official scorecard session: "
        f"session={best.get('session_id')} "
        f"game={best.get('game')} "
        f"score={scorecard.get('official_score')}/100 "
        f"environments={scorecard.get('total_environments_completed')}/"
        f"{scorecard.get('total_environments')} "
        f"levels={scorecard.get('levels_completed')}/{scorecard.get('total_levels')} "
        f"actions={scorecard.get('total_actions')} "
        f"reward={best_reward:.4f}"
    )
    write_reward(
        best_reward,
        "official ARC-AGI-3 score normalized to 0-1",
        reward_metrics_for(best, best_reward, best_progress, len(snapshots)),
        {
            "best_session": best,
            "session_count": len(snapshots),
            "session_policy": session_policy,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
