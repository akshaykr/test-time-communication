#!/usr/bin/env python3
"""Thin ARC-AGI-3 client for Harbor CLI agents."""

from __future__ import annotations

import argparse
from http.client import HTTPConnection
import json
import os
from typing import Any


HOST = os.environ.get("ARCCTL_HOST", "127.0.0.1")
PORT = int(os.environ.get("ARCCTL_PORT", "8765"))
DEFAULT_GAME = os.environ.get("ARCCTL_DEFAULT_GAME", "ls20")
SERVER_TIMEOUT = float(os.environ.get("ARCCTL_SERVER_TIMEOUT", "120"))
SESSION_TOKEN_ENV = "ARCCTL_SESSION_TOKEN"
SESSION_ID_ENV = "ARCCTL_SESSION_ID"


def json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def request(payload: dict[str, Any]) -> dict[str, Any]:
    conn = HTTPConnection(HOST, PORT, timeout=SERVER_TIMEOUT)
    try:
        conn.request(
            "POST",
            "/",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read().decode("utf-8")
    finally:
        conn.close()

    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {"ok": False, "error": body}
    if response.status >= 400:
        data.setdefault("ok", False)
    return data


def parse_json_argument(value: str | None, *, name: str) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be valid JSON: {exc}") from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def print_result(data: dict[str, Any]) -> int:
    print(json.dumps(data, default=json_default, indent=2))
    return 0 if data.get("ok") else 1


def add_frame_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-cols", type=int, default=64)
    parser.add_argument("--no-frame", action="store_true")


def frame_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "include_frame": not getattr(args, "no_frame", False),
        "max_frames": getattr(args, "max_frames", 1),
        "max_rows": getattr(args, "max_rows", 64),
        "max_cols": getattr(args, "max_cols", 64),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control ARC-AGI-3 environments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    game_help = "Configured benchmark game; other games are rejected"

    subparsers.add_parser("ping", help="Check ARC sidecar readiness")
    subparsers.add_parser("list-games", help="List available games")

    configure = subparsers.add_parser(
        "configure-sessions",
        help="Register benchmark session tokens with the ARC sidecar",
    )
    configure.add_argument(
        "--bindings",
        required=True,
        type=lambda value: parse_json_argument(value, name="--bindings"),
    )
    configure.add_argument("--game", default=DEFAULT_GAME, help=game_help)
    configure.add_argument("--fixed-level-budget", type=positive_int, default=None)
    configure.add_argument("--target-levels", type=nonnegative_int, default=None)
    configure.add_argument("--sync-freq", type=nonnegative_int, default=None)

    reset = subparsers.add_parser("reset", help="Reset the current ARC session")
    reset.add_argument("--game", default=None, help=game_help)
    add_frame_options(reset)

    observe = subparsers.add_parser("observe", help="Print the current observation")
    observe.add_argument("--game", default=DEFAULT_GAME, help=game_help)
    add_frame_options(observe)

    actions = subparsers.add_parser("actions", help="Print available actions")
    actions.add_argument("--game", default=DEFAULT_GAME, help=game_help)

    step = subparsers.add_parser("step", help="Take one action")
    step.add_argument("action", help="ACTION1, ACTION2, ..., 1, 2, ..., or RESET")
    step.add_argument("--data", type=lambda value: parse_json_argument(value, name="--data"))
    step.add_argument(
        "--reasoning",
        type=lambda value: parse_json_argument(value, name="--reasoning"),
    )
    step.add_argument("--game", default=DEFAULT_GAME, help=game_help)
    add_frame_options(step)

    score = subparsers.add_parser("score", help="Print current progress score")
    score.add_argument("--game", default=DEFAULT_GAME, help=game_help)

    barrier = subparsers.add_parser(
        "barrier",
        help="Print only the sync-barrier status (lightweight; no grid frame)",
    )
    barrier.add_argument("--game", default=DEFAULT_GAME, help=game_help)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload: dict[str, Any] = {"command": args.command}
    if args.command == "configure-sessions":
        payload["bindings"] = args.bindings
        payload["game"] = args.game
        if args.fixed_level_budget is not None:
            payload["fixed_level_budget"] = args.fixed_level_budget
        if args.target_levels is not None:
            payload["target_levels"] = args.target_levels
        if args.sync_freq is not None:
            payload["sync_freq"] = args.sync_freq
    else:
        session_token = os.environ.get(SESSION_TOKEN_ENV)
        if session_token:
            payload["session_token"] = session_token
        session_id = os.environ.get(SESSION_ID_ENV) or os.environ.get("AGENT_ID")
        if session_id:
            payload["agent_id"] = session_id
        if hasattr(args, "game"):
            payload["game"] = args.game
        if args.command in {"reset", "observe", "step"}:
            payload.update(frame_payload(args))
        if args.command == "step":
            payload.update(
                {
                    "action": args.action,
                    "data": args.data,
                    "reasoning": args.reasoning,
                }
            )
    try:
        data = request(payload)
    except OSError as exc:
        data = {
            "ok": False,
            "error": f"could not reach ARC sidecar at {HOST}:{PORT}: {exc}",
        }
    return print_result(data)


if __name__ == "__main__":
    raise SystemExit(main())
