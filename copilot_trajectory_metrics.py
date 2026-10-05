#!/usr/bin/env python3
"""Capture local Copilot OpenTelemetry records in a JSONL trajectory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaptureResult:
    records: int
    errors: int


def _append_event(handle, event_type: str, data: dict[str, object]) -> None:
    handle.write(
        json.dumps(
            {"type": event_type, "data": data},
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    )


def append_copilot_otel_events(
    trajectory_path: str | Path,
    otel_path: str | Path,
) -> CaptureResult:
    """Append raw local OTel records to a Copilot trajectory as typed events."""
    trajectory = Path(trajectory_path)
    otel = Path(otel_path)
    source = otel.name
    records = 0
    errors = 0

    with trajectory.open("a", encoding="utf-8") as output:
        if not otel.exists():
            _append_event(
                output,
                "copilot.telemetry_error",
                {"reason": "otel_file_missing", "source": source},
            )
            return CaptureResult(records=0, errors=1)

        with otel.open(errors="replace") as telemetry:
            for line_number, line in enumerate(telemetry, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    _append_event(
                        output,
                        "copilot.telemetry_error",
                        {
                            "line": line_number,
                            "reason": "invalid_otel_json",
                            "source": source,
                        },
                    )
                    continue
                records += 1
                _append_event(
                    output,
                    "copilot.telemetry",
                    {"record": record, "source": source},
                )

        if records == 0:
            errors += 1
            _append_event(
                output,
                "copilot.telemetry_error",
                {"reason": "otel_file_empty", "source": source},
            )
        _append_event(
            output,
            "copilot.telemetry_capture",
            {
                "errorCount": errors,
                "recordCount": records,
                "source": source,
            },
        )

    return CaptureResult(records=records, errors=errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("otel", type=Path)
    args = parser.parse_args()
    result = append_copilot_otel_events(args.trajectory, args.otel)
    return 0 if result.records > 0 and result.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
