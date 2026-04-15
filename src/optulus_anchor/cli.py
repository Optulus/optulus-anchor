from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

_TRACE_SUBDIR = ".trace"
_DB_FILENAME = "traces.sqlite"


@dataclass(frozen=True)
class ToolStats:
    tool_name: str
    calls: int
    failures: int

    @property
    def failure_rate(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.failures / self.calls


@dataclass(frozen=True)
class DriftHint:
    field_name: str
    since_timestamp: datetime


def _trace_db_path() -> Path:
    raw = os.environ.get("OPTULUS_ANCHOR_TRACE_DIR", "").strip()
    base = Path(raw) if raw else Path.cwd()
    return base / _TRACE_SUBDIR / _DB_FILENAME


def _report_window_start(now: datetime, *, hours: int) -> datetime:
    return now - timedelta(hours=hours)


def _extract_missing_field(error_text: str) -> str | None:
    normalized = error_text.replace("\n", " ").strip()
    missing_field = re.search(
        r"missing field[:\s]+[`'\"]?([A-Za-z_][A-Za-z0-9_\.]*)[`'\"]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if missing_field:
        return missing_field.group(1).split(".")[-1]

    required_quoted = re.search(
        r"[`'\"]([A-Za-z_][A-Za-z0-9_\.]*)[`'\"]\s+field required",
        normalized,
        flags=re.IGNORECASE,
    )
    if required_quoted:
        return required_quoted.group(1).split(".")[-1]

    required_plain = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_\.]*)\b\s+field required",
        normalized,
        flags=re.IGNORECASE,
    )
    if required_plain:
        return required_plain.group(1).split(".")[-1]
    return None


def _parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _fetch_tool_stats(conn: sqlite3.Connection, start_time: datetime) -> list[ToolStats]:
    rows = conn.execute(
        """
        SELECT
            tool_name,
            COUNT(*) AS calls,
            SUM(CASE WHEN status != 'PASS' THEN 1 ELSE 0 END) AS failures
        FROM trace_events
        WHERE timestamp >= ?
        GROUP BY tool_name
        ORDER BY tool_name ASC
        """,
        (start_time.isoformat(),),
    ).fetchall()
    return [ToolStats(tool_name=row[0], calls=row[1], failures=row[2] or 0) for row in rows]


def _fetch_drift_hints(
    conn: sqlite3.Connection, start_time: datetime
) -> dict[str, DriftHint]:
    rows = conn.execute(
        """
        SELECT tool_name, timestamp, errors_json
        FROM trace_events
        WHERE timestamp >= ? AND status = 'RESPONSE_FAIL'
        ORDER BY timestamp ASC
        """,
        (start_time.isoformat(),),
    ).fetchall()

    hints: dict[str, DriftHint] = {}
    for tool_name, timestamp, errors_json in rows:
        try:
            errors = json.loads(errors_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(errors, list):
            continue

        field_name = _first_missing_field(errors)
        if field_name is None:
            continue

        parsed_ts = _parse_timestamp(timestamp)
        existing = hints.get(tool_name)
        if existing is None or parsed_ts < existing.since_timestamp:
            hints[tool_name] = DriftHint(field_name=field_name, since_timestamp=parsed_ts)
    return hints


def _first_missing_field(errors: Iterable[object]) -> str | None:
    for err in errors:
        if not isinstance(err, str):
            continue
        if field_name := _extract_missing_field(err):
            return field_name
    return None


@dataclass(frozen=True)
class CycleStats:
    tool_name: str
    total_cycles: int
    resolved: int
    exhausted: int
    avg_attempts: float


def _fetch_correction_cycles(
    conn: sqlite3.Connection, start_time: datetime
) -> list[CycleStats]:
    rows = conn.execute(
        """
        SELECT tool_name, correction_cycle_id,
               COUNT(*) AS attempts,
               MAX(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) AS resolved
        FROM trace_events
        WHERE correction_cycle_id IS NOT NULL
          AND timestamp >= ?
        GROUP BY tool_name, correction_cycle_id
        HAVING COUNT(*) > 1
            OR SUM(CASE WHEN status != 'PASS' THEN 1 ELSE 0 END) > 0
        """,
        (start_time.isoformat(),),
    ).fetchall()

    per_tool: dict[str, dict[str, int | float]] = {}
    for tool_name, _cid, attempts, resolved in rows:
        bucket = per_tool.setdefault(
            tool_name, {"total": 0, "resolved": 0, "exhausted": 0, "sum_attempts": 0}
        )
        bucket["total"] += 1
        bucket["sum_attempts"] += attempts
        if resolved:
            bucket["resolved"] += 1
        else:
            bucket["exhausted"] += 1

    return [
        CycleStats(
            tool_name=name,
            total_cycles=int(b["total"]),
            resolved=int(b["resolved"]),
            exhausted=int(b["exhausted"]),
            avg_attempts=round(b["sum_attempts"] / b["total"], 1) if b["total"] else 0,
        )
        for name, b in sorted(per_tool.items())
    ]


def _format_since(ts: datetime, now: datetime) -> str:
    local_ts = ts.astimezone()
    local_now = now.astimezone()
    if local_ts.date() == local_now.date():
        return f"since {local_ts:%H:%M} today"
    if local_ts.date() == (local_now.date() - timedelta(days=1)):
        return f"since {local_ts:%H:%M} yesterday"
    return f"since {local_ts:%Y-%m-%d %H:%M}"


def render_report(*, hours: int = 24, now: datetime | None = None) -> str:
    current_time = now or datetime.now().astimezone()
    start_time = _report_window_start(current_time, hours=hours)
    db_path = _trace_db_path()

    title = f"Tool Health Report — Last {hours}h"
    rule = "─" * 41

    if not db_path.exists():
        return "\n".join(
            [
                title,
                rule,
                f"No trace database found at: {db_path}",
                "Run instrumented tools first, then re-run `anchor report`.",
            ]
        )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        stats = _fetch_tool_stats(conn, start_time)
        drift_hints = _fetch_drift_hints(conn, start_time)
        cycles = _fetch_correction_cycles(conn, start_time)

    if not stats:
        return "\n".join(
            [
                title,
                rule,
                "No trace events found in the selected time window.",
                "Run tool calls and try again.",
            ]
        )

    name_width = max(len(item.tool_name) for item in stats)
    lines = [title, rule]
    for item in stats:
        icon = "✓" if item.failures == 0 else "⚠"
        line = (
            f"{item.tool_name:<{name_width}}  {icon} {item.calls} calls"
            f"   {item.failures} failures"
        )
        if item.tool_name in drift_hints:
            line = f"{line}  <- SCHEMA DRIFT DETECTED"
        lines.append(line)
        hint = drift_hints.get(item.tool_name)
        if hint is not None:
            lines.append(
                f"  └─ Missing field: {hint.field_name} ({_format_since(hint.since_timestamp, current_time)})"
            )

    if cycles:
        lines.append("")
        lines.append("Correction Cycles")
        lines.append("─" * 41)
        for cs in cycles:
            lines.append(
                f"{cs.tool_name}  {cs.total_cycles} cycle(s)"
                f"   avg {cs.avg_attempts} attempts"
                f"   {cs.resolved} resolved  {cs.exhausted} exhausted"
            )

    lines.append(rule)
    most_unreliable = max(
        stats,
        key=lambda item: (item.failure_rate, item.failures, item.calls, item.tool_name),
    )
    lines.append(
        "Most unreliable: "
        f"{most_unreliable.tool_name} ({most_unreliable.failure_rate:.0%} failure rate)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anchor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Summarize recent tool health")
    report_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24)",
    )

    args = parser.parse_args(argv)
    if args.command == "report":
        if args.hours <= 0:
            parser.error("--hours must be a positive integer")
        print(render_report(hours=args.hours))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
