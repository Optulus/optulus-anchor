from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from optulus_anchor.cli import main, render_report


def _db_path(trace_root: Path) -> Path:
    return trace_root / ".trace" / "traces.sqlite"


def _init_db(trace_root: Path) -> Path:
    db_path = _db_path(trace_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL,
                params_valid INTEGER,
                response_valid INTEGER,
                errors_json TEXT NOT NULL
            );
            """
        )
    return db_path


def _insert_event(
    db_path: Path,
    *,
    timestamp: datetime,
    tool_name: str,
    status: str,
    errors: list[str] | None = None,
) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO trace_events (
                timestamp, tool_name, status, latency_ms,
                params_valid, response_valid, errors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                tool_name,
                status,
                None,
                None,
                None,
                json.dumps(errors or []),
            ),
        )


def test_render_report_aggregates_and_flags_drift(tmp_path) -> None:
    now = datetime(2026, 4, 14, 16, 0, tzinfo=UTC)
    db_path = _init_db(tmp_path)

    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=3),
        tool_name="send_email",
        status="PASS",
    )
    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=2),
        tool_name="send_email",
        status="PASS",
    )
    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=2),
        tool_name="charge_stripe",
        status="PASS",
    )
    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=1, minutes=30),
        tool_name="charge_stripe",
        status="EXECUTION_FAIL",
        errors=["boom"],
    )
    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=1, minutes=15),
        tool_name="charge_stripe",
        status="RESPONSE_FAIL",
        errors=["payment_method_details Field required [type=missing]"],
    )

    output = render_report(now=now, hours=24)

    assert "Tool Health Report — Last 24h" in output
    assert "send_email" in output
    assert "charge_stripe" in output
    assert "SCHEMA DRIFT DETECTED" in output
    assert "Missing field: payment_method_details" in output
    assert "Most unreliable: charge_stripe (67% failure rate)" in output


def test_render_report_handles_missing_database(tmp_path) -> None:
    output = render_report(now=datetime(2026, 4, 14, 16, 0, tzinfo=UTC), hours=24)
    assert "No trace database found at:" in output
    assert str(_db_path(tmp_path)) in output


def test_render_report_handles_empty_window(tmp_path) -> None:
    now = datetime(2026, 4, 14, 16, 0, tzinfo=UTC)
    db_path = _init_db(tmp_path)
    _insert_event(
        db_path,
        timestamp=now - timedelta(hours=30),
        tool_name="search_docs",
        status="PASS",
    )

    output = render_report(now=now, hours=24)
    assert "No trace events found in the selected time window." in output


def test_main_report_command_prints_report(capsys, tmp_path) -> None:
    now = datetime.now(tz=UTC)
    db_path = _init_db(tmp_path)
    _insert_event(
        db_path,
        timestamp=now - timedelta(minutes=30),
        tool_name="search_docs",
        status="PASS",
    )

    exit_code = main(["report", "--hours", "24"])
    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "Tool Health Report — Last 24h" in stdout
    assert "search_docs" in stdout


def test_main_rejects_non_positive_hours() -> None:
    with pytest.raises(SystemExit):
        main(["report", "--hours", "0"])
