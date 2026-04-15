from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from optulus_anchor import (
    ToolValidationError,
    disable_persistent_tracelog,
    enable_persistent_tracelog,
    validate_tool,
)


class P(BaseModel):
    x: str


class R(BaseModel):
    y: str


def _db_path(trace_root: Path) -> Path:
    return trace_root / ".trace" / "traces.sqlite"


@validate_tool(params_schema=P, response_schema=R)
def _tool_ok(x: str) -> dict[str, str]:
    return {"y": x}


@validate_tool(params_schema=P, response_schema=R)
def _tool_raises(x: str) -> dict[str, str]:
    raise RuntimeError("boom")


def test_tracelog_pass_row(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()
    _tool_ok(x="a")

    db = _db_path(tmp_path)
    assert db.is_file()
    con = sqlite3.connect(db)
    rows = con.execute("SELECT status, latency_ms, errors_json FROM trace_events").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "PASS"
    assert rows[0][1] is not None
    assert rows[0][1] >= 0
    assert json.loads(rows[0][2]) == []


def test_tracelog_param_fail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()
    with pytest.raises(ToolValidationError):
        _tool_ok(wrong=1)  # type: ignore[call-arg]

    con = sqlite3.connect(_db_path(tmp_path))
    rows = con.execute("SELECT status, errors_json FROM trace_events").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "PARAM_FAIL"
    assert len(json.loads(rows[0][1])) >= 1


def test_tracelog_response_fail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()

    @validate_tool(params_schema=P, response_schema=R)
    def bad(x: str) -> dict[str, str]:
        return {"not_y": x}

    bad(x="u")
    con = sqlite3.connect(_db_path(tmp_path))
    rows = con.execute("SELECT status, latency_ms, errors_json FROM trace_events").fetchall()
    con.close()
    assert len(rows) == 2
    assert rows[0][0] == "RESPONSE_FAIL"
    assert rows[1][0] == "PASS"
    assert rows[0][1] is not None


def test_tracelog_execution_fail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()
    with pytest.raises(RuntimeError, match="boom"):
        _tool_raises(x="a")

    con = sqlite3.connect(_db_path(tmp_path))
    rows = con.execute("SELECT status, errors_json FROM trace_events").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "EXECUTION_FAIL"
    assert "boom" in json.dumps(json.loads(rows[0][1]))


def test_tracelog_disabled_via_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("OPTULUS_ANCHOR_NO_TRACE", "1")
    _tool_ok(x="b")
    assert not _db_path(tmp_path).exists()


def test_tracelog_disabled_programmatically(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    disable_persistent_tracelog()
    try:
        _tool_ok(x="c")
    finally:
        enable_persistent_tracelog()
    assert not _db_path(tmp_path).exists()


def test_tracelog_correction_columns(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """correction_cycle_id and correction_attempt are persisted when set."""
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()

    from optulus_anchor.logger import log_trace

    log_trace(
        "_tool_x",
        "PARAM_FAIL",
        errors=["bad x"],
        params_valid=False,
        correction_cycle_id="cycle-abc",
        correction_attempt=1,
    )
    log_trace(
        "_tool_x",
        "PASS",
        latency_ms=5.0,
        params_valid=True,
        correction_cycle_id="cycle-abc",
        correction_attempt=2,
    )

    con = sqlite3.connect(_db_path(tmp_path))
    rows = con.execute(
        "SELECT status, correction_cycle_id, correction_attempt FROM trace_events ORDER BY id"
    ).fetchall()
    con.close()

    assert len(rows) == 2
    assert rows[0] == ("PARAM_FAIL", "cycle-abc", 1)
    assert rows[1] == ("PASS", "cycle-abc", 2)


def test_tracelog_correction_columns_null_by_default(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """correction columns default to NULL when not supplied."""
    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()
    _tool_ok(x="z")

    con = sqlite3.connect(_db_path(tmp_path))
    row = con.execute(
        "SELECT correction_cycle_id, correction_attempt FROM trace_events"
    ).fetchone()
    con.close()
    assert row == (None, None)


def test_tracelog_schema_migration_v1_to_v2(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing v1 databases get the new columns via migration."""
    db = _db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO schema_meta (key, value) VALUES ('version', '1');

        CREATE TABLE trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_ms REAL,
            params_valid INTEGER,
            response_valid INTEGER,
            errors_json TEXT NOT NULL
        );
        INSERT INTO trace_events (timestamp, tool_name, status, latency_ms,
            params_valid, response_valid, errors_json)
        VALUES ('2026-01-01T00:00:00', 'old_tool', 'PASS', 1.0, 1, 1, '[]');
        """
    )
    con.close()

    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()

    from optulus_anchor.logger import log_trace

    log_trace("new_tool", "PASS", latency_ms=2.0, correction_cycle_id="cyc1", correction_attempt=1)

    con = sqlite3.connect(db)
    version = con.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()[0]
    assert version == "2"

    rows = con.execute(
        "SELECT tool_name, correction_cycle_id, correction_attempt FROM trace_events ORDER BY id"
    ).fetchall()
    con.close()

    assert rows[0] == ("old_tool", None, None)
    assert rows[1] == ("new_tool", "cyc1", 1)
