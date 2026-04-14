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
