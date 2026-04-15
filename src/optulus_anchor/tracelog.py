from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

_logger = logging.getLogger("optulus_anchor.tracelog")

_TRACE_SUBDIR = ".trace"
_DB_FILENAME = "traces.sqlite"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None
_explicitly_disabled: bool = False


def _env_disables_persistent_trace() -> bool:
    raw = os.environ.get("OPTULUS_ANCHOR_NO_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _trace_db_path() -> Path:
    if raw := os.environ.get("OPTULUS_ANCHOR_TRACE_DIR", "").strip():
        base = Path(raw)
    else:
        base = Path.cwd()
    return base / _TRACE_SUBDIR / _DB_FILENAME


def disable_persistent_tracelog() -> None:
    """Disable writing trace events to SQLite for this process (until re-enabled)."""
    global _explicitly_disabled
    _explicitly_disabled = True


def enable_persistent_tracelog() -> None:
    """Re-enable SQLite trace persistence after ``disable_persistent_tracelog``."""
    global _explicitly_disabled
    _explicitly_disabled = False


def _persistent_trace_active() -> bool:
    return not _explicitly_disabled and not _env_disables_persistent_trace()


_CURRENT_SCHEMA_VERSION = "2"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_ms REAL,
            params_valid INTEGER,
            response_valid INTEGER,
            errors_json TEXT NOT NULL,
            correction_cycle_id TEXT,
            correction_attempt INTEGER
        );
        """
    )
    conn.commit()

    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    current = row[0] if row else None

    if current is None:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
            (_CURRENT_SCHEMA_VERSION,),
        )
    elif current < _CURRENT_SCHEMA_VERSION:
        _migrate(conn, current)
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'version'",
            (_CURRENT_SCHEMA_VERSION,),
        )


def _migrate(conn: sqlite3.Connection, from_version: str) -> None:
    if from_version == "1":
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(trace_events)").fetchall()
        }
        if "correction_cycle_id" not in columns:
            conn.execute(
                "ALTER TABLE trace_events ADD COLUMN correction_cycle_id TEXT DEFAULT NULL"
            )
        if "correction_attempt" not in columns:
            conn.execute(
                "ALTER TABLE trace_events ADD COLUMN correction_attempt INTEGER DEFAULT NULL"
            )


def _bool_to_sql(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _get_connection(path: Path) -> sqlite3.Connection:
    global _conn, _conn_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        timeout=30.0,
        isolation_level=None,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _ensure_schema(conn)
    _conn = conn
    _conn_path = path
    return conn


def persist_trace_entry(entry: dict[str, Any]) -> None:
    """
    Append one trace row to the local SQLite tracelog if persistence is enabled.

    Swallows SQLite errors after logging a warning so tool execution is not affected.
    """
    if not _persistent_trace_active():
        return

    path = _trace_db_path()
    try:
        with _lock:
            global _conn, _conn_path
            if _conn is None or _conn_path != path:
                if _conn is not None:
                    try:
                        _conn.close()
                    except OSError:
                        pass
                    _conn = None
                    _conn_path = None
                conn = _get_connection(path)
            else:
                conn = _conn

            conn.execute(
                """
                INSERT INTO trace_events (
                    timestamp, tool_name, status, latency_ms,
                    params_valid, response_valid, errors_json,
                    correction_cycle_id, correction_attempt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["timestamp"],
                    entry["tool"],
                    entry["status"],
                    entry["latency_ms"],
                    _bool_to_sql(entry.get("params_valid")),
                    _bool_to_sql(entry.get("response_valid")),
                    json.dumps(entry.get("errors") or []),
                    entry.get("correction_cycle_id"),
                    entry.get("correction_attempt"),
                ),
            )
    except Exception:
        _logger.warning("Failed to persist trace event to SQLite", exc_info=True)
