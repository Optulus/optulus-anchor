from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from optulus_anchor.tracelog import persist_trace_entry

logger = logging.getLogger("optulus_anchor.tool_validator")

TraceSink = Callable[[dict[str, Any]], None]
_trace_sink: TraceSink | None = None


def set_trace_sink(sink: TraceSink | None) -> None:
    """
    Register or clear a callback that receives each emitted validation trace event.

    Use this to route structured trace events into your own telemetry pipeline
    (for example: test assertions, analytics ingestion, or custom observability).

    Args:
        sink: Callable that accepts one trace entry dictionary, or ``None`` to
            disable callback delivery.

    Returns:
        ``None``. This function updates process-global tracing configuration.

    Example:
        ```python
        from optulus_anchor import set_trace_sink

        events: list[dict[str, object]] = []

        def capture(event: dict[str, object]) -> None:
            events.append(event)

        set_trace_sink(capture)
        # ... run validated tools ...
        set_trace_sink(None)  # cleanup
        ```
    """
    global _trace_sink
    _trace_sink = sink


def _get_correction_context() -> tuple[str | None, int | None]:
    """Read correction cycle context vars set by ``AnchorToolNode`` (if active)."""
    try:
        from optulus_anchor.integrations._correction_ctx import (
            correction_attempt_var,
            correction_cycle_id_var,
        )

        return correction_cycle_id_var.get(), correction_attempt_var.get()
    except Exception:  # noqa: BLE001
        return None, None


def log_trace(
    tool_name: str,
    status: str,
    *,
    errors: list[str] | None = None,
    latency_ms: float | None = None,
    params_valid: bool | None = None,
    response_valid: bool | None = None,
    correction_cycle_id: str | None = None,
    correction_attempt: int | None = None,
) -> None:
    """
    Emit a structured trace event for tool validation and execution lifecycle.

    The event is logged via the package logger as JSON, persisted to a local
    SQLite file under ``.trace/`` when persistent tracing is enabled (see
    ``optulus_anchor.tracelog``), and optionally forwarded to the sink
    configured with ``set_trace_sink``.

    Args:
        tool_name: Logical tool/function name associated with the event.
        status: Lifecycle status token such as ``"PASS"``, ``"PARAM_FAIL"``,
            ``"RESPONSE_FAIL"``, or ``"EXECUTION_FAIL"``.
        errors: Optional list of human-readable validation or runtime errors.
        latency_ms: Optional execution latency in milliseconds.
        params_valid: Optional input validation result.
        response_valid: Optional output validation result.
        correction_cycle_id: Optional id grouping events in a single
            correction cycle (set automatically by ``AnchorToolNode``).
        correction_attempt: Optional 1-based attempt index within a
            correction cycle.

    Returns:
        ``None``. Side effects are log emission, optional SQLite persistence, and
        optional sink callback.

    Example:
        ```python
        from optulus_anchor.logger import log_trace

        log_trace(
            tool_name="search_documents",
            status="PASS",
            latency_ms=42,
            params_valid=True,
            response_valid=True,
        )
        ```
    """
    if correction_cycle_id is None or correction_attempt is None:
        ctx_cid, ctx_att = _get_correction_context()
        if correction_cycle_id is None:
            correction_cycle_id = ctx_cid
        if correction_attempt is None:
            correction_attempt = ctx_att

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool_name,
        "status": status,
        "latency_ms": latency_ms,
        "params_valid": params_valid,
        "response_valid": response_valid,
        "errors": errors or [],
        "correction_cycle_id": correction_cycle_id,
        "correction_attempt": correction_attempt,
    }

    message = json.dumps(entry)
    if status == "PASS":
        logger.info(message)
    else:
        logger.warning(message)

    persist_trace_entry(entry)

    if _trace_sink is not None:
        _trace_sink(entry)
