from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("optulus_anchor.tool_validator")

TraceSink = Callable[[dict[str, Any]], None]
_trace_sink: TraceSink | None = None


def set_trace_sink(sink: TraceSink | None) -> None:
    """Set an optional callback that receives every trace entry."""
    global _trace_sink
    _trace_sink = sink


def log_trace(
    tool_name: str,
    status: str,
    *,
    errors: list[str] | None = None,
    latency_ms: int | None = None,
    params_valid: bool | None = None,
    response_valid: bool | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool_name,
        "status": status,
        "latency_ms": latency_ms,
        "params_valid": params_valid,
        "response_valid": response_valid,
        "errors": errors or [],
    }

    message = json.dumps(entry)
    if status == "PASS":
        logger.info(message)
    else:
        logger.warning(message)

    if _trace_sink is not None:
        _trace_sink(entry)
