"""Thread-safe correction-cycle context propagated via :mod:`contextvars`.

``AnchorToolNode`` sets these variables around each tool invocation so that
:func:`optulus_anchor.logger.log_trace` can automatically attach
``correction_cycle_id`` and ``correction_attempt`` to every trace event
emitted by the ``@validate_tool`` decorator — without changing the
decorator's own API.

This module is intentionally lightweight and has **no** LangGraph / LangChain
imports so ``logger.py`` can safely import it at any time.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

correction_cycle_id_var: ContextVar[str | None] = ContextVar(
    "correction_cycle_id", default=None
)
correction_attempt_var: ContextVar[int | None] = ContextVar(
    "correction_attempt", default=None
)


@contextmanager
def correction_context(
    cycle_id: str | None, attempt: int | None
) -> Iterator[None]:
    """Set correction context vars for the duration of a ``with`` block."""
    cid_token = correction_cycle_id_var.set(cycle_id)
    att_token = correction_attempt_var.set(attempt)
    try:
        yield
    finally:
        correction_cycle_id_var.reset(cid_token)
        correction_attempt_var.reset(att_token)
