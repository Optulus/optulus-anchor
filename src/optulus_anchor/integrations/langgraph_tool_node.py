"""
Drop-in replacement for :class:`langgraph.prebuilt.ToolNode` that understands
:func:`optulus_anchor.validate_tool`.

Usage::

    from optulus_anchor.integrations.langgraph_tool_node import AnchorToolNode

    graph.add_node("tools", AnchorToolNode(tools))

That's it — no custom state, no extra graph keys. Your state only needs
``messages`` (any ``TypedDict`` or ``MessagesState`` that LangGraph's
``tools_condition`` already works with).

How it differs from stock ``ToolNode``:

- Catches :class:`~optulus_anchor.ToolCorrectionNeeded` and converts it to a
  :class:`~langchain_core.messages.ToolMessage` containing the SDK's
  ``correction_prompt`` so the model can retry with fixed arguments.
- Enforces ``max_correction_attempts`` by **counting tagged correction
  messages in the conversation history** — no hidden kwargs, no extra state
  keys, no Store. Messages are already checkpointed, replay-safe, and
  scoped per thread.
- Catches :class:`~optulus_anchor.ToolValidationError` (strict ``"raise"``
  policies) and surfaces the error text as a ``ToolMessage``.
- Calls ``StructuredTool.func`` directly so ``validate_tool`` sees the raw
  arguments before LangChain's ``args_schema`` layer can reject them.

For ``on_param_error`` policies ``"raise"``, ``"log"``, or ``"warn"`` you can
also use stock ``ToolNode`` — ``AnchorToolNode`` just adds graceful handling.

Install: ``pip install optulus-anchor[langgraph]`` (or add ``langgraph`` and
``langchain-core`` to your environment).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from optulus_anchor import ToolCorrectionNeeded, ToolValidationError

logger = logging.getLogger(__name__)

_CORRECTION_META_KEY = "__anchor_correction"
_CORRECTION_TOOL_KEY = "__anchor_tool"


def _invoke_tool(tool: BaseTool, payload: dict[str, Any]) -> Any:
    """Call ``StructuredTool.func`` when available so ``validate_tool`` sees raw args."""

    if isinstance(tool, StructuredTool) and tool.func is not None:
        return tool.func(**payload)
    return tool.invoke(payload)


def _result_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


def _count_prior_corrections(messages: Sequence[BaseMessage], tool_name: str) -> int:
    """
    Count tagged correction ``ToolMessage``s for *tool_name* in *messages*.

    Only the current "correction run" matters: we stop counting once we hit a
    successful (non-correction) ``ToolMessage`` for the same tool, because that
    means a previous correction cycle completed.
    """

    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        extra = msg.additional_kwargs or {}
        if extra.get(_CORRECTION_TOOL_KEY) == tool_name:
            if extra.get(_CORRECTION_META_KEY):
                count += 1
            else:
                break
    return count


def _make_correction_message(
    exc: ToolCorrectionNeeded, tool_call_id: str, *, tool_spec_name: str
) -> ToolMessage:
    # Tag with ``tool_spec_name`` (``BaseTool.name`` / model tool_calls name), not
    # ``exc.tool_name`` (wrapped Python function name) — counting keys must match.
    return ToolMessage(
        content=exc.correction_prompt,
        tool_call_id=tool_call_id,
        additional_kwargs={
            _CORRECTION_META_KEY: True,
            _CORRECTION_TOOL_KEY: tool_spec_name,
        },
    )


def _make_exhausted_message(
    exc: ToolCorrectionNeeded, tool_call_id: str, prior: int, *, tool_spec_name: str
) -> ToolMessage:
    return ToolMessage(
        content=(
            f"Parameter correction budget exhausted ({prior + 1} "
            f"attempt(s) of {exc.max_attempts}).\n"
            f"errors={exc.errors}\n"
            f"attempted_params={json.dumps(exc.attempted_params, default=str)}"
        ),
        tool_call_id=tool_call_id,
        additional_kwargs={
            _CORRECTION_TOOL_KEY: tool_spec_name,
        },
    )


def _make_validation_error_message(
    exc: ToolValidationError, tool_call_id: str
) -> ToolMessage:
    return ToolMessage(
        content=(
            "ToolValidationError — read the schema, fix types/keys, "
            f"then call again.\n{exc}"
        ),
        tool_call_id=tool_call_id,
    )


def _make_success_message(result: Any, tool_call_id: str, tool_name: str) -> ToolMessage:
    return ToolMessage(
        content=_result_content(result),
        tool_call_id=tool_call_id,
        additional_kwargs={
            _CORRECTION_TOOL_KEY: tool_name,
        },
    )


class AnchorToolNode:
    """
    LangGraph node that executes tools with ``optulus-anchor`` validation awareness.

    Use exactly like :class:`langgraph.prebuilt.ToolNode`::

        graph.add_node("tools", AnchorToolNode(tools))

    Your graph state only needs ``messages`` — no ``correction_carry`` or other
    extra keys.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self._by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = list(state.get("messages") or [])
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        tool_messages: list[ToolMessage] = []

        for call in last.tool_calls:
            name: str = call["name"]
            tid: str = call["id"]
            args: dict[str, Any] = dict(call.get("args") or {})

            tool = self._by_name.get(name)
            if tool is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool {name!r}.", tool_call_id=tid)
                )
                continue

            try:
                result = _invoke_tool(tool, args)
                tool_messages.append(_make_success_message(result, tid, name))

            except ToolCorrectionNeeded as exc:
                prior = _count_prior_corrections(messages, name)
                if prior + 1 >= exc.max_attempts:
                    tool_messages.append(
                        _make_exhausted_message(exc, tid, prior, tool_spec_name=name)
                    )
                    logger.warning("Correction budget exhausted for %s", name)
                else:
                    tool_messages.append(
                        _make_correction_message(exc, tid, tool_spec_name=name)
                    )

            except ToolValidationError as exc:
                tool_messages.append(_make_validation_error_message(exc, tid))

        return {"messages": tool_messages}
