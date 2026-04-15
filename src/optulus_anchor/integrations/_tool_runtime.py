from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from optulus_anchor import ToolCorrectionNeeded, ToolValidationError
from optulus_anchor.integrations._correction_ctx import correction_context

logger = logging.getLogger(__name__)

_CORRECTION_META_KEY = "__anchor_correction"
_CORRECTION_TOOL_KEY = "__anchor_tool"
_CORRECTION_CYCLE_KEY = "__anchor_cycle_id"


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


def _count_prior_corrections(
    messages: Sequence[BaseMessage], tool_name: str
) -> tuple[int, str | None]:
    """Return ``(count, cycle_id)`` for the current correction run of *tool_name*."""

    count = 0
    cycle_id: str | None = None
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        extra = msg.additional_kwargs or {}
        if extra.get(_CORRECTION_TOOL_KEY) == tool_name:
            if extra.get(_CORRECTION_META_KEY):
                count += 1
                if cycle_id is None:
                    cycle_id = extra.get(_CORRECTION_CYCLE_KEY)
            else:
                break
    return count, cycle_id


def _make_correction_message(
    exc: ToolCorrectionNeeded,
    tool_call_id: str,
    *,
    tool_spec_name: str,
    cycle_id: str,
) -> ToolMessage:
    return ToolMessage(
        content=exc.correction_prompt,
        tool_call_id=tool_call_id,
        additional_kwargs={
            _CORRECTION_META_KEY: True,
            _CORRECTION_TOOL_KEY: tool_spec_name,
            _CORRECTION_CYCLE_KEY: cycle_id,
        },
    )


def _make_exhausted_message(
    exc: ToolCorrectionNeeded,
    tool_call_id: str,
    prior: int,
    *,
    tool_spec_name: str,
    cycle_id: str,
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
            _CORRECTION_CYCLE_KEY: cycle_id,
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


def execute_tool_calls(
    *,
    tools_by_name: Mapping[str, BaseTool],
    messages: Sequence[BaseMessage],
    tool_calls: Sequence[dict[str, Any]],
) -> list[ToolMessage]:
    """
    Execute tool calls with correction-aware behavior and return ToolMessages.

    This is framework-agnostic runtime logic shared by LangGraph and LangChain
    integrations.
    """

    tool_messages: list[ToolMessage] = []

    for call in tool_calls:
        name = str(call["name"])
        tool_call_id = str(call.get("id") or call.get("tool_call_id") or "")
        args = dict(call.get("args") or {})

        tool = tools_by_name.get(name)
        if tool is None:
            tool_messages.append(
                ToolMessage(content=f"Unknown tool {name!r}.", tool_call_id=tool_call_id)
            )
            continue

        prior, existing_cycle_id = _count_prior_corrections(messages, name)
        attempt = prior + 1
        cycle_id = existing_cycle_id if prior > 0 else uuid.uuid4().hex

        try:
            with correction_context(cycle_id, attempt):
                result = _invoke_tool(tool, args)
            tool_messages.append(_make_success_message(result, tool_call_id, name))

        except ToolCorrectionNeeded as exc:
            if attempt >= exc.max_attempts:
                tool_messages.append(
                    _make_exhausted_message(
                        exc, tool_call_id, prior, tool_spec_name=name, cycle_id=cycle_id
                    )
                )
                logger.warning("Correction budget exhausted for %s", name)
            else:
                tool_messages.append(
                    _make_correction_message(
                        exc, tool_call_id, tool_spec_name=name, cycle_id=cycle_id
                    )
                )

        except ToolValidationError as exc:
            tool_messages.append(_make_validation_error_message(exc, tool_call_id))

    return tool_messages
