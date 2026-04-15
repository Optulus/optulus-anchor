"""
LangChain message-loop helper for ``optulus-anchor`` correction-aware tool execution.

Usage::

    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

    executor = AnchorToolExecutor(tools)
    ai_message = llm.invoke(messages)
    messages.append(ai_message)
    messages.extend(executor.execute(messages=messages, ai_message=ai_message))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from optulus_anchor.integrations._tool_runtime import execute_tool_calls


class AnchorToolExecutor:
    """
    Execute tool calls from an ``AIMessage`` and return correction-aware ToolMessages.

    This helper is designed for LangChain chat loops where callers own the
    message history and append model/tool messages each turn.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self._by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def execute(
        self, *, messages: Sequence[BaseMessage], ai_message: AIMessage
    ) -> list[ToolMessage]:
        """Run tool calls from ``ai_message`` using ``messages`` for retry context."""

        if not ai_message.tool_calls:
            return []
        return execute_tool_calls(
            tools_by_name=self._by_name,
            messages=messages,
            tool_calls=ai_message.tool_calls,
        )

    def execute_last(self, *, messages: Sequence[BaseMessage]) -> list[ToolMessage]:
        """Run tool calls from the last message if it is an ``AIMessage``."""

        if not messages:
            return []
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return []
        return self.execute(messages=messages, ai_message=last)

    __call__ = execute
