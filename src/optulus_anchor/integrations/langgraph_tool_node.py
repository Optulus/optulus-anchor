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

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from optulus_anchor.integrations._tool_runtime import execute_tool_calls


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
        return {
            "messages": execute_tool_calls(
                tools_by_name=self._by_name,
                messages=messages,
                tool_calls=last.tool_calls,
            )
        }
