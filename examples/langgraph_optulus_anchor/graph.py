"""
LangGraph ReAct loop using :class:`~optulus_anchor.integrations.langgraph_tool_node.AnchorToolNode`.

Uses LangGraph’s built-in :class:`~langgraph.graph.message.MessagesState` (only ``messages``).
Swap ``ToolNode`` for ``AnchorToolNode`` — no SDK-specific state type.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import tools_condition

from optulus_anchor.integrations.langgraph_tool_node import AnchorToolNode

from .tools import ALL_TOOLS


def build_agent_graph(*, llm: BaseChatModel, tools: Sequence[BaseTool] | None = None) -> Any:
    """
    Compile a checkpointed graph: ``agent`` → (tools | END) → ``agent``.

    ``llm`` should already have ``bind_tools`` applied when using a real chat model.
    """

    tool_list = list(tools or ALL_TOOLS)

    def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        return {"messages": [llm.invoke(list(state["messages"]), config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", AnchorToolNode(tool_list))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


def demo_mock_llm_script() -> list[AIMessage]:
    """
    Scripted model turns: wrong reservation -> corrected reservation + bad weather ->
    good weather -> final natural-language answer.
    """

    def tc(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "id": str(uuid4()),
            "name": name,
            "args": args,
        }

    return [
        AIMessage(
            content="",
            tool_calls=[
                tc(
                    "reserve_table",
                    {
                        "party_size": "two",
                        "reservation_time": "tomorrow at 7",
                        "guest_name": "Jordan",
                    },
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tc(
                    "reserve_table",
                    {
                        "party_size": 2,
                        "reservation_time": "2026-04-15T19:00:00Z",
                        "guest_name": "Jordan",
                    },
                ),
                tc(
                    "get_weather",
                    {"city": 10001},
                ),
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tc("get_weather", {"city": "New York"}),
            ],
        ),
        AIMessage(
            content=(
                "Reservation confirmed with the returned confirmation code, "
                "and the weather stub shows the forecast for New York."
            ),
        ),
    ]


def demo_mock_llm_script_exhaust_correction_budget() -> list[AIMessage]:
    """
    Model keeps sending invalid ``reserve_table`` calls until ``AnchorToolNode``
    hits ``max_correction_attempts`` on the tool (3 for ``reserve_table_impl``).

    Use this to verify attempt counting: the third bad call should produce a
    ``ToolMessage`` whose body contains "correction budget exhausted".
    """

    def tc(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "id": str(uuid4()),
            "name": name,
            "args": args,
        }

    bad = {
        "party_size": "still-not-an-int",
        "reservation_time": "not-iso",
        "guest_name": "Jordan",
    }
    return [
        AIMessage(content="", tool_calls=[tc("reserve_table", bad)]),
        AIMessage(content="", tool_calls=[tc("reserve_table", bad)]),
        AIMessage(content="", tool_calls=[tc("reserve_table", bad)]),
        AIMessage(
            content="Stopping: the tool reported the correction budget was exhausted."
        ),
    ]


def run_mock_demo() -> dict[str, Any]:
    from langchain_core.language_models import FakeMessagesListChatModel

    tools = ALL_TOOLS
    llm = FakeMessagesListChatModel(responses=demo_mock_llm_script())
    graph = build_agent_graph(llm=llm, tools=tools)

    return graph.invoke(
        {"messages": [
            HumanMessage(
                content=(
                    "Book a table for two people at 2026-04-15T19:00:00Z UTC "
                    "under the name Jordan, then tell me the weather in New York."
                )
            )
        ]},
        {"configurable": {"thread_id": "mock-demo"}},
    )


def run_mock_demo_exhaust_correction_budget() -> dict[str, Any]:
    """Mock run where invalid ``reserve_table`` args repeat until the budget is exhausted."""

    from langchain_core.language_models import FakeMessagesListChatModel

    tools = ALL_TOOLS
    llm = FakeMessagesListChatModel(
        responses=demo_mock_llm_script_exhaust_correction_budget()
    )
    graph = build_agent_graph(llm=llm, tools=tools)

    return graph.invoke(
        {"messages": [HumanMessage("Book a table (mock will keep bad args).")]},
        {"configurable": {"thread_id": "mock-demo-exhaust"}, "recursion_limit": 25},
    )
