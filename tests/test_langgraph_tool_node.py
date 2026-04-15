from typing import Annotated, Any

import pytest


def test_anchor_tool_node_self_correct_then_success() -> None:
    """Bad args -> correction ToolMessage -> fixed args -> success."""

    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")

    from langchain_core.language_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import tools_condition
    from pydantic import BaseModel, ConfigDict
    from typing_extensions import TypedDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langgraph_tool_node import AnchorToolNode

    class StrictParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: int

    class LooseLcArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: Any

    @validate_tool(params_schema=StrictParams, on_param_error="self_correct", max_correction_attempts=2)
    def strict_add(x: int) -> dict[str, int]:
        return {"y": x + 1}

    tool = StructuredTool.from_function(
        name="strict_add",
        description="add one to x",
        func=strict_add,
        args_schema=LooseLcArgs,
    )

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"type": "tool_call", "id": "1", "name": "strict_add", "args": {"x": "nope"}}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"type": "tool_call", "id": "2", "name": "strict_add", "args": {"x": 1}}
            ],
        ),
        AIMessage(content="done"),
    ]
    llm = FakeMessagesListChatModel(responses=responses)

    class S(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def agent(state: S, config: object) -> dict[str, object]:
        return {"messages": [llm.invoke(list(state["messages"]), config)]}

    graph = StateGraph(S)
    graph.add_node("agent", agent)
    graph.add_node("tools", AnchorToolNode([tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile(checkpointer=MemorySaver())

    final = app.invoke(
        {"messages": [HumanMessage("run")]},
        {"configurable": {"thread_id": "t"}, "recursion_limit": 20},
    )
    tool_contents = [m.content for m in final["messages"] if m.type == "tool"]
    assert any('"y": 2' in c for c in tool_contents), f"Expected success payload, got {tool_contents}"


def test_anchor_tool_node_budget_exhausted() -> None:
    """Correction budget (max_correction_attempts=1) is enforced from message history."""

    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")

    from langchain_core.language_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import tools_condition
    from pydantic import BaseModel, ConfigDict
    from typing_extensions import TypedDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langgraph_tool_node import AnchorToolNode

    class P(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: int

    class Lc(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: Any

    @validate_tool(params_schema=P, on_param_error="self_correct", max_correction_attempts=1)
    def add_one(x: int) -> dict[str, int]:
        return {"y": x + 1}

    tool = StructuredTool.from_function(
        name="add_one", description="add one", func=add_one, args_schema=Lc,
    )

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"type": "tool_call", "id": "1", "name": "add_one", "args": {"x": "bad"}}
            ],
        ),
        AIMessage(content="gave up"),
    ]
    llm = FakeMessagesListChatModel(responses=responses)

    class S(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def agent(state: S, config: object) -> dict[str, object]:
        return {"messages": [llm.invoke(list(state["messages"]), config)]}

    graph = StateGraph(S)
    graph.add_node("agent", agent)
    graph.add_node("tools", AnchorToolNode([tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile(checkpointer=MemorySaver())

    final = app.invoke(
        {"messages": [HumanMessage("run")]},
        {"configurable": {"thread_id": "t"}, "recursion_limit": 20},
    )
    tool_contents = [m.content for m in final["messages"] if m.type == "tool"]
    assert any("budget exhausted" in c.lower() for c in tool_contents), (
        f"Expected budget exhausted message, got {tool_contents}"
    )


def test_anchor_tool_node_strict_raise() -> None:
    """on_param_error='raise' -> ToolValidationError -> error ToolMessage."""

    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")

    from langchain_core.language_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import tools_condition
    from pydantic import BaseModel, ConfigDict
    from typing_extensions import TypedDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langgraph_tool_node import AnchorToolNode

    class P(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        city: str

    class Lc(BaseModel):
        model_config = ConfigDict(extra="forbid")
        city: Any

    @validate_tool(params_schema=P, on_param_error="raise")
    def get_weather(city: Any) -> dict[str, str]:
        return {"city": city, "condition": "sunny"}

    tool = StructuredTool.from_function(
        name="get_weather", description="weather", func=get_weather, args_schema=Lc,
    )

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"type": "tool_call", "id": "1", "name": "get_weather", "args": {"city": 123}}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"type": "tool_call", "id": "2", "name": "get_weather", "args": {"city": "NYC"}}
            ],
        ),
        AIMessage(content="done"),
    ]
    llm = FakeMessagesListChatModel(responses=responses)

    class S(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def agent(state: S, config: object) -> dict[str, object]:
        return {"messages": [llm.invoke(list(state["messages"]), config)]}

    graph = StateGraph(S)
    graph.add_node("agent", agent)
    graph.add_node("tools", AnchorToolNode([tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile(checkpointer=MemorySaver())

    final = app.invoke(
        {"messages": [HumanMessage("run")]},
        {"configurable": {"thread_id": "t"}, "recursion_limit": 20},
    )
    tool_contents = [m.content for m in final["messages"] if m.type == "tool"]
    assert any("ToolValidationError" in c for c in tool_contents)
    assert any('"city": "NYC"' in c or '"condition"' in c for c in tool_contents)
