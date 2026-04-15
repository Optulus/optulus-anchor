import sqlite3
from pathlib import Path
from typing import Any

import pytest


def test_anchor_tool_executor_self_correct_then_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langchain_core")

    from langchain_core.language_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    from optulus_anchor import enable_persistent_tracelog, validate_tool
    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

    monkeypatch.setenv("OPTULUS_ANCHOR_TRACE_DIR", str(tmp_path))
    enable_persistent_tracelog()

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
    executor = AnchorToolExecutor([tool])

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
    messages = [HumanMessage("run")]

    for _ in range(3):
        ai = llm.invoke(messages)
        messages.append(ai)
        tool_messages = executor.execute(messages=messages, ai_message=ai)
        if not tool_messages:
            break
        messages.extend(tool_messages)

    tool_contents = [m.content for m in messages if m.type == "tool"]
    assert any('"y": 2' in c for c in tool_contents), f"Expected success payload, got {tool_contents}"

    db = tmp_path / ".trace" / "traces.sqlite"
    assert db.is_file()
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT status, correction_cycle_id, correction_attempt FROM trace_events ORDER BY id"
    ).fetchall()
    con.close()
    fail_row = next(r for r in rows if r[0] == "PARAM_FAIL")
    pass_row = next(r for r in rows if r[0] == "PASS")
    assert fail_row[1] is not None
    assert fail_row[1] == pass_row[1]
    assert fail_row[2] == 1
    assert pass_row[2] == 2


def test_anchor_tool_executor_budget_exhausted() -> None:
    pytest.importorskip("langchain_core")

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

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
        name="add_one",
        description="add one",
        func=add_one,
        args_schema=Lc,
    )
    executor = AnchorToolExecutor([tool])

    ai = AIMessage(
        content="",
        tool_calls=[{"type": "tool_call", "id": "1", "name": "add_one", "args": {"x": "bad"}}],
    )
    messages = [HumanMessage("run"), ai]
    tool_messages = executor.execute(messages=messages, ai_message=ai)
    assert any("budget exhausted" in m.content.lower() for m in tool_messages)


def test_anchor_tool_executor_strict_raise() -> None:
    pytest.importorskip("langchain_core")

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

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
        name="get_weather",
        description="weather",
        func=get_weather,
        args_schema=Lc,
    )
    executor = AnchorToolExecutor([tool])

    bad_ai = AIMessage(
        content="",
        tool_calls=[
            {"type": "tool_call", "id": "1", "name": "get_weather", "args": {"city": 123}}
        ],
    )
    messages = [HumanMessage("run"), bad_ai]
    first_tool_messages = executor.execute(messages=messages, ai_message=bad_ai)
    assert any("ToolValidationError" in m.content for m in first_tool_messages)

    good_ai = AIMessage(
        content="",
        tool_calls=[
            {"type": "tool_call", "id": "2", "name": "get_weather", "args": {"city": "NYC"}}
        ],
    )
    messages.extend(first_tool_messages)
    messages.append(good_ai)
    second_tool_messages = executor.execute(messages=messages, ai_message=good_ai)
    assert any('"city": "NYC"' in m.content or '"condition"' in m.content for m in second_tool_messages)


def test_anchor_tool_executor_bypasses_args_schema_for_validate_tool() -> None:
    pytest.importorskip("langchain_core")

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    from optulus_anchor import validate_tool
    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

    class StrictParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: int

    class StrictLcArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        x: int

    @validate_tool(params_schema=StrictParams, on_param_error="self_correct", max_correction_attempts=2)
    def strict_add(x: int) -> dict[str, int]:
        return {"y": x + 1}

    tool = StructuredTool.from_function(
        name="strict_add",
        description="add one to x",
        func=strict_add,
        args_schema=StrictLcArgs,
    )
    executor = AnchorToolExecutor([tool])

    ai = AIMessage(
        content="",
        tool_calls=[
            {"type": "tool_call", "id": "1", "name": "strict_add", "args": {"x": "bad"}}
        ],
    )
    messages = [HumanMessage("run"), ai]
    tool_messages = executor.execute(messages=messages, ai_message=ai)

    assert len(tool_messages) == 1
    assert "parameter validation failed" in tool_messages[0].content.lower()


def test_anchor_tool_executor_unknown_tool() -> None:
    pytest.importorskip("langchain_core")

    from langchain_core.messages import AIMessage, HumanMessage

    from optulus_anchor.integrations.langchain_tool_executor import AnchorToolExecutor

    executor = AnchorToolExecutor([])
    ai = AIMessage(
        content="",
        tool_calls=[{"type": "tool_call", "id": "missing", "name": "not_real", "args": {}}],
    )
    messages = [HumanMessage("run"), ai]
    tool_messages = executor.execute(messages=messages, ai_message=ai)
    assert len(tool_messages) == 1
    assert "Unknown tool" in tool_messages[0].content
