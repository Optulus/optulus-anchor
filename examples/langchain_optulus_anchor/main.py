"""
Run the LangChain + AnchorToolExecutor example.

From repository root:

    pip install -e .
    pip install -r examples/langchain_optulus_anchor/requirements.txt
    python -m examples.langchain_optulus_anchor.main

Or from this directory:

    python main.py

Environment:

- ``ANCHOR_LANGCHAIN_DEMO_EXHAUST_BUDGET=1`` to run budget-exhaustion script.
- ``ANCHOR_LANGCHAIN_PRINT_TRACES=1`` to print trace entries as they are emitted.
- ``OPTULUS_ANCHOR_TRACE_DIR`` to set persistent trace directory.
- ``OPTULUS_ANCHOR_NO_TRACE=1`` to disable persistent trace writes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

# ``python main.py`` leaves ``__package__`` unset; align with ``python -m ...``.
_main = sys.modules["__main__"]
if not getattr(_main, "__package__", None):
    _here = Path(__file__).resolve().parent
    _repo = _here.parents[2]
    _src = _repo / "src"
    for p in (str(_repo), str(_src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _main.__package__ = "examples.langchain_optulus_anchor"

from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from optulus_anchor import set_trace_sink
from optulus_anchor.integrations import AnchorToolExecutor

from tools import ALL_TOOLS

logger = logging.getLogger("examples.langchain_optulus_anchor")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _maybe_register_trace_sink() -> None:
    if os.environ.get("ANCHOR_LANGCHAIN_PRINT_TRACES", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    def sink(entry: dict[str, Any]) -> None:
        print("[trace_sink]", json.dumps(entry, default=str))

    set_trace_sink(sink)
    logger.info("Registered trace sink printer.")


def _print_message(message: BaseMessage) -> None:
    role = type(message).__name__.replace("Message", "")
    content = (message.content or "").strip()
    if content:
        print(f"\n[{role}]\n{content}")
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"[{role} tool_calls]")
        for call in message.tool_calls:
            print(f"  - {call['name']}({json.dumps(call.get('args'), default=str)})")
    if isinstance(message, ToolMessage):
        extra = message.additional_kwargs or {}
        if extra:
            print(f"  additional_kwargs={json.dumps(extra, default=str)}")


def _tc(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": str(uuid4()), "name": name, "args": args}


def scripted_self_correct_and_strict_raise() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                _tc(
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
                _tc(
                    "reserve_table",
                    {
                        "party_size": 2,
                        "reservation_time": "2026-04-15T19:00:00Z",
                        "guest_name": "Jordan",
                    },
                ),
                _tc("get_weather", {"city": 10001}),
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tc("get_weather", {"city": "New York"}),
                _tc("search_docs", {"query": "optulus anchor", "limit": 2}),
            ],
        ),
        AIMessage(
            content=(
                "Reservation completed, weather fetched after correction, and docs searched."
            )
        ),
    ]


def scripted_exhaust_budget() -> list[AIMessage]:
    bad = {
        "party_size": "still-not-an-int",
        "reservation_time": "not-iso",
        "guest_name": "Jordan",
    }
    return [
        AIMessage(content="", tool_calls=[_tc("reserve_table", bad)]),
        AIMessage(content="", tool_calls=[_tc("reserve_table", bad)]),
        AIMessage(content="Stopping after budget exhaustion."),
    ]


def run_with_script(messages_script: list[AIMessage], *, max_turns: int = 10) -> list[BaseMessage]:
    llm = FakeMessagesListChatModel(responses=messages_script)
    executor = AnchorToolExecutor(ALL_TOOLS)
    messages: list[BaseMessage] = [
        HumanMessage(
            "Run the scripted tool calls and recover from validation failures where applicable."
        )
    ]

    for _ in range(max_turns):
        ai = llm.invoke(messages)
        messages.append(ai)
        tool_messages = executor.execute(messages=messages, ai_message=ai)
        if not tool_messages:
            break
        messages.extend(tool_messages)
    return messages


def _verify_exhaustion(messages: list[BaseMessage]) -> None:
    tool_bodies = [(m.content or "") for m in messages if getattr(m, "type", None) == "tool"]
    if any("correction budget exhausted" in body.lower() for body in tool_bodies):
        print("\n[verify] Found correction budget exhausted message.")
    else:
        print("\n[verify] Expected correction budget exhausted message, but none found.")


def main() -> int:
    _configure_logging()
    _maybe_register_trace_sink()

    exhaust = os.environ.get("ANCHOR_LANGCHAIN_DEMO_EXHAUST_BUDGET", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if exhaust:
        print("--- LangChain mock demo: exhaust self-correction budget ---")
        final_messages = run_with_script(scripted_exhaust_budget(), max_turns=5)
    else:
        print("--- LangChain mock demo: self_correct + strict_raise + response_log ---")
        final_messages = run_with_script(scripted_self_correct_and_strict_raise(), max_turns=8)

    for message in final_messages:
        _print_message(message)

    if exhaust:
        _verify_exhaustion(final_messages)

    print(
        "\nDone. Inspect `.trace/traces.sqlite` (unless disabled) or run `anchor report --hours 24`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
