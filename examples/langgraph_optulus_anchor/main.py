"""
Run the LangGraph + optulus-anchor example.

From the repository root (with the SDK installed editable)::

    pip install -e .
    pip install -r examples/langgraph_optulus_anchor/requirements.txt

Deterministic mock (no API keys) — from repo root::

    python -m examples.langgraph_optulus_anchor.main

Or from this directory (``examples/langgraph_optulus_anchor``)::

    python main.py

Live Groq (requires ``GROQ_API_KEY``)::

    ANCHOR_LANGGRAPH_USE_MOCK=0 python -m examples.langgraph_optulus_anchor.main

Environment:

- ``ANCHOR_LANGGRAPH_USE_MOCK``: ``1`` (default) runs a scripted fake model;
  ``0`` uses ``ChatGroq`` (``langchain-groq``).
- ``GROQ_API_KEY``: required for the live path (see https://console.groq.com/).
- ``ANCHOR_GROQ_MODEL``: optional model id (default ``llama-3.3-70b-versatile``).
- ``OPTULUS_ANCHOR_TRACE_DIR``: optional trace root for SQLite persistence.
- ``OPTULUS_ANCHOR_NO_TRACE``: set to disable SQLite trace persistence.
- ``ANCHOR_LANGGRAPH_DEMO_EXHAUST_BUDGET``: when ``1`` with the mock model, runs a
  scripted loop that keeps invalid ``reserve_table`` args until
  ``AnchorToolNode`` emits a **correction budget exhausted** tool result (see
  ``run_mock_demo_exhaust_correction_budget``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``python main.py`` sets ``__name__ == "__main__"`` but leaves ``__package__`` unset,
# which breaks relative imports. Align with ``python -m examples.langgraph_optulus_anchor.main``.
_main = sys.modules["__main__"]
if not getattr(_main, "__package__", None):
    _here = Path(__file__).resolve().parent
    # .../repo/examples/langgraph_optulus_anchor/main.py -> repo is parents[1]
    _repo = _here.parents[1]
    _src = _repo / "src"
    for p in (str(_repo), str(_src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _main.__package__ = "examples.langgraph_optulus_anchor"

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from optulus_anchor import set_trace_sink

from .graph import (
    build_agent_graph,
    run_mock_demo,
    run_mock_demo_exhaust_correction_budget,
)
from .tools import ALL_TOOLS as TOOLS_LIST

logger = logging.getLogger("examples.langgraph_optulus_anchor")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _print_message(m: BaseMessage) -> None:
    role = type(m).__name__.replace("Message", "")
    body = (m.content or "").strip()
    if body:
        print(f"\n[{role}]\n{body}")
    if isinstance(m, AIMessage) and m.tool_calls:
        print(f"[{role} tool_calls]")
        for tc in m.tool_calls:
            print(f"  - {tc['name']}({json.dumps(tc.get('args'), default=str)})")


def _maybe_register_trace_sink() -> None:
    if os.environ.get("ANCHOR_LANGGRAPH_PRINT_TRACES", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    def sink(entry: dict[str, Any]) -> None:
        print("[trace_sink]", json.dumps(entry, default=str))

    set_trace_sink(sink)
    logger.info("Registered set_trace_sink printer (ANCHOR_LANGGRAPH_PRINT_TRACES).")


def run_groq_demo() -> dict[str, Any]:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=os.environ.get("ANCHOR_GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(TOOLS_LIST)
    graph = build_agent_graph(llm=llm_with_tools, tools=TOOLS_LIST)

    user = (
        "You are testing a reservation and weather API. "
        "First, deliberately call reserve_table with party_size as the string 'two' "
        "and reservation_time as 'tomorrow at 7' (invalid). "
        "After you read the tool error, call reserve_table correctly with party_size=2, "
        "guest_name='Jordan', reservation_time='2026-04-15T19:00:00Z'. "
        "Then call get_weather with city as the integer 10001 (wrong type), read the error, "
        "and finally call get_weather with city='New York'. "
        "Summarize what happened."
    )

    return graph.invoke(
        {"messages": [HumanMessage(content=user)]},
        {"configurable": {"thread_id": "groq-demo"}, "recursion_limit": 40},
    )


def main() -> int:
    _configure_logging()
    _maybe_register_trace_sink()

    use_mock = os.environ.get("ANCHOR_LANGGRAPH_USE_MOCK", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
        "",
    }

    exhaust_demo = os.environ.get("ANCHOR_LANGGRAPH_DEMO_EXHAUST_BUDGET", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if use_mock and exhaust_demo:
        print(
            "--- Mock model: exhaust ``reserve_table`` correction budget "
            "(expect a ToolMessage containing 'correction budget exhausted') ---\n"
        )
        final = run_mock_demo_exhaust_correction_budget()
    elif use_mock:
        print("--- Mock model (FakeMessagesListChatModel) ---\n")
        final = run_mock_demo()
    else:
        if not os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY is required when ANCHOR_LANGGRAPH_USE_MOCK=0", file=sys.stderr)
            return 2
        print("--- Groq (ChatGroq) ---\n")
        final = run_groq_demo()

    for m in final["messages"]:
        _print_message(m)

    if use_mock and exhaust_demo:
        tool_bodies = [
            (m.content or "")
            for m in final["messages"]
            if getattr(m, "type", None) == "tool"
        ]
        if any("correction budget exhausted" in t.lower() for t in tool_bodies):
            print(
                "\n[verify] Attempt cap: found 'correction budget exhausted' in a tool result."
            )
        else:
            print(
                "\n[verify] Expected 'correction budget exhausted' in some tool message — not found.",
                file=sys.stderr,
            )

    print(
        "\nDone. Inspect `.trace/traces.sqlite` (unless disabled) or run `anchor report --hours 24`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
