"""
LangGraph agent backed by Groq, with tools wrapped in ``validate_tool``.

This shows how to keep LangChain ``StructuredTool`` definitions aligned with
Pydantic-validated inputs/outputs from optulus-anchor.

Install (in addition to ``optulus-anchor`` / ``pydantic``):

    pip install langgraph langchain-groq langchain-core

Run:

    export GROQ_API_KEY=gsk_...
    python examples/langgraph.py
"""

from __future__ import annotations

import os
import sys
import dotenv
from pathlib import Path

dotenv.load_dotenv()

# This file is named ``langgraph.py``; Python puts its directory first on
# ``sys.path``, which would make ``import langgraph`` load this script. Drop
# that entry so the real LangGraph package is imported.
_examples_dir = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _examples_dir:
    sys.path.pop(0)

# Allow running as ``python examples/langgraph.py`` without installing the package.
_root = _examples_dir.parent
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from optulus_anchor import ToolCorrectionNeeded, ToolValidationError, validate_tool


class MultiplyParams(BaseModel):
    """Arguments the model must pass to ``multiply_integers``."""

    a: int = Field(description="First factor")
    b: int = Field(description="Second factor")


class MultiplyResponse(BaseModel):
    product: int


class MismatchedMultiplyParams(BaseModel):
    """Intentionally wrong args schema to demonstrate PARAM_FAIL."""

    x: int = Field(description="Wrong parameter name for first factor")
    y: int = Field(description="Wrong parameter name for second factor")


class MismatchedChargeCustomerParams(BaseModel):
    """Intentionally wrong args schema to trigger self-correction."""

    amount: int = Field(description="Amount in cents")
    currency: str = Field(description="ISO currency code")
    customerid: str = Field(description="Wrong key; should be customer_id")


class LookupCustomerParams(BaseModel):
    customer_id: str = Field(description="Customer identifier")


class LookupCustomerResponse(BaseModel):
    name: str
    tier: str


class ChargeCardParams(BaseModel):
    customer_id: str
    amount_cents: int


class ChargeCardResponse(BaseModel):
    status: str
    payment_method_details: dict[str, str]


class ChargeCustomerParams(BaseModel):
    amount: int
    currency: str
    customer_id: str


class SearchDocsParams(BaseModel):
    query: str
    limit: int = 3


class SearchDocsResponse(BaseModel):
    results: list[str]


@validate_tool(
    params_schema=MultiplyParams,
    response_schema=MultiplyResponse,
    on_param_error="raise",
    on_response_error="log",
)
def multiply_integers(a: int, b: int) -> dict[str, int]:
    """Business logic for multiplication; validation runs around this call."""
    return {"product": a * b}


@validate_tool(
    params_schema=LookupCustomerParams,
    response_schema=LookupCustomerResponse,
    on_param_error="raise",
    on_response_error="raise",
)
def get_customer(customer_id: str) -> dict[str, str]:
    return {"name": f"Customer {customer_id}", "tier": "gold"}


@validate_tool(
    params_schema=ChargeCardParams,
    response_schema=ChargeCardResponse,
    on_param_error="raise",
    on_response_error="log",
)
def charge_card(customer_id: str, amount_cents: int) -> dict[str, object]:
    # Intentionally omit payment_method_details to demonstrate RESPONSE_FAIL drift.
    return {"status": f"charged:{customer_id}:{amount_cents}"}


@validate_tool(
    params_schema=SearchDocsParams,
    response_schema=SearchDocsResponse,
    on_param_error="raise",
    on_response_error="log",
)
def search_docs(query: str, limit: int = 3) -> dict[str, list[str]]:
    results = [f"{query}-result-{i}" for i in range(limit)]
    return {"results": results}


@validate_tool(
    params_schema=LookupCustomerParams,
    response_schema=LookupCustomerResponse,
    on_param_error="raise",
    on_response_error="raise",
)
def flaky_customer_lookup(customer_id: str) -> dict[str, str]:
    raise RuntimeError(f"downstream timeout for {customer_id}")


@validate_tool(
    params_schema=ChargeCustomerParams,
    on_param_error="self_correct",
    max_correction_attempts=2,
)
def charge_customer(amount: int, currency: str, customer_id: str) -> dict[str, str]:
    return {"status": f"charged:{customer_id}:{currency}:{amount}"}


multiply_tool = StructuredTool.from_function(
    func=multiply_integers,
    name="multiply_integers",
    description="Multiply two integers and return the product.",
    args_schema=MultiplyParams,
)


search_docs_tool = StructuredTool.from_function(
    func=search_docs,
    name="search_docs",
    description="Search docs and return matching snippets.",
    args_schema=SearchDocsParams,
)


mismatched_multiply_tool = StructuredTool.from_function(
    func=multiply_integers,
    name="multiply_integers_mismatch",
    description=(
        "Intentional failure demo: exposes x/y, but the wrapped function "
        "requires a/b and validate_tool should fail."
    ),
    args_schema=MismatchedMultiplyParams,
)


charge_customer_mismatch_tool = StructuredTool.from_function(
    func=charge_customer,
    name="charge_customer_mismatch",
    description=(
        "Intentional self-correction demo: exposes customerid, but validated "
        "tool expects customer_id and raises ToolCorrectionNeeded."
    ),
    args_schema=MismatchedChargeCustomerParams,
)


charge_customer_tool = StructuredTool.from_function(
    func=charge_customer,
    name="charge_customer",
    description="Charge a customer using amount, currency, and customer_id.",
    args_schema=ChargeCustomerParams,
)


def build_agent(tools: list[StructuredTool]):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Export it before running this example."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )
    return create_react_agent(llm, tools)


def _print_messages(label: str, messages: list[object]) -> None:
    print(f"\n=== {label} ===")
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", None)
        print(f"[{role}] {content}")


def run_llm_tool_fail_demo() -> None:
    """Run a graph where the LLM calls a tool with mismatched args."""
    fail_agent = build_agent([mismatched_multiply_tool])
    fail_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use multiply_integers_mismatch with x=12 and y=11. "
                        "Do not use any other tool."
                    )
                )
            ]
        },
        config={"recursion_limit": 5},
    )


def run_llm_self_correction_demo() -> None:
    """Run a graph where the LLM receives correction and retries."""
    print("\n=== LLM self-correction demo ===")
    messages: list[HumanMessage] = [
        HumanMessage(
            content=(
                "First call charge_customer_mismatch with amount=50, currency='usd', "
                "and customerid='cus_123'."
            )
        )
    ]

    first_attempt_agent = build_agent([charge_customer_mismatch_tool])
    retry_agent = build_agent([charge_customer_tool])

    try:
        first_attempt_agent.invoke({"messages": messages}, config={"recursion_limit": 5})
    except ToolCorrectionNeeded as exc:
        payload = exc.to_dict()
        print("Attempt 1 failed as expected; forwarding correction prompt to LLM:")
        print(payload["correction_prompt"])

        messages.append(
            HumanMessage(
                content=(
                    "Your previous tool call failed validation. "
                    f"{payload['correction_prompt']}\n"
                    "Retry now by calling charge_customer with corrected arguments only."
                )
            )
        )

        retry_result = retry_agent.invoke({"messages": messages}, config={"recursion_limit": 5})
        _print_messages("LLM retry after correction prompt", retry_result["messages"])
        return

    print("Expected ToolCorrectionNeeded was not raised on first attempt.")


def run_local_trace_demo() -> None:
    """
    Trigger several trace outcomes without requiring the LLM to decide tool args.

    This writes PASS / PARAM_FAIL / RESPONSE_FAIL / EXECUTION_FAIL events into
    `.trace/traces.sqlite` (or OPTULUS_ANCHOR_TRACE_DIR override).
    """
    print("\n=== Local trace demo ===")

    # PASS
    print("PASS:", get_customer(customer_id="cus_123"))
    print("PASS:", search_docs(query="anchor report", limit=2))

    # PARAM_FAIL (missing required field)
    try:
        get_customer(wrong_id="oops")  # type: ignore[call-arg]
    except ToolValidationError as exc:
        print("PARAM_FAIL captured:", exc)

    # RESPONSE_FAIL (schema drift)
    charge_result = charge_card(customer_id="cus_123", amount_cents=1999)
    print("RESPONSE_FAIL logged, function still returned:", charge_result)

    # EXECUTION_FAIL
    try:
        flaky_customer_lookup(customer_id="cus_timeout")
    except RuntimeError as exc:
        print("EXECUTION_FAIL captured:", exc)


def run_self_correction_handoff_demo() -> None:
    """
    Demonstrate framework-agnostic self-correction handoff.

    In real integrations, an agent runtime catches ToolCorrectionNeeded and uses
    correction_prompt to ask the LLM for a corrected tool call.
    """
    print("\n=== Self-correction handoff demo ===")

    correction_history: list[dict[str, object]] = []
    for attempt in range(1, 3):
        if attempt == 1:
            tool_kwargs: dict[str, object] = {
                "amount": "50",
                "currency": "usd",
                "customerid": "cus_123",
            }
        else:
            tool_kwargs = {
                "amount": 50,
                "currency": "usd",
                "customer_id": "cus_123",
            }

        try:
            result = charge_customer(
                **tool_kwargs,
                __tool_correction_attempt=attempt,
                __tool_correction_history=correction_history,
            )
            print("Self-corrected call succeeded:", result)
            return
        except ToolCorrectionNeeded as exc:
            payload = exc.to_dict()
            correction_history = payload["correction_history"]
            print("Correction needed:")
            print(payload["correction_prompt"])

    print("No corrected call produced within max attempts.")


def main() -> None:
    run_local_trace_demo()
    run_self_correction_handoff_demo()

    run_llm_self_correction_demo()
    run_llm_tool_fail_demo()



if __name__ == "__main__":
    main()
