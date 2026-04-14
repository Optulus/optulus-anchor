from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, StrictInt

from optulus_anchor import ToolCorrectionNeeded, validate_tool


class ChargeParams(BaseModel):
    amount: StrictInt
    currency: str
    customer_id: str


def test_self_correct_raises_on_bind_failure_with_structured_payload() -> None:
    calls = {"count": 0}

    @validate_tool(
        params_schema=ChargeParams,
        on_param_error="self_correct",
        max_correction_attempts=2,
    )
    def charge_customer(amount: int, currency: str, customer_id: str) -> dict[str, str]:
        calls["count"] += 1
        return {"status": f"{customer_id}:{currency}:{amount}"}

    with pytest.raises(ToolCorrectionNeeded) as exc_info:
        charge_customer(amount="50", currency="usd", customerid="cus_123")  # type: ignore[call-arg,arg-type]

    exc = exc_info.value
    assert calls["count"] == 0
    assert exc.tool_name == "charge_customer"
    assert exc.attempt == 1
    assert exc.max_attempts == 2
    assert "customerid" in exc.correction_prompt
    assert "Expected schema" in exc.correction_prompt
    payload = exc.to_dict()
    json.dumps(payload)
    assert len(payload["correction_history"]) == 1


def test_self_correct_raises_on_schema_failure_with_attempt_context() -> None:
    @validate_tool(
        params_schema=ChargeParams,
        on_param_error="self_correct",
        max_correction_attempts=3,
    )
    def charge_customer(amount: int, currency: str, customer_id: str) -> dict[str, str]:
        return {"status": f"{customer_id}:{currency}:{amount}"}

    prior_history = [{"attempt": 1, "errors": ["amount: bad type"]}]

    with pytest.raises(ToolCorrectionNeeded) as exc_info:
        charge_customer(
            amount="50",  # type: ignore[arg-type]
            currency="usd",
            customer_id="cus_123",
            __tool_correction_attempt=2,
            __tool_correction_history=prior_history,
        )

    exc = exc_info.value
    assert exc.attempt == 2
    assert exc.max_attempts == 3
    assert "amount" in " ".join(exc.errors)
    assert len(exc.correction_history) == 2
    assert exc.correction_history[0] == prior_history[0]
    assert exc.correction_history[1]["attempt"] == 2


def test_self_correct_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_correction_attempts must be >= 1"):

        @validate_tool(
            params_schema=ChargeParams,
            on_param_error="self_correct",
            max_correction_attempts=0,
        )
        def charge_customer(amount: int, currency: str, customer_id: str) -> dict[str, str]:
            return {"status": f"{customer_id}:{currency}:{amount}"}

