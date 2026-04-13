from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import BaseModel

from optulus_anchor import SchemaDriftError, ToolValidationError, validate_tool


class GetCustomerParams(BaseModel):
    customer_id: str


class CustomerResponse(BaseModel):
    id: str
    email: str
    created_at: int


@validate_tool(params_schema=GetCustomerParams, response_schema=CustomerResponse)
def get_customer(customer_id: str) -> dict[str, object]:
    return {"id": customer_id, "email": "user@example.com", "created_at": 1700000000}


@validate_tool(params_schema=GetCustomerParams, response_schema=CustomerResponse)
async def get_customer_async(customer_id: str) -> dict[str, object]:
    await asyncio.sleep(0)
    return {"id": customer_id, "email": "user@example.com", "created_at": 1700000000}


def test_validate_tool_accepts_positional_args() -> None:
    result = get_customer("cus_123")
    assert result["id"] == "cus_123"


def test_validate_tool_raises_on_param_error() -> None:
    with pytest.raises(ToolValidationError):
        get_customer(customerId="cus_123")  # type: ignore[call-arg]


def test_validate_tool_logs_response_drift_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @validate_tool(params_schema=GetCustomerParams, response_schema=CustomerResponse)
    def broken_customer(customer_id: str) -> dict[str, object]:
        return {"id": customer_id, "email": "user@example.com", "created_time": 1}

    with caplog.at_level(logging.WARNING, logger="optulus_anchor.tool_validator"):
        result = broken_customer(customer_id="cus_123")

    assert result["id"] == "cus_123"
    combined = " ".join(record.message for record in caplog.records)
    assert '"status": "RESPONSE_FAIL"' in combined
    assert "created_at" in combined


def test_validate_tool_can_raise_on_response_error() -> None:
    @validate_tool(
        params_schema=GetCustomerParams,
        response_schema=CustomerResponse,
        on_response_error="raise",
    )
    def broken_customer(customer_id: str) -> dict[str, object]:
        return {"id": customer_id, "email": "user@example.com", "created_time": 1}

    with pytest.raises(SchemaDriftError):
        broken_customer(customer_id="cus_123")


def test_validate_tool_supports_async_functions() -> None:
    result = asyncio.run(get_customer_async("cus_999"))
    assert result["id"] == "cus_999"
