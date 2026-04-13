from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from optulus_anchor import SchemaDriftError, ToolValidationError, validate_tool


class SimpleResponse(BaseModel):
    ok: bool


class ProfileParams(BaseModel):
    first_name: str
    last_name: str
    age: int
    middle_name: str | None = None


@validate_tool(params_schema=ProfileParams, response_schema=SimpleResponse)
def upsert_profile(
    first_name: str,
    last_name: str,
    age: int,
    middle_name: str | None = None,
) -> dict[str, bool]:
    return {"ok": True}


class ShippingAddress(BaseModel):
    line1: str
    city: str
    postal_code: str | None = None


class LineItem(BaseModel):
    sku: str
    quantity: int


class CreateOrderParams(BaseModel):
    customer_id: str
    shipping: ShippingAddress
    items: list[LineItem]


@validate_tool(params_schema=CreateOrderParams, response_schema=SimpleResponse)
def create_order(
    customer_id: str,
    shipping: dict[str, object],
    items: list[dict[str, object]],
) -> dict[str, bool]:
    return {"ok": True}


def test_input_parameter_different_ordering_still_validates() -> None:
    result = upsert_profile(last_name="Lhamo", age=31, first_name="Kinzang")
    assert result["ok"] is True


def test_input_parameter_null_for_non_nullable_field_fails() -> None:
    with pytest.raises(ToolValidationError):
        upsert_profile(first_name=None, last_name="Lhamo", age=31)  # type: ignore[arg-type]


def test_input_parameter_null_for_nullable_field_passes() -> None:
    result = upsert_profile(
        first_name="Kinzang",
        last_name="Lhamo",
        age=31,
        middle_name=None,
    )
    assert result["ok"] is True


def test_input_parameter_wrong_type_fails() -> None:
    with pytest.raises(ToolValidationError):
        upsert_profile(first_name="Kinzang", last_name="Lhamo", age="thirty-one")  # type: ignore[arg-type]


def test_nested_parameter_ordering_still_validates() -> None:
    result = create_order(
        items=[
            {"quantity": 2, "sku": "sku-001"},
            {"sku": "sku-002", "quantity": 1},
        ],
        customer_id="cus_123",
        shipping={"city": "Thimphu", "postal_code": None, "line1": "Changzamtok"},
    )
    assert result["ok"] is True


class StrictProfileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str


@validate_tool(params_schema=StrictProfileParams, response_schema=SimpleResponse)
def create_strict_profile(first_name: str, last_name: str) -> dict[str, bool]:
    return {"ok": True}


def test_extra_unknown_field_fails_in_strict_schema() -> None:
    with pytest.raises(ToolValidationError):
        create_strict_profile(
            first_name="Kinzang",
            last_name="Lhamo",
            nickname="KZ",  # type: ignore[call-arg]
        )


def test_missing_nested_required_field_fails() -> None:
    with pytest.raises(ToolValidationError):
        create_order(
            customer_id="cus_123",
            shipping={"city": "Thimphu"},  # missing line1
            items=[{"sku": "sku-001", "quantity": 1}],
        )


def test_null_inside_nested_non_nullable_list_field_fails() -> None:
    with pytest.raises(ToolValidationError):
        create_order(
            customer_id="cus_123",
            shipping={"line1": "Changzamtok", "city": "Thimphu"},
            items=[{"sku": "sku-001", "quantity": None}],  # type: ignore[list-item]
        )


def test_param_validation_can_log_and_continue(caplog: pytest.LogCaptureFixture) -> None:
    @validate_tool(
        params_schema=StrictProfileParams,
        response_schema=SimpleResponse,
        on_param_error="log",
    )
    def relaxed_profile(first_name: str, last_name: str, **kwargs: object) -> dict[str, bool]:
        return {"ok": True}

    with caplog.at_level("WARNING", logger="optulus_anchor.tool_validator"):
        result = relaxed_profile(first_name="Kinzang", last_name="Lhamo", nickname="KZ")

    assert result["ok"] is True
    combined = " ".join(record.message for record in caplog.records)
    assert '"status": "PARAM_FAIL"' in combined


def test_nested_response_drift_raises_when_configured() -> None:
    class NestedResponse(BaseModel):
        ok: bool
        payload: ShippingAddress

    @validate_tool(
        params_schema=ProfileParams,
        response_schema=NestedResponse,
        on_response_error="raise",
    )
    def broken_nested_response(
        first_name: str, last_name: str, age: int, middle_name: str | None = None
    ) -> dict[str, object]:
        return {
            "ok": True,
            "payload": {"line1": "Changzamtok"},  # missing city
        }

    with pytest.raises(SchemaDriftError):
        broken_nested_response(first_name="Kinzang", last_name="Lhamo", age=31)
