from __future__ import annotations

import pytest
from hypothesis import given  # pyright: ignore[reportMissingImports]
from hypothesis import strategies as st  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, StrictInt, StrictStr

from optulus_anchor import ToolValidationError, validate_tool


class FuzzAddress(BaseModel):
    line1: str
    city: str
    postal_code: str | None = None


class FuzzLineItem(BaseModel):
    sku: StrictStr
    quantity: StrictInt


class FuzzOrderParams(BaseModel):
    customer_id: StrictStr
    shipping: FuzzAddress
    items: list[FuzzLineItem]
    notes: str | None = None


class OkResponse(BaseModel):
    ok: bool


@validate_tool(params_schema=FuzzOrderParams, response_schema=OkResponse)
def submit_order(
    customer_id: str,
    shipping: dict[str, object],
    items: list[dict[str, object]],
    notes: str | None = None,
) -> dict[str, bool]:
    return {"ok": True}


valid_order_payloads = st.fixed_dictionaries(
    {
        "customer_id": st.text(min_size=1, max_size=20),
        "shipping": st.fixed_dictionaries(
            {
                "line1": st.text(min_size=1, max_size=30),
                "city": st.text(min_size=1, max_size=30),
                "postal_code": st.one_of(st.none(), st.text(min_size=1, max_size=12)),
            }
        ),
        "items": st.lists(
            st.fixed_dictionaries(
                {
                    "sku": st.text(min_size=1, max_size=20),
                    "quantity": st.integers(min_value=1, max_value=100),
                }
            ),
            min_size=1,
            max_size=5,
        ),
        "notes": st.one_of(st.none(), st.text(max_size=50)),
    }
)

invalid_order_payloads = st.one_of(
    # Wrong primitive type on top-level field.
    st.fixed_dictionaries(
        {
            "customer_id": st.integers(min_value=0, max_value=1000),
            "shipping": st.fixed_dictionaries(
                {
                    "line1": st.text(min_size=1, max_size=20),
                    "city": st.text(min_size=1, max_size=20),
                    "postal_code": st.none(),
                }
            ),
            "items": st.lists(
                st.fixed_dictionaries(
                    {
                        "sku": st.text(min_size=1, max_size=20),
                        "quantity": st.integers(min_value=1, max_value=5),
                    }
                ),
                min_size=1,
                max_size=3,
            ),
            "notes": st.none(),
        }
    ),
    # Missing nested required field.
    st.fixed_dictionaries(
        {
            "customer_id": st.text(min_size=1, max_size=20),
            "shipping": st.fixed_dictionaries(
                {
                    "city": st.text(min_size=1, max_size=20),
                    "postal_code": st.none(),
                }
            ),
            "items": st.lists(
                st.fixed_dictionaries(
                    {
                        "sku": st.text(min_size=1, max_size=20),
                        "quantity": st.integers(min_value=1, max_value=5),
                    }
                ),
                min_size=1,
                max_size=3,
            ),
            "notes": st.none(),
        }
    ),
    # Wrong nested type in list.
    st.fixed_dictionaries(
        {
            "customer_id": st.text(min_size=1, max_size=20),
            "shipping": st.fixed_dictionaries(
                {
                    "line1": st.text(min_size=1, max_size=20),
                    "city": st.text(min_size=1, max_size=20),
                    "postal_code": st.none(),
                }
            ),
            "items": st.lists(
                st.fixed_dictionaries(
                    {
                        "sku": st.text(min_size=1, max_size=20),
                        "quantity": st.text(min_size=1, max_size=5),
                    }
                ),
                min_size=1,
                max_size=3,
            ),
            "notes": st.none(),
        }
    ),
)


@given(payload=valid_order_payloads)
def test_fuzz_valid_nested_payloads_pass(payload: dict[str, object]) -> None:
    result = submit_order(**payload)
    assert result["ok"] is True


@given(payload=invalid_order_payloads)
def test_fuzz_invalid_nested_payloads_fail(payload: dict[str, object]) -> None:
    with pytest.raises(ToolValidationError):
        submit_order(**payload)

