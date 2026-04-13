from __future__ import annotations

import logging

import pytest
from hypothesis import given  # pyright: ignore[reportMissingImports]
from hypothesis import strategies as st  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from optulus_anchor import SchemaDriftError, validate_tool


class BasicParams(BaseModel):
    request_id: str


class NestedMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: StrictInt
    source: StrictStr
    tags: list[StrictStr]


class ItemResult(BaseModel):
    sku: StrictStr
    quantity: StrictInt


class StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StrictStr
    message: str | None = None
    meta: NestedMeta
    items: list[ItemResult]


@validate_tool(
    params_schema=BasicParams,
    response_schema=StrictResponse,
    on_response_error="raise",
)
def strict_tool_response(request_id: str, response_payload: dict[str, object]) -> dict[str, object]:
    return response_payload


@validate_tool(params_schema=BasicParams, response_schema=StrictResponse)
def log_only_tool_response(request_id: str, response_payload: dict[str, object]) -> dict[str, object]:
    return response_payload


def test_response_nested_different_ordering_passes() -> None:
    payload = {
        "items": [{"quantity": 2, "sku": "sku-001"}],
        "meta": {"tags": ["priority"], "source": "api", "created_at": 1700000000},
        "message": None,
        "status": "ok",
    }
    result = strict_tool_response(request_id="req-1", response_payload=payload)
    assert result["status"] == "ok"


def test_response_null_for_non_nullable_field_fails() -> None:
    payload = {
        "status": "ok",
        "message": None,
        "meta": {"created_at": None, "source": "api", "tags": ["priority"]},
        "items": [{"sku": "sku-001", "quantity": 1}],
    }
    with pytest.raises(SchemaDriftError):
        strict_tool_response(request_id="req-1", response_payload=payload)


def test_response_null_for_nullable_field_passes() -> None:
    payload = {
        "status": "ok",
        "message": None,
        "meta": {"created_at": 1700000000, "source": "api", "tags": ["priority"]},
        "items": [{"sku": "sku-001", "quantity": 1}],
    }
    result = strict_tool_response(request_id="req-1", response_payload=payload)
    assert result["status"] == "ok"


def test_response_wrong_type_fails() -> None:
    payload = {
        "status": "ok",
        "message": "done",
        "meta": {"created_at": 1700000000, "source": "api", "tags": ["priority"]},
        "items": [{"sku": "sku-001", "quantity": "2"}],
    }
    with pytest.raises(SchemaDriftError):
        strict_tool_response(request_id="req-1", response_payload=payload)


def test_response_missing_nested_required_field_fails() -> None:
    payload = {
        "status": "ok",
        "message": "done",
        "meta": {"created_at": 1700000000, "source": "api"},  # missing tags
        "items": [{"sku": "sku-001", "quantity": 2}],
    }
    with pytest.raises(SchemaDriftError):
        strict_tool_response(request_id="req-1", response_payload=payload)


def test_response_extra_field_fails_with_strict_response_model() -> None:
    payload = {
        "status": "ok",
        "message": "done",
        "meta": {
            "created_at": 1700000000,
            "source": "api",
            "tags": ["priority"],
            "unexpected": "value",
        },
        "items": [{"sku": "sku-001", "quantity": 2}],
    }
    with pytest.raises(SchemaDriftError):
        strict_tool_response(request_id="req-1", response_payload=payload)


def test_response_default_log_policy_logs_and_returns_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "status": "ok",
        "message": "done",
        "meta": {"created_at": 1700000000, "source": "api"},  # missing tags
        "items": [{"sku": "sku-001", "quantity": 2}],
    }
    with caplog.at_level(logging.WARNING, logger="optulus_anchor.tool_validator"):
        result = log_only_tool_response(request_id="req-1", response_payload=payload)

    assert result["status"] == "ok"
    combined = " ".join(record.message for record in caplog.records)
    assert '"status": "RESPONSE_FAIL"' in combined
    assert "meta.tags" in combined


valid_response_payloads = st.fixed_dictionaries(
    {
        "status": st.text(min_size=1, max_size=16),
        "message": st.one_of(st.none(), st.text(max_size=40)),
        "meta": st.fixed_dictionaries(
            {
                "created_at": st.integers(min_value=0, max_value=2_000_000_000),
                "source": st.text(min_size=1, max_size=16),
                "tags": st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=5),
            }
        ),
        "items": st.lists(
            st.fixed_dictionaries(
                {
                    "sku": st.text(min_size=1, max_size=16),
                    "quantity": st.integers(min_value=1, max_value=50),
                }
            ),
            min_size=1,
            max_size=5,
        ),
    }
)

invalid_response_payloads = st.one_of(
    st.fixed_dictionaries(
        {
            "status": st.text(min_size=1, max_size=16),
            "message": st.one_of(st.none(), st.text(max_size=40)),
            "meta": st.fixed_dictionaries(
                {
                    "created_at": st.integers(min_value=0, max_value=2_000_000_000),
                    "source": st.text(min_size=1, max_size=16),
                    # missing tags
                }
            ),
            "items": st.lists(
                st.fixed_dictionaries(
                    {
                        "sku": st.text(min_size=1, max_size=16),
                        "quantity": st.integers(min_value=1, max_value=50),
                    }
                ),
                min_size=1,
                max_size=5,
            ),
        }
    ),
    st.fixed_dictionaries(
        {
            "status": st.text(min_size=1, max_size=16),
            "message": st.one_of(st.none(), st.text(max_size=40)),
            "meta": st.fixed_dictionaries(
                {
                    "created_at": st.integers(min_value=0, max_value=2_000_000_000),
                    "source": st.text(min_size=1, max_size=16),
                    "tags": st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=5),
                }
            ),
            "items": st.lists(
                st.fixed_dictionaries(
                    {
                        "sku": st.text(min_size=1, max_size=16),
                        "quantity": st.text(min_size=1, max_size=4),
                    }
                ),
                min_size=1,
                max_size=5,
            ),
        }
    ),
)


@given(payload=valid_response_payloads)
def test_fuzz_valid_responses_pass(payload: dict[str, object]) -> None:
    result = strict_tool_response(request_id="req-fuzz", response_payload=payload)
    assert result["status"]


@given(payload=invalid_response_payloads)
def test_fuzz_invalid_responses_raise(payload: dict[str, object]) -> None:
    with pytest.raises(SchemaDriftError):
        strict_tool_response(request_id="req-fuzz", response_payload=payload)
