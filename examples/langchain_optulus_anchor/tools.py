"""
Validated tools for the LangChain + AnchorToolExecutor demo.

- ``reserve_table`` uses ``on_param_error="self_correct"``.
- ``get_weather`` uses strict ``on_param_error="raise"``.
- ``search_docs`` uses ``on_response_error="log"`` to demonstrate response drift logging.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from optulus_anchor import validate_tool

_ISO8601_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ReserveTableParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    party_size: int = Field(..., ge=1, le=20)
    reservation_time: str = Field(
        ...,
        description="UTC ISO-8601 with Z suffix, e.g. 2026-04-15T19:00:00Z.",
    )
    guest_name: str = Field(..., min_length=1, max_length=80)

    @field_validator("reservation_time")
    @classmethod
    def utc_zulu(cls, value: str) -> str:
        if not _ISO8601_Z.match(value):
            raise ValueError("reservation_time must be ISO-8601 in UTC with a Z suffix")
        return value


class ReserveTableResponse(BaseModel):
    confirmation_code: str
    party_size: int
    reservation_time: str
    guest_name: str


@validate_tool(
    params_schema=ReserveTableParams,
    response_schema=ReserveTableResponse,
    on_param_error="self_correct",
    on_response_error="raise",
    max_correction_attempts=2,
)
def reserve_table_impl(
    party_size: Any, reservation_time: Any, guest_name: Any
) -> dict[str, Any]:
    code = f"RT-{str(guest_name)[:3].upper()}-{int(party_size):02d}"
    return {
        "confirmation_code": code,
        "party_size": party_size,
        "reservation_time": reservation_time,
        "guest_name": guest_name,
    }


class ReserveTableLcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    party_size: Any
    reservation_time: Any
    guest_name: Any


reserve_table = StructuredTool.from_function(
    name="reserve_table",
    description=(
        "Reserve a table. party_size must be integer and reservation_time must be "
        "ISO-8601 UTC with Z."
    ),
    func=reserve_table_impl,
    args_schema=ReserveTableLcArgs,
)


class WeatherParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    city: str = Field(..., min_length=1, max_length=64)


class WeatherResponse(BaseModel):
    city: str
    celsius: float
    condition: str


@validate_tool(
    params_schema=WeatherParams,
    response_schema=WeatherResponse,
    on_param_error="raise",
    on_response_error="raise",
)
def get_weather_impl(city: Any) -> dict[str, Any]:
    seed = sum(ord(c) for c in city) % 7
    return {"city": city.title(), "celsius": float(18 + seed), "condition": "clear"}


class WeatherLcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: Any


get_weather = StructuredTool.from_function(
    name="get_weather",
    description="Return a deterministic weather stub for a city name string.",
    func=get_weather_impl,
    args_schema=WeatherLcArgs,
)


class SearchDocsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    limit: int = 3


class SearchDocsResponse(BaseModel):
    results: list[str]
    count: int


@validate_tool(
    params_schema=SearchDocsParams,
    response_schema=SearchDocsResponse,
    on_param_error="raise",
    on_response_error="log",
)
def search_docs_impl(query: str, limit: int = 3) -> dict[str, Any]:
    # Intentionally missing "count" to trigger RESPONSE_FAIL while still returning.
    return {"results": [f"{query}-result-{i}" for i in range(limit)]}


class SearchDocsLcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Any
    limit: Any = 3


search_docs = StructuredTool.from_function(
    name="search_docs",
    description="Search docs by query and return result snippets.",
    func=search_docs_impl,
    args_schema=SearchDocsLcArgs,
)


ALL_TOOLS: list[StructuredTool] = [reserve_table, get_weather, search_docs]
