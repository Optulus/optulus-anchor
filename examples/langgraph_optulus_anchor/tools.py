"""
Validated tools for the LangGraph demo.

- ``reserve_table``: ``on_param_error="self_correct"`` — bad model args raise
  ``ToolCorrectionNeeded`` so the graph can feed ``correction_prompt`` back to the model.
- ``get_weather``: ``on_param_error="raise"`` — invalid args raise ``ToolValidationError``;
  the graph surfaces that as a ``ToolMessage`` so the model can fix the call.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from optulus_anchor import validate_tool

# LangChain validates ``args_schema`` before your function runs. Use a permissive
# surface schema so malformed model output still reaches ``validate_tool``.

_ISO8601_Z = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


class ReserveTableParams(BaseModel):
    """Restaurant reservation — strict enough that models often slip on the first try."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    party_size: int = Field(..., ge=1, le=20, description="Head count (integer).")
    reservation_time: str = Field(
        ...,
        description="UTC instant in ISO-8601 ending with Z, e.g. 2026-04-15T19:00:00Z.",
    )
    guest_name: str = Field(..., min_length=1, max_length=80)

    @field_validator("reservation_time")
    @classmethod
    def utc_zulu(cls, v: str) -> str:
        if not _ISO8601_Z.match(v):
            raise ValueError("reservation_time must be ISO-8601 in UTC with a Z suffix")
        return v


class ReserveTableResponse(BaseModel):
    confirmation_code: str
    party_size: int
    reservation_time: str
    guest_name: str


@validate_tool(
    params_schema=ReserveTableParams,
    response_schema=ReserveTableResponse,
    on_param_error="self_correct",
    on_response_error="log",
    max_correction_attempts=3,
)
def reserve_table_impl(
    party_size: Any,
    reservation_time: Any,
    guest_name: Any,
) -> dict[str, Any]:
    code = f"RT-{guest_name[:3].upper()}-{party_size:02d}"
    return {
        "confirmation_code": code,
        "party_size": party_size,
        "reservation_time": reservation_time,
        "guest_name": guest_name,
    }


class ReserveTableLcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    party_size: Any = Field(..., description="Integer head count.")
    reservation_time: Any = Field(..., description="ISO-8601 UTC string ending with Z.")
    guest_name: Any = Field(..., description="Guest name string.")


reserve_table = StructuredTool.from_function(
    name="reserve_table",
    description=(
        "Reserve a restaurant table. party_size must be an integer. "
        "reservation_time must be full ISO-8601 UTC with Z (not 'tomorrow 7pm')."
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
    # Deterministic stub — real code would call an API.
    seed = sum(ord(c) for c in city) % 7
    return {"city": city.title(), "celsius": float(18 + seed), "condition": "clear"}


class WeatherLcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: Any = Field(..., description="City name as a string, e.g. 'New York'.")


get_weather = StructuredTool.from_function(
    name="get_weather",
    description="Return a stub weather forecast for a city name (string).",
    func=get_weather_impl,
    args_schema=WeatherLcArgs,
)


ALL_TOOLS: list[StructuredTool] = [reserve_table, get_weather]
