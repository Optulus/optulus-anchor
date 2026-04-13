from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def _format_pydantic_errors(exc: PydanticValidationError) -> list[str]:
    formatted: list[str] = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", [])) or "root"
        formatted.append(f"{location}: {err.get('msg', 'validation error')}")
    return formatted


def _validate_with_schema(payload: Any, schema: type[Any]) -> None:
    if hasattr(schema, "model_validate"):
        schema.model_validate(payload)
        return

    if isinstance(payload, dict):
        schema(**payload)
        return

    schema(payload)


def validate_params(params: dict[str, Any], schema: type[Any]) -> ValidationResult:
    try:
        _validate_with_schema(params, schema)
        return ValidationResult(valid=True, errors=[])
    except PydanticValidationError as exc:
        return ValidationResult(valid=False, errors=_format_pydantic_errors(exc))


def validate_response(response: Any, schema: type[Any]) -> ValidationResult:
    try:
        _validate_with_schema(response, schema)
        return ValidationResult(valid=True, errors=[])
    except PydanticValidationError as exc:
        return ValidationResult(valid=False, errors=_format_pydantic_errors(exc))
