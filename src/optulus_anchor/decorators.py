from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

from optulus_anchor.arg_binding import bind_arguments
from optulus_anchor.exceptions import (
    SchemaDriftError,
    ToolCorrectionNeeded,
    ToolValidationError,
)
from optulus_anchor.logger import log_trace
from optulus_anchor.validator import validate_params, validate_response

F = TypeVar("F", bound=Callable[..., Any])
ParamErrorPolicy = Literal["raise", "log", "warn", "self_correct"]
ResponseErrorPolicy = Literal["raise", "log", "warn"]


def validate_tool(
    *,
    params_schema: type[Any] | None = None,
    response_schema: type[Any] | None = None,
    on_param_error: ParamErrorPolicy = "raise",
    on_response_error: ResponseErrorPolicy = "log",
    max_correction_attempts: int = 2,
) -> Callable[[F], F]:
    """
    Decorate a tool function with schema-based input/output validation and trace logging.

    This decorator helps catch malformed tool arguments (often caused by agent
    hallucination) and response shape drift (often caused by external API changes)
    before those issues silently propagate through an agent workflow.

    Args:
        params_schema: Optional schema class used to validate incoming arguments.
            Validation runs before the wrapped function executes. The bound argument
            mapping is passed to the schema.
        response_schema: Optional schema class used to validate the function result.
            Validation runs after the wrapped function returns.
        on_param_error: Policy for parameter validation failures.
            - ``"raise"``: raise ``ToolValidationError`` and stop execution.
            - ``"log"``: record failure and continue.
            - ``"warn"``: record failure and continue.
            - ``"self_correct"``: raise ``ToolCorrectionNeeded`` with a
              structured correction payload for external LLM retry orchestration.
        on_response_error: Policy for response validation failures.
            - ``"raise"``: raise ``SchemaDriftError``.
            - ``"log"``: record failure and continue.
            - ``"warn"``: record failure and continue.
        max_correction_attempts: Max retry attempts available to upstream
            orchestrators when ``on_param_error="self_correct"``. The SDK does
            not perform retries itself; this value is included in the raised
            payload for framework-managed loops.

    Returns:
        A decorator that preserves the wrapped function signature and returns a
        wrapped callable with validation and tracing behavior.

    Example:
        ```python
        from pydantic import BaseModel
        from optulus_anchor import ToolCorrectionNeeded, validate_tool

        class SearchParams(BaseModel):
            query: str
            max_results: int = 5

        class SearchResponse(BaseModel):
            results: list[str]

        @validate_tool(
            params_schema=SearchParams,
            response_schema=SearchResponse,
            on_param_error="raise",
            on_response_error="log",
        )
        def search_documents(query: str, max_results: int = 5) -> dict[str, list[str]]:
            return {"results": [f"Result for {query}"][:max_results]}

        @validate_tool(
            params_schema=SearchParams,
            on_param_error="self_correct",
            max_correction_attempts=2,
        )
        def search_documents_with_correction(query: str, max_results: int = 5) -> dict[str, list[str]]:
            return {"results": [f"Result for {query}"][:max_results]}

        try:
            search_documents_with_correction(query=123)  # type: ignore[arg-type]
        except ToolCorrectionNeeded as exc:
            payload = exc.to_dict()
            correction_prompt = payload["correction_prompt"]
            # Forward correction_prompt into your LLM orchestration loop.
        ```
    """
    if on_param_error not in {"raise", "log", "warn", "self_correct"}:
        raise ValueError("on_param_error must be one of: raise, log, warn, self_correct")
    if on_response_error not in {"raise", "log", "warn"}:
        raise ValueError("on_response_error must be one of: raise, log, warn")
    if max_correction_attempts < 1:
        raise ValueError("max_correction_attempts must be >= 1")

    def _extract_schema_fields(schema: type[Any] | None) -> dict[str, str]:
        if schema is None:
            return {}

        fields = getattr(schema, "model_fields", None)
        if isinstance(fields, dict):
            extracted: dict[str, str] = {}
            for name, field in fields.items():
                annotation = getattr(field, "annotation", Any)
                extracted[name] = getattr(annotation, "__name__", str(annotation))
            return extracted

        return {}

    def _format_correction_prompt(
        *,
        tool_name: str,
        attempted_params: dict[str, Any] | None,
        errors: list[str],
    ) -> str:
        return (
            f"Tool '{tool_name}' parameter validation failed.\n\n"
            f"Attempted parameters:\n"
            f"{json.dumps(attempted_params or {}, indent=2, default=str)}\n\n"
            f"Errors:\n"
            f"{chr(10).join(f'- {error}' for error in errors)}\n\n"
            f"Expected schema:\n"
            f"{json.dumps(_extract_schema_fields(params_schema), indent=2)}\n\n"
            "Please retry the tool call with corrected parameters."
        )

    def _extract_correction_context(kwargs: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        attempt_raw = kwargs.pop("__tool_correction_attempt", 1)
        history_raw = kwargs.pop("__tool_correction_history", [])

        attempt = attempt_raw if isinstance(attempt_raw, int) and attempt_raw >= 1 else 1
        history = history_raw if isinstance(history_raw, list) else []
        return attempt, history

    def _handle_param_failure(
        *,
        tool_name: str,
        errors: list[str],
        attempted_params: dict[str, Any] | None,
        correction_attempt: int,
        correction_history: list[dict[str, Any]],
    ) -> None:
        log_trace(
            tool_name,
            "PARAM_FAIL",
            errors=errors,
            params_valid=False,
        )
        if on_param_error == "self_correct":
            correction_prompt = _format_correction_prompt(
                tool_name=tool_name,
                attempted_params=attempted_params,
                errors=errors,
            )
            history_with_attempt = [
                *correction_history,
                {
                    "attempt": correction_attempt,
                    "attempted_params": attempted_params,
                    "errors": errors,
                },
            ]
            raise ToolCorrectionNeeded(
                tool_name=tool_name,
                attempt=correction_attempt,
                max_attempts=max_correction_attempts,
                attempted_params=attempted_params,
                errors=errors,
                correction_prompt=correction_prompt,
                correction_history=history_with_attempt,
            )
        if on_param_error == "raise":
            raise ToolValidationError(
                f"Parameter validation failed for {tool_name}: {errors}"
            )

    def _handle_response_failure(
        tool_name: str, errors: list[str], latency_ms: float
    ) -> None:
        log_trace(
            tool_name,
            "RESPONSE_FAIL",
            errors=errors,
            latency_ms=latency_ms,
            response_valid=False,
        )
        if on_response_error == "raise":
            raise SchemaDriftError(
                f"Response validation failed for {tool_name}: {errors}"
            )

    def decorator(func: F) -> F:
        def _validate_incoming(*args: Any, **kwargs: Any) -> dict[str, Any]:
            sanitized_kwargs = dict(kwargs)
            correction_attempt, correction_history = _extract_correction_context(
                sanitized_kwargs
            )
            if params_schema is None:
                return sanitized_kwargs
            try:
                bound_args = bind_arguments(func, *args, **sanitized_kwargs)
            except TypeError as exc:
                _handle_param_failure(
                    tool_name=func.__name__,
                    errors=[str(exc)],
                    attempted_params={"args": list(args), "kwargs": sanitized_kwargs},
                    correction_attempt=correction_attempt,
                    correction_history=correction_history,
                )
                return sanitized_kwargs
            result = validate_params(bound_args, params_schema)
            if not result.valid:
                _handle_param_failure(
                    tool_name=func.__name__,
                    errors=result.errors,
                    attempted_params=bound_args,
                    correction_attempt=correction_attempt,
                    correction_history=correction_history,
                )
            return sanitized_kwargs

        def _validate_outgoing(result: Any, latency_ms: float) -> None:
            if response_schema is None:
                return
            validation_result = validate_response(result, response_schema)
            if not validation_result.valid:
                _handle_response_failure(func.__name__, validation_result.errors, latency_ms)

        def _log_pass(latency_ms: float) -> None:
            log_trace(
                func.__name__,
                "PASS",
                latency_ms=latency_ms,
                params_valid=True if params_schema else None,
                response_valid=True if response_schema else None,
            )

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                sanitized_kwargs = _validate_incoming(*args, **kwargs)
                start = time.perf_counter()
                try:
                    result = await func(*args, **sanitized_kwargs)
                except Exception as exc:
                    log_trace(func.__name__, "EXECUTION_FAIL", errors=[str(exc)])
                    raise

                latency_ms = round((time.perf_counter() - start) * 1000, 3)
                _validate_outgoing(result, latency_ms)
                _log_pass(latency_ms)
                return result

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sanitized_kwargs = _validate_incoming(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **sanitized_kwargs)
            except Exception as exc:
                log_trace(func.__name__, "EXECUTION_FAIL", errors=[str(exc)])
                raise

            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            _validate_outgoing(result, latency_ms)
            _log_pass(latency_ms)
            return result

        return cast(F, wrapper)

    return decorator
