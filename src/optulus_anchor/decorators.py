from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

from optulus_anchor.arg_binding import bind_arguments
from optulus_anchor.exceptions import SchemaDriftError, ToolValidationError
from optulus_anchor.logger import log_trace
from optulus_anchor.validator import validate_params, validate_response

F = TypeVar("F", bound=Callable[..., Any])
ErrorPolicy = Literal["raise", "log", "warn"]


def validate_tool(
    *,
    params_schema: type[Any] | None = None,
    response_schema: type[Any] | None = None,
    on_param_error: ErrorPolicy = "raise",
    on_response_error: ErrorPolicy = "log",
) -> Callable[[F], F]:
    if on_param_error not in {"raise", "log", "warn"}:
        raise ValueError("on_param_error must be one of: raise, log, warn")
    if on_response_error not in {"raise", "log", "warn"}:
        raise ValueError("on_response_error must be one of: raise, log, warn")

    def _handle_param_failure(tool_name: str, errors: list[str]) -> None:
        log_trace(
            tool_name,
            "PARAM_FAIL",
            errors=errors,
            params_valid=False,
        )
        if on_param_error == "raise":
            raise ToolValidationError(
                f"Parameter validation failed for {tool_name}: {errors}"
            )

    def _handle_response_failure(
        tool_name: str, errors: list[str], latency_ms: int
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
        def _validate_incoming(*args: Any, **kwargs: Any) -> None:
            if params_schema is None:
                return
            try:
                bound_args = bind_arguments(func, *args, **kwargs)
            except TypeError as exc:
                _handle_param_failure(func.__name__, [str(exc)])
                return
            result = validate_params(bound_args, params_schema)
            if not result.valid:
                _handle_param_failure(func.__name__, result.errors)

        def _validate_outgoing(result: Any, latency_ms: int) -> None:
            if response_schema is None:
                return
            validation_result = validate_response(result, response_schema)
            if not validation_result.valid:
                _handle_response_failure(func.__name__, validation_result.errors, latency_ms)

        def _log_pass(latency_ms: int) -> None:
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
                _validate_incoming(*args, **kwargs)
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    log_trace(func.__name__, "EXECUTION_FAIL", errors=[str(exc)])
                    raise

                latency_ms = round((time.perf_counter() - start) * 1000)
                _validate_outgoing(result, latency_ms)
                _log_pass(latency_ms)
                return result

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _validate_incoming(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                log_trace(func.__name__, "EXECUTION_FAIL", errors=[str(exc)])
                raise

            latency_ms = round((time.perf_counter() - start) * 1000)
            _validate_outgoing(result, latency_ms)
            _log_pass(latency_ms)
            return result

        return cast(F, wrapper)

    return decorator
