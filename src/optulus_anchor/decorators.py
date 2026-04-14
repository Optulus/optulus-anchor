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
        on_response_error: Policy for response validation failures.
            - ``"raise"``: raise ``SchemaDriftError``.
            - ``"log"``: record failure and continue.
            - ``"warn"``: record failure and continue.

    Returns:
        A decorator that preserves the wrapped function signature and returns a
        wrapped callable with validation and tracing behavior.

    Example:
        ```python
        from pydantic import BaseModel
        from optulus_anchor import validate_tool

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
        ```
    """
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
                _validate_incoming(*args, **kwargs)
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
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
            _validate_incoming(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                log_trace(func.__name__, "EXECUTION_FAIL", errors=[str(exc)])
                raise

            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            _validate_outgoing(result, latency_ms)
            _log_pass(latency_ms)
            return result

        return cast(F, wrapper)

    return decorator
