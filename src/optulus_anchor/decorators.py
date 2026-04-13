from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def validate_tool(
    *, params_schema: type[Any], response_schema: type[Any]
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(
                "validate_tool call: function=%s params_schema=%r response_schema=%r",
                func.__name__,
                params_schema,
                response_schema,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
