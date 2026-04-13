from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def bind_arguments(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Bind positional/keyword args to parameter names for validation."""
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    normalized = dict(bound.arguments)
    normalized.pop("self", None)
    normalized.pop("cls", None)
    return normalized
