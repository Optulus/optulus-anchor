from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def bind_arguments(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Normalize a function call into a name-to-value argument mapping.

    This helper binds positional and keyword arguments using the target function
    signature, applies default values, and removes conventional instance/class
    receiver parameters (``self`` and ``cls``).

    Args:
        func: Callable whose signature should be used for argument binding.
        *args: Positional arguments passed to ``func``.
        **kwargs: Keyword arguments passed to ``func``.

    Returns:
        Dictionary mapping parameter names to resolved values.

    Example:
        ```python
        def search(query: str, limit: int = 10) -> None:
            pass

        bound = bind_arguments(search, "llm tools")
        # {"query": "llm tools", "limit": 10}
        ```
    """
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    normalized = dict(bound.arguments)
    normalized.pop("self", None)
    normalized.pop("cls", None)
    return normalized
