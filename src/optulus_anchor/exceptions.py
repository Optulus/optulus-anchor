from __future__ import annotations

from typing import Any


class ToolValidationError(Exception):
    """
    Exception raised when tool validation fails under a strict error policy.

    This is typically triggered when ``validate_tool(..., on_param_error="raise")``
    detects that incoming arguments do not conform to the declared parameter schema.

    Example:
        ```python
        from optulus_anchor import ToolValidationError

        try:
            raise ToolValidationError("query: Field required")
        except ToolValidationError:
            # Handle invalid tool inputs.
            pass
        ```
    """


class SchemaDriftError(ToolValidationError):
    """
    Exception raised when a tool response no longer matches its declared schema.

    This is typically triggered when
    ``validate_tool(..., on_response_error="raise")`` detects output shape drift,
    such as an upstream API changing response fields unexpectedly.

    Example:
        ```python
        from optulus_anchor import SchemaDriftError

        try:
            raise SchemaDriftError("results: Field required")
        except SchemaDriftError:
            # Handle response schema drift.
            pass
        ```
    """

class ToolCorrectionNeeded(ToolValidationError):
    """
    Exception raised when parameter validation should be handled by an LLM retry loop.

    This exception is emitted by ``validate_tool(..., on_param_error="self_correct")``.
    The payload is intentionally framework-agnostic so agent runtimes can catch this
    exception, append the correction prompt to model context, and retry the tool call
    with corrected arguments.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        attempt: int,
        max_attempts: int,
        attempted_params: dict[str, Any] | None,
        errors: list[str],
        correction_prompt: str,
        correction_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.attempted_params = attempted_params
        self.errors = errors
        self.correction_prompt = correction_prompt
        self.correction_history = correction_history or []
        super().__init__(self.__str__())

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable correction payload."""
        return {
            "tool_name": self.tool_name,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "attempted_params": self.attempted_params,
            "errors": self.errors,
            "correction_prompt": self.correction_prompt,
            "correction_history": self.correction_history,
        }

    def __str__(self) -> str:
        return (
            f"Tool '{self.tool_name}' needs parameter correction "
            f"(attempt {self.attempt}/{self.max_attempts})."
        )
