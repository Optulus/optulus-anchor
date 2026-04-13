from __future__ import annotations


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
