from __future__ import annotations


class ToolValidationError(Exception):
    """Raised when tool input or output fails declared schema validation."""


class SchemaDriftError(ToolValidationError):
    """Raised when a tool response no longer matches the expected schema."""
