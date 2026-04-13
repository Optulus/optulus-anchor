# optulus-anchor

`optulus-anchor` is a Python SDK for validating AI tool calls at runtime.
It wraps tool functions with schema checks for both incoming parameters and outgoing responses, then emits structured JSON traces for observability.

## Install

```bash
pip install optulus-anchor
```

## Why use it

- Catch bad tool arguments before execution.
- Detect response shape drift after execution.
- Control failures with per-stage error policies (`raise`, `log`, `warn`).
- Capture trace events for logs, tests, and telemetry pipelines.
- Works with normal and async Python callables.

## Quickstart

```python
from pydantic import BaseModel

from optulus_anchor import validate_tool


class SearchParams(BaseModel):
    query: str
    limit: int = 10


class SearchResponse(BaseModel):
    results: list[str]
    count: int


@validate_tool(
    params_schema=SearchParams,
    response_schema=SearchResponse,
    on_param_error="raise",   # default
    on_response_error="log",  # default
)
def search_docs(query: str, limit: int = 10) -> dict[str, object]:
    return {"results": [f"Result for {query}"], "count": 1}
```

## Validation flow

1. **Before execution**: incoming arguments are bound to the function signature and validated against `params_schema`.
2. **Function execution**: the wrapped callable runs (sync or async).
3. **After execution**: returned value is validated against `response_schema`.
4. **Trace emission**: JSON trace event is emitted for pass/fail status.

## Error policies

- `on_param_error="raise"`: raises `ToolValidationError` and stops execution.
- `on_param_error="log"` / `"warn"`: logs the validation failure and continues.
- `on_response_error="raise"`: raises `SchemaDriftError`.
- `on_response_error="log"` / `"warn"`: logs response drift and returns the tool result.

## Tracing

By default, traces are emitted to logger `optulus_anchor.tool_validator` as JSON.

Example trace shape:

```json
{
  "timestamp": "2026-04-13T00:00:00+00:00",
  "tool": "search_docs",
  "status": "PASS",
  "latency_ms": 12,
  "params_valid": true,
  "response_valid": true,
  "errors": []
}
```

Possible `status` values:

- `PASS`
- `PARAM_FAIL`
- `RESPONSE_FAIL`
- `EXECUTION_FAIL`

You can also register a custom trace callback:

```python
from optulus_anchor import set_trace_sink

events: list[dict[str, object]] = []
set_trace_sink(events.append)
```

## Public API

- `validate_tool(...)`
- `set_trace_sink(...)`
- `ToolValidationError`
- `SchemaDriftError`

For LLM-focused discoverability docs, see `llms.txt` and `llms-full.txt`.
