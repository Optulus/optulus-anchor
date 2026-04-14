# optulus-anchor

Python runtime guardrails for AI tool functions:

- validate tool inputs against a schema before execution
- validate tool outputs against a schema after execution
- emit structured trace events for observability and drift detection

## Install

```bash
pip install optulus-anchor
```

## 30-Second Example

```python
from pydantic import BaseModel
from optulus_anchor import validate_tool


class SearchParams(BaseModel):
    query: str
    limit: int = 3


class SearchResponse(BaseModel):
    results: list[str]
    count: int


@validate_tool(
    params_schema=SearchParams,
    response_schema=SearchResponse,
    on_param_error="raise",
    on_response_error="log",
)
def search_docs(query: str, limit: int = 3) -> dict[str, object]:
    selected = [f"{query}-a", f"{query}-b", f"{query}-c"][:limit]
    return {"results": selected, "count": len(selected)}
```

## Public API

```python
from optulus_anchor import (
    SchemaDriftError,
    ToolCorrectionNeeded,
    ToolValidationError,
    disable_persistent_tracelog,
    enable_persistent_tracelog,
    set_trace_sink,
    validate_tool,
)
```

### `validate_tool`

```python
validate_tool(
    *,
    params_schema: type[Any] | None = None,
    response_schema: type[Any] | None = None,
    on_param_error: Literal["raise", "log", "warn", "self_correct"] = "raise",
    on_response_error: Literal["raise", "log", "warn"] = "log",
    max_correction_attempts: int = 2,
) -> Callable[[F], F]
```

#### Parameter validation (`params_schema`)

- argument binding uses the wrapped function signature
- defaults are applied
- `self` / `cls` are excluded from validation payloads

`on_param_error` behavior:

- `"raise"`: emit `PARAM_FAIL`, raise `ToolValidationError`, do not execute the tool
- `"log"` / `"warn"`: emit `PARAM_FAIL`, continue execution
- `"self_correct"`: emit `PARAM_FAIL`, raise `ToolCorrectionNeeded` with:
  - tool name, attempt index, max attempts
  - attempted params
  - normalized validation errors
  - a generated correction prompt for LLM retry loops
  - correction history

#### Response validation (`response_schema`)

`on_response_error` behavior:

- `"raise"`: emit `RESPONSE_FAIL`, raise `SchemaDriftError`
- `"log"` / `"warn"`: emit `RESPONSE_FAIL`, return original tool output

#### Execution behavior

- works with sync and async functions
- on runtime exceptions: emits `EXECUTION_FAIL`, then re-raises
- always emits a `PASS` trace after successful function execution unless an exception is raised
  earlier; with non-raising response policy this means both `RESPONSE_FAIL` and `PASS` can be
  emitted for the same call

### `ToolValidationError`

- strict parameter validation failure (`on_param_error="raise"`)

### `SchemaDriftError`

- subclass of `ToolValidationError`
- strict response validation failure (`on_response_error="raise"`)

### `ToolCorrectionNeeded`

- subclass of `ToolValidationError`
- raised when `on_param_error="self_correct"`
- call `.to_dict()` to get a JSON-serializable payload for orchestrator retries

### `set_trace_sink`

```python
set_trace_sink(sink: Callable[[dict[str, Any]], None] | None) -> None
```

Yes, trace sink callback delivery is supported.

- pass a callable to receive every emitted trace event
- pass `None` to clear callback delivery
- callback is process-global

### Persistent tracelog controls

`optulus-anchor` persists traces to SQLite by default.

- default path: `.trace/traces.sqlite` in current working directory
- override root with `OPTULUS_ANCHOR_TRACE_DIR`
- disable with env var `OPTULUS_ANCHOR_NO_TRACE=1` (or `true/yes/on`)
- disable/enable in-process with:
  - `disable_persistent_tracelog()`
  - `enable_persistent_tracelog()`

## Trace event shape

```json
{
  "timestamp": "ISO-8601 UTC string",
  "tool": "tool_function_name",
  "status": "PASS | PARAM_FAIL | RESPONSE_FAIL | EXECUTION_FAIL",
  "latency_ms": 12.345,
  "params_valid": true,
  "response_valid": true,
  "errors": []
}
```

## CLI

The package installs `anchor`:

```bash
anchor report --hours 24
```

`anchor report` reads the SQLite trace DB and prints:

- tool-level calls and failures in a lookback window
- drift hints inferred from `RESPONSE_FAIL` errors (for example missing fields)
- most unreliable tool by failure rate

## Works With

This SDK wraps regular Python callables, so it can sit under most tool ecosystems:
LangChain, OpenAI tool calling, Anthropic tool use, MCP servers, CrewAI, or custom runtimes.

## LLM Discoverability Files

- `llm.txt` (quick pointer doc)
- `llms.txt` (short context for coding agents)
- `llms-full.txt` (full machine-readable SDK reference)
