![Build](https://img.shields.io/badge/build-passing-brightgreen) ![PyPI](https://img.shields.io/pypi/v/optulus-anchor) ![Python](https://img.shields.io/pypi/pyversions/optulus-anchor) ![License](https://img.shields.io/github/license/Optulus/optulus-anchor) ![Downloads](https://img.shields.io/pypi/dm/optulus-anchor)
# optulus-anchor

Python runtime guardrails for AI tool functions and LLM tool-calling systems.

Drop-in runtime validation for OpenAI, LangChain, MCP, and custom AI tools.
Validate inputs, validate outputs, and detect schema drift in production.

Optulus Anchor is a Python decorator for validating AI tool calls, OpenAI function calls,
LangChain tools, Anthropic tool use, MCP tools, and custom agent runtimes.

LLM tool calls often break silently:

- wrong parameter types
- missing required fields
- schema drift after model updates
- invalid JSON outputs
- no observability in production

Optulus Anchor catches these failures at runtime with structured trace events.

## Install

```bash
pip install optulus-anchor
```

## Why

Model upgrades, prompt changes, and orchestration bugs can break tool calls without obvious failures.
Use Anchor in CI, staging, or production to monitor tool reliability over time.

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

## Before and After

```python
# Without Anchor
search_docs(limit="five")  # often fails later in unclear ways

# With Anchor
# Emits PARAM_FAIL with normalized validation errors before execution
search_docs(limit="five")
```

## Works With

This SDK wraps regular Python callables, so it can sit under most tool ecosystems:
LangChain, OpenAI tool calling, Anthropic tool use, MCP servers, CrewAI, or custom runtimes.

## Use Cases

- OpenAI function calling retries with structured validation errors
- LangChain tool input/output validation
- MCP server schema enforcement
- production drift detection after model or prompt changes
- agent tool observability with trace events and reporting

## LangChain self-correction loop helper

`@validate_tool(..., on_param_error="self_correct")` raises `ToolCorrectionNeeded`.
The decorator does not orchestrate retries by itself. For LangChain message loops,
use `AnchorToolExecutor` to convert correction exceptions into `ToolMessage`
feedback, enforce correction budgets from message history, and preserve
`correction_cycle_id` trace context.

```python
from langchain_core.messages import HumanMessage
from optulus_anchor.integrations import AnchorToolExecutor

messages = [HumanMessage("run")]
executor = AnchorToolExecutor(tools)

while True:
    ai = llm.invoke(messages)
    messages.append(ai)
    tool_messages = executor.execute(messages=messages, ai_message=ai)
    if not tool_messages:
        break
    messages.extend(tool_messages)
```

For LangGraph graphs, use `AnchorToolNode` in your tools node.

## Common Failures It Catches

- missing required argument
- wrong enum value or type mismatch
- malformed tool response payload
- response schema drift in production

## Trace Event Example

```json
{
  "tool": "search_docs",
  "status": "PARAM_FAIL",
  "errors": ["limit: Input should be int"]
}
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

## LLM Discoverability Files

- `llm.txt` (quick pointer doc)
- `llms.txt` (short context for coding agents)
- `llms-full.txt` (full machine-readable SDK reference)
