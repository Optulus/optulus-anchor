# optulus-anchor

> Python decorator that validates AI agent tool calls against Pydantic schemas and detects external API drift.

## Install

```bash
pip install optulus-anchor
```

## CLI

`optulus-anchor` includes an `anchor` command with a `report` subcommand for trace health:

```bash
anchor report
```

By default, it reads from `.trace/traces.sqlite` in your current working directory.
If you set `OPTULUS_ANCHOR_TRACE_DIR`, it reads from
`$OPTULUS_ANCHOR_TRACE_DIR/.trace/traces.sqlite` instead.

The report summarizes each tool's call volume/failures for the last 24 hours and
highlights response-schema drift signals detected in trace validation errors.

## 30-Second Example

```python
import logging

from pydantic import BaseModel

from optulus_anchor import validate_tool


logging.basicConfig(level=logging.INFO)


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
    all_hits = [f"{query}-a", f"{query}-b", f"{query}-c", f"{query}-d"]
    selected = all_hits[:limit]
    return {"results": selected, "count": len(selected)}


if __name__ == "__main__":
    print(search_docs(query="anchor sdk", limit=2))
```

## Works With

This library wraps regular Python functions, so it can sit behind common tool ecosystems.

### LangChain (`@tool`)

```python
from langchain_core.tools import tool
from pydantic import BaseModel
from optulus_anchor import validate_tool


class WeatherParams(BaseModel):
    city: str


class WeatherResponse(BaseModel):
    forecast: str


@tool
@validate_tool(params_schema=WeatherParams, response_schema=WeatherResponse)
def weather_tool(city: str) -> dict[str, str]:
    return {"forecast": f"Sunny in {city}"}
```

### OpenAI tool calling

```python
from pydantic import BaseModel
from optulus_anchor import validate_tool


class LookupParams(BaseModel):
    account_id: str


class LookupResponse(BaseModel):
    status: str


@validate_tool(params_schema=LookupParams, response_schema=LookupResponse)
def lookup_account(account_id: str) -> dict[str, str]:
    return {"status": f"active:{account_id}"}
```

### Anthropic tool use

```python
from pydantic import BaseModel
from optulus_anchor import validate_tool


class SearchParams(BaseModel):
    query: str


class SearchResponse(BaseModel):
    answer: str


@validate_tool(params_schema=SearchParams, response_schema=SearchResponse)
def claude_search(query: str) -> dict[str, str]:
    return {"answer": f"Result for {query}"}
```

### MCP tools

```python
from pydantic import BaseModel
from optulus_anchor import validate_tool


class TicketParams(BaseModel):
    ticket_id: str


class TicketResponse(BaseModel):
    title: str


@validate_tool(params_schema=TicketParams, response_schema=TicketResponse)
def get_ticket(ticket_id: str) -> dict[str, str]:
    return {"title": f"Ticket {ticket_id}"}
```

### CrewAI tools

```python
from pydantic import BaseModel
from optulus_anchor import validate_tool


class CalcParams(BaseModel):
    a: int
    b: int


class CalcResponse(BaseModel):
    result: int


@validate_tool(params_schema=CalcParams, response_schema=CalcResponse)
def add_tool(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}
```

## Why This Exists

Agent tool calls fail in two high-cost ways: the model sends malformed arguments (hallucinated fields, wrong types, missing required keys), or the downstream API changes response shape over time. Both failures are common in production and can silently degrade agent behavior if they are not caught at the tool boundary.

`optulus-anchor` adds a lightweight validation boundary around each tool function. It validates inputs before execution, validates outputs after execution, and emits structured trace events so teams can alert, debug, and quantify drift without rewriting their tool stack.

## Full API Reference

### `validate_tool`

```python
validate_tool(
    *,
    params_schema: type[Any] | None = None,
    response_schema: type[Any] | None = None,
    on_param_error: Literal["raise", "log", "warn"] = "raise",
    on_response_error: Literal["raise", "log", "warn"] = "log",
) -> Callable[[F], F]
```

Parameters:

- `params_schema`: schema class used to validate incoming bound arguments before execution.
- `response_schema`: schema class used to validate returned value after execution.
- `on_param_error`: behavior when parameter validation fails.
  - `"raise"`: raise `ToolValidationError` and stop execution.
  - `"log"`: emit `PARAM_FAIL` trace and continue.
  - `"warn"`: emit `PARAM_FAIL` trace and continue.
- `on_response_error`: behavior when response validation fails.
  - `"raise"`: raise `SchemaDriftError`.
  - `"log"`: emit `RESPONSE_FAIL` trace and return result.
  - `"warn"`: emit `RESPONSE_FAIL` trace and return result.

Behavior summary:

- Works for sync and async functions.
- Emits `EXECUTION_FAIL` trace on runtime exceptions, then re-raises.
- Emits `PASS` trace with latency on successful validation path.

### `set_trace_sink`

```python
set_trace_sink(sink: Callable[[dict[str, Any]], None] | None) -> None
```

Parameters:

- `sink`: callback that receives each trace event dictionary.
- pass `None` to clear callback delivery.

Default logging behavior:

- logger name: `optulus_anchor.tool_validator`
- `PASS` logs at INFO
- `PARAM_FAIL`, `RESPONSE_FAIL`, and `EXECUTION_FAIL` log at WARNING

### `ToolValidationError`

- Raised by `validate_tool(..., on_param_error="raise")` when parameter validation fails.

### `SchemaDriftError`

- Subclass of `ToolValidationError`.
- Raised by `validate_tool(..., on_response_error="raise")` when response validation fails.

### Trace event shape

```json
{
  "timestamp": "ISO-8601 UTC string",
  "tool": "tool_function_name",
  "status": "PASS | PARAM_FAIL | RESPONSE_FAIL | EXECUTION_FAIL",
  "latency_ms": 12,
  "params_valid": true,
  "response_valid": true,
  "errors": []
}
```

## LLM Discoverability Files

- `llms.txt` (short context for coding agents)
- `llms-full.txt` (complete machine-readable SDK reference)
