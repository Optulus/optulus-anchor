# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-15

### Breaking Changes

- None.

### Added

- Added LangGraph integration via `optulus_anchor.integrations.AnchorToolNode`:
  - catches `ToolCorrectionNeeded` and returns correction `ToolMessage` feedback
  - enforces correction attempt budgets from message history
  - routes strict `ToolValidationError` failures as `ToolMessage` guidance
  - invokes `StructuredTool.func` directly so `validate_tool` receives raw LLM args
- Added LangChain integration via `optulus_anchor.integrations.AnchorToolExecutor`:
  - executes `AIMessage.tool_calls` against current message history
  - converts correction and validation exceptions into model-visible `ToolMessage`s
  - preserves correction cycle metadata for retries and observability
- Added correction cycle observability in traces:
  - `correction_cycle_id`
  - `correction_attempt`
- Added correction-aware reporting support in the CLI and trace aggregation paths.
- Added runnable framework examples:
  - `examples/langgraph_optulus_anchor`
  - `examples/langchain_optulus_anchor`
- Added test coverage for LangGraph and LangChain correction-aware tool execution behavior.

### Changed

- Updated `README.md` to match the current SDK surface, including:
  - LangChain retry orchestration guidance with `AnchorToolExecutor`
  - LangGraph usage with `AnchorToolNode`
  - runnable example commands and optional dependency notes
- Improved integration internals by extracting shared tool-runtime logic used by both
  LangGraph and LangChain adapters.
- Expanded trace/report behavior to include correction-cycle context in production diagnostics.

## [0.2.0]

### Breaking Changes

- None.

### Added

- Added this `CHANGELOG.md` file.
- Added documentation coverage for trace sink callback support via `set_trace_sink(...)`.
- Added documentation for self-correction flow:
  - `on_param_error="self_correct"`
  - `max_correction_attempts`
  - `ToolCorrectionNeeded` payload behavior
- Added documentation for persistent tracelog controls:
  - `enable_persistent_tracelog()`
  - `disable_persistent_tracelog()`
  - `OPTULUS_ANCHOR_TRACE_DIR`
  - `OPTULUS_ANCHOR_NO_TRACE`

### Changed

- Updated `README.md` to match implemented SDK behavior and exported API surface.
- Updated `llm.txt`, `llms.txt`, and `llms-full.txt` for consistency with source code and tests.
- Clarified trace emission behavior when response validation is non-raising (`RESPONSE_FAIL` and `PASS` can both be emitted for one call).

## [0.1.0] - 2026-04-14

### Added

- Initial release of `optulus-anchor`.
- `validate_tool` decorator for parameter and response schema validation.
- Exception types: `ToolValidationError`, `SchemaDriftError`, `ToolCorrectionNeeded`.
- Trace logging with statuses: `PASS`, `PARAM_FAIL`, `RESPONSE_FAIL`, `EXECUTION_FAIL`.
- `set_trace_sink(...)` callback hook for trace event delivery.
- Persistent SQLite tracelog support under `.trace/traces.sqlite`.
- CLI `anchor report` for tool health and drift hint reporting.
