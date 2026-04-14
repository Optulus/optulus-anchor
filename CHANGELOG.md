# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
