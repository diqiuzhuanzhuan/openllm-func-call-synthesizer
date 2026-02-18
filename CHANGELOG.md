# Changelog

All notable changes to this project will be documented here. The format loosely follows the guidance from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-18
### Changed
- Expanded `README.md` with a full walk-through of `examples/conf/synthesizer/default.yaml`, clarifying how MCP bootstrap, query/fn-call generation, critic scoring, and export stages connect.
- Clarified documentation around downstream exporters (LlamaFactory/Verl) and configuration knobs to reduce confusion for first-time users.

## [0.1.1] - 2026-02-14
### Added
- Initial public release of the synthesizer toolkit for generating OpenAI-style function-call datasets from structured function documentation.
- Hydra-configured pipeline covering query generation, function-call synthesis, optional critic scoring, and downstream exports to JSONL/CSV/Parquet as well as LlamaFactory-ready splits.
- CLI entrypoints via `python -m apps.main` and the installed `openllm-func-call-synthesizer` script, plus sample configs under `examples/conf/` to simplify overrides.
- Built-in support for multiple LLM providers/backends (OpenAI-compatible, Google, etc.) with `.env` loading for API keys and reusable provider blocks.
- Helper scripts (e.g., `bin/run_pipeline.sh`) and documentation/README walkthroughs covering installation, uv/pip workflows, and pytest-based verification.

## [0.1.0] - 2025-08-15
### Added
- First internal builds of the synthesizer published to PyPI as a preview for early adopters.
