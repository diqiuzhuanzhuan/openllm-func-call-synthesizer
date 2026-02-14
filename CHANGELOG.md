# Changelog

All notable changes to this project will be documented here. The format loosely follows the guidance from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-14
### Added
- Initial public release of the synthesizer toolkit for generating OpenAI-style function-call datasets from structured function documentation.
- Hydra-configured pipeline covering query generation, function-call synthesis, optional critic scoring, and downstream exports to JSONL/CSV/Parquet as well as LlamaFactory-ready splits.
- CLI entrypoints via `python -m apps.main` and the installed `openllm-func-call-synthesizer` script, plus sample configs under `examples/conf/` to simplify overrides.
- Built-in support for multiple LLM providers/backends (OpenAI-compatible, Google, etc.) with `.env` loading for API keys and reusable provider blocks.
- Helper scripts (e.g., `bin/run_pipeline.sh`) and documentation/README walkthroughs covering installation, uv/pip workflows, and pytest-based verification.
