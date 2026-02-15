# openllm-func-call-synthesizer

Lightweight toolkit to synthesize function-call datasets and convert them to formats compatible with OpenAI-style function-call training and downstream tooling (including Llama Factory compatible exports).

This README highlights how to configure and run the synthesizer component using the example configuration at `examples/conf/synthesizer/default.yaml`.

**Quick Overview**
- **Purpose:** generate queries and function-call examples from function documentation, optionally critique them, and export to multiple formats.
- **Main script:** `apps/main.py` (Hydra-configured entrypoint).
- **Primary outputs:** JSONL/CSV/Parquet under the `data/` folder, organized by job `name`.

**Prerequisites**
- Python 3.12+ (match environment used by the project). See `pyproject.toml` for pinned dependencies.
- API credentials for any LLM backend you plan to use (OpenAI, Google, etc.). Put them in environment variables or a `.env` file. The app uses `python-dotenv` to load `.env` automatically.

Common env vars:
- `OPENAI_API_KEY` — OpenAI-compatible backends.
- (other providers) configure according to your backend provider's SDK. Refering to .env.example is a good choice.

Installation

```bash
python -m pip install -e .
# or using poetry/pip-tools consistent with the project
```

Configuration
-----------
The canonical synthesizer configuration lives under `examples/conf/synthesizer/default.yaml`. Important sections:

- `mcp_servers` — MCP server(s) to query for available tools. Example key: `ugreen_mcp` with `transport` URL.
- `choose_part_tools` — if true, filters toolset to only a chosen subset.
- `query_generation` — controls generation of seed queries. Key fields:
  - `enable` — enable query generation (True/False)
  - `function_docs` — path to the function docs JSON used as seeds (e.g. `examples/function_docs.json`)
  - `languages` — list of languages to synthesize (English, Chinese, Japanese, German, ...)
  - `name` and `output_dir` — determine where generated outputs are written (e.g. `data/function_query`).
  - `providers` — backends and models to use (OpenAI, Google, etc.) with `backend_params`.

- `function_call_generation` — generate function-call pairs using a model and the previously generated queries. Key fields:
  - `enable`, `function_dataset` (path to query dataset), `max_num`, `provider` (model/backend/generation params), `output_dir`, `name`.

- `critic` — optional critic pass which scores or critiques generated function calls. Key fields mirror the function call dataset layout (`query_field`, `task_prompt_field`, `label_field`, `response_field`) and include provider settings.

- `llama_factory` — convenience exporter that converts the critic dataset into a LlamaFactory-compatible format. Controls `score_field`, `score_threshold`, `split_ratio`, and `system_prompt`.

Usage
-----

Run the synthesizer entrypoint. The script uses Hydra for config management; override values via the CLI.

Basic run (uses the config files found under `examples/conf`):

```bash
python -m apps.main
```

Enable only query generation from the command-line:

```bash
python -m apps.main synthesizer.query_generation.enable=True
```

Enable function-call generation and set a custom output name:

```bash
python -m apps.main synthesizer.function_call_generation.enable=True synthesizer.function_call_generation.name=function_call_gpt_4o
```

Notes on overrides:
- Hydras CLI override syntax mirrors the config keys. For example, to change languages on the fly:

```bash
python -m apps.main synthesizer.query_generation.languages=[English,Spanish]
```

Outputs
-------
- Generated datasets are written to `data/<name>/` (the `output_dir` combined with `name`). Each run produces `train.jsonl`, `output.csv`, and `output.parquet` where applicable.
- The `llama_factory` step writes a Llama Factory compatible `train.jsonl` within its configured `output_dir`.

Tips & Troubleshooting
----------------------
- If you get connection errors while fetching tools, verify `mcp_servers` transport URL and network access.
- Confirm API keys are available in the environment. The code loads `.env` at startup.
- If you produce many provider/model/language combinations, generation may take long — consider limiting `providers` or `models` in the config, or running specific steps selectively using Hydra overrides.

Developing & Testing
---------------------
- Run the test suite:

```bash
pytest -q
```

- To iterate quickly, run only the portion you need (e.g. enable query generation only).

Next steps / Suggestions
------------------------
- Add a small example `examples/function_docs.json` snapshot (already present) and a minimal `.env.example` listing required env vars.
- Add a `docs/` quickstart that includes pre-built example outputs and sample commands for common workflows (query → function_call → critic → llama_factory).

If you'd like, I can:
- generate a minimal `.env.example` file for the repo
- add CLI examples for common Hydra overrides
- or open a PR with a short `examples/quickstart.sh` script that runs a complete pipeline using the default config

---
Generated from `examples/conf/synthesizer/default.yaml` to highlight relevant fields and example runs.

Using the bundled parallel runner
-------------------------------

This repository includes a helper script `bin/run_pipeline.sh` that launches multiple synthesizer runs in parallel using background processes. The script expects a Python virtual environment at `.venv/` in the project root and accepts one or more synthesizer config names (the Hydra config group key) as positional arguments.

Behavior summary:
- Ensures the script runs from the project root and activates `.venv/bin/activate`.
- Requires at least one positional argument; each argument is passed to the main app as `synthesizer=<arg>`.
- Launches each `python -m apps.main` invocation in the background and waits for all to finish.

Example usage:

```bash
# make script executable (only needed once)
chmod +x bin/run_pipeline.sh

# run two synthesizer jobs in parallel (use synthesizer keys defined in your configs)
# e.g. default.yaml and other.yaml under examples/conf/synthesizer/
bin/run_pipeline.sh default other
```

Notes & Troubleshooting:
- Ensure you have a virtual environment at `.venv` with the project dependencies installed. The script will exit if `.venv/bin/activate` is missing.
- If you prefer to use a system or conda environment, either create the `.venv` stub or run the same commands manually from your preferred environment:

```bash
python -m apps.main synthesizer=default &
python -m apps.main synthesizer=other &
wait
```

- The script returns non-zero if the venv is missing or if any background process fails; logs are printed to the console so you can inspect failures.

If you want, I can also:
- make the script more robust by adding per-job logging files, timeouts, or retry logic
- add a small `examples/quickstart.sh` that creates `.venv`, installs dependencies, and runs the pipeline using the default config

# openllm-func-call-synthesizer

![PyPI version](https://img.shields.io/pypi/v/openllm-func-call-synthesizer.svg)
[![Documentation Status](https://readthedocs.org/projects/openllm-func-call-synthesizer/badge/?version=latest)](https://openllm-func-call-synthesizer.readthedocs.io/en/latest/?version=latest)

A tool for generating synthetic function call datasets for Large Language Models (LLMs).

* PyPI package: https://pypi.org/project/openllm-func-call-synthesizer/
* Free software: MIT License
* Documentation: https://openllm-func-call-synthesizer.readthedocs.io.

## Features

- Generate synthetic function call datasets for LLM training and evaluation
- Flexible configuration via YAML and Hydra
- CLI interface powered by Typer & Rich
- Utility functions for dataset manipulation
- Extensible and easy to integrate into your own pipeline

## Installation
### prerequisition

```sh
# install uv (a modern python package management tool)
# just skip it if installed.
curl -LsSf https://astral.sh/uv/install.sh | sh
```


You can install the stable release from PyPI:

```sh
uv add openllm-func-call-synthesizer
# or
pip install openllm-func-call-synthesizer
```

Install from the source code
```sh
git clone https://github.com/diqiuzhuanzhuan/openllm-func-call-synthesizer.git
cd openllm-func-call-synthesizer
uv sync
```

For more installation options, see [docs/installation.md](docs/installation.md).

## Usage

Basic usage:

```python
import openllm_func_call_synthesizer
# Please refer to specific documentation for CLI useage
```

CLI Usage：

```sh
openllm_func_call_synthesizer --help
```

more details  [docs/usage.md](docs/usage.md)。

## Quickstart

```sh
openllm_func_call_synthesizer
# or
python -m openllm_func_call_synthesizer
```

## Testing

Run tests with pytest:

```sh
pytest
```

## Contributing

Welcome to contribute！Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://openllm-func-call-synthesizer.readthedocs.io)
- [PyPI](https://pypi.org/project/openllm-func-call-synthesizer/)
- [GitHub](https://github.com/diqiuzhuanzhuan/openllm-func-call-synthesizer)

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=diqiuzhuanzhuan/openllm-func-call-synthesizer&type=Date)](https://www.star-history.com/#diqiuzhuanzhuan/openllm-func-call-synthesizer&Date)
