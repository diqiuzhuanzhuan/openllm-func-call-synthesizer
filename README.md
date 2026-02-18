# 🛠️ openllm-func-call-synthesizer

![PyPI version](https://img.shields.io/pypi/v/openllm-func-call-synthesizer.svg)
[![Documentation Status](https://readthedocs.org/projects/openllm-func-call-synthesizer/badge/?version=latest)](https://openllm-func-call-synthesizer.readthedocs.io/en/latest/?version=latest)

> Lightweight toolkit to synthesize function-call datasets and convert them to formats compatible with OpenAI-style function-call training and downstream tooling (including Llama Factory compatible exports).

---

## ✨ Features

- 📝 Generate synthetic function call datasets for LLM training and evaluation
- ⚙️ Flexible configuration via YAML and Hydra
- 💻 CLI interface powered by Typer & Rich
- 🔧 Utility functions for dataset manipulation
- 🔄 Extensible and easy to integrate into your own pipeline
- 🌐 Supports multiple LLM backends (OpenAI, Google, etc.)
- 📊 Export formats: JSONL, CSV, Parquet, LlamaFactory-compatible

---

## 🛠 Installation

### Prerequisites

- Python 3.12+ (match environment used by the project)
- API credentials for any LLM backend (set via environment variables or `.env` file)
  - Example: `OPENAI_API_KEY`
  - See `.env.example` for reference

- 🔌 MCP Server (Required)

	This project relies on an MCP server to provide tool/function metadata.

	Before running the synthesizer, you must start an MCP server.

	▶ Start the example MCP server

	An example MCP server is included in the repository:

	python examples/mcp_example_sserver/server.py

	This will start a local MCP server that the synthesizer can connect to.

	Make sure your configuration (e.g. mcp_servers.transport) matches the server address.

	⸻

	⚠ Important
	* The synthesizer will fail if no MCP server is available.
	* Ensure the server is running before executing:

python -m apps.main

	* If you see connection errors, verify:
	* The server is running
	* The transport URL in your config is correct
	* Network/firewall settings allow local connections

⸻
---

### Install from PyPI

```bash
pip install openllm-func-call-synthesizer
# or using uv
uv add openllm-func-call-synthesizer
```

Install from source
```bash
git clone https://github.com/diqiuzhuanzhuan/openllm-func-call-synthesizer.git
cd openllm-func-call-synthesizer
uv sync
```
Is there no tool named 'uv'? You can install it with just one command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

⸻

⚡ Quickstart

Run the synthesizer with default config:
```bash
python -m apps.main
```

Enable only query generation:
```bash
python -m apps.main synthesizer.query_generation.enable=True
```

Enable function-call generation with custom name:
```bash
python -m apps.main synthesizer.function_call_generation.enable=True synthesizer.function_call_generation.name=function_call_gpt_4o
```

Override languages dynamically:

```bash
python -m apps.main synthesizer.query_generation.languages=[English,Spanish]
```

⸻

📂 Outputs
* Generated datasets are written to data/<name>/
* Each run produces:
* train.jsonl
* output.csv
* output.parquet
* llama_factory step creates LlamaFactory-compatible train.jsonl

⸻

🧪 Testing

Run the test suite:
```bash
pytest -q
```


⸻

📝 Configuration Highlights

Configuration file: examples/conf/synthesizer/default.yaml
* mcp_servers — MCP server(s) to query for available tools
* choose_part_tools — filter toolset to a subset
* query_generation — generate seed queries from function docs
* function_call_generation — generate function-call pairs from queries
* critic — optional scoring/critique step
* llama_factory — export to LlamaFactory-compatible dataset
* verl - export to verl-compatible dataset

See docs for full field descriptions.

### Default pipeline walk-through

The provided `examples/conf/synthesizer/default.yaml` wires every stage together:

- **MCP bootstrap**: points to a local `ugreen_mcp` server on `http://localhost:8000/mcp`; leave it running before launching the synth job or queries will fail.
- **Tool filtering**: `choose_part_tools: false` keeps the full toolset; set it to a list (e.g. `["search_photos"]`) to restrict generations to specific tools.
- **Query generation**: reads `examples/function_docs.json`, emits multilingual prompts (English/Chinese/Japanese/German) under `data/function_query` via parallel OpenAI + Google model pools, each with generous TPM throttles for high-throughput runs.
- **Function-call synthesis**: consumes the query dataset, calls `gpt-4o` through the OpenAI backend, and writes `data/function_call_gpt_4o/*.jsonl` (set `max_num` to limit volume or switch `output_format`).
- **Critic pass**: re-scores every call with `gpt-5-mini-2025-08-07`, expecting `query/prompt/function_call/functions/answer` fields and emitting a scored dataset named `function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07`.
- **Downstream exports**: both `llama_factory` and `verl` blocks draw from the critic output, keep only rows with `score >= 8`, and materialize ready-to-train JSONL files plus optional train/val splits.

Feel free to copy the default file, tweak model lists or directories, and pass it via `python -m apps.main synthesizer=@your_config.yaml` for customized runs. For custom configurations, please refer to `example/conf/synthesizer/default.yaml`.
⸻

🐚 Parallel Runner

Helper script: bin/run_pipeline.sh
* Launch multiple synthesizer runs in parallel
* Requires .venv virtual environment
* Example usage:

```bash
chmod +x bin/run_pipeline.sh
bin/run_pipeline.sh default other
```
* Logs are printed to console; returns non-zero if any run fails
* Can also run manually using:

```bash
python -m apps.main synthesizer=default &
python -m apps.main synthesizer=other &
wait
```

⸻

## Contributing

Welcome to contribute！Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://openllm-func-call-synthesizer.readthedocs.io)
- [PyPI](https://pypi.org/project/openllm-func-call-synthesizer/)
- [GitHub](https://github.com/diqiuzhuanzhuan/openllm-func-call-synthesizer)

⸻

🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=diqiuzhuanzhuan/openllm-func-call-synthesizer&type=Date)](https://www.star-history.com/#diqiuzhuanzhuan/openllm-func-call-synthesizer&Date)
