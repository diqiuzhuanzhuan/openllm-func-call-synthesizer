#!/usr/bin/env bash

export HYDRA_FULL_ERROR=1
source .venv/bin/activate

python  apps/main.py \
    synthesizer=default \
    synthesizer.mcp_servers.ugreen_mcp.transport="http://localhost:8000/mcp" \
    synthesizer.query_generation.enable=True \
    synthesizer.query_generation.function_docs="examples/function_docs.example.json" \
    synthesizer.function_call_generation.enable=True \
    synthesizer.function_call_generation.function_dataset="data/function_query" \
    synthesizer.critic=True


exit 0
