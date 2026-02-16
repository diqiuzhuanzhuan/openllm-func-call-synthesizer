#!/bin/bash
set -e

# ===============================
# config
# ===============================
SERVER_SCRIPT="examples/mcp_example_server/server.py"
MAIN_SCRIPT="apps/main.py"

# server start at 8000
SERVER_URL="http://localhost:8000/mcp"
# auto restart delay (seconds)
RESTART_DELAY=2

# ===============================
# start and daemonize server
# ===============================
start_server() {
    echo "[INFO] start MCP Server: $SERVER_URL"
    while true; do
        python $SERVER_SCRIPT &
        SERVER_PID=$!
        echo "[INFO] Server PID=$SERVER_PID"

        # wait for server to exit
        wait $SERVER_PID

        echo "[WARN] Server exited，will restart after $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
    done
}

# ===============================
# cleanup on exit
# ===============================
cleanup() {
    echo "[INFO] script exit, kill server..."
    # kill all child processes
    pkill -P $$ || true
    exit 0
}

trap cleanup EXIT

# start server in background
start_server &

# ===============================
# start main script
# ===============================
python $MAIN_SCRIPT \
    synthesizer=default \
    synthesizer.mcp_servers.ugreen_mcp.transport="$SERVER_URL" \
    synthesizer.query_generation.enable=True \
    synthesizer.query_generation.function_docs="examples/function_docs.example.json" \
    synthesizer.function_call_generation.enable=True \
    synthesizer.function_call_generation.function_dataset="data/function_query" \
    synthesizer.critic.enable=True
