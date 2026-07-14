#!/usr/bin/env bash
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
# install system command dependencies
# ===============================
install_command() {
    local command_name="$1"
    local apt_package="$2"
    local rpm_package="$3"
    local apk_package="$4"

    if command -v "$command_name" >/dev/null 2>&1; then
        return 0
    fi

    echo "[INFO] '$command_name' is missing; installing it..."

    local privilege=()
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "[ERROR] Installing '$command_name' requires root privileges or sudo."
            exit 1
        fi
        privilege=(sudo)
    fi

    if command -v apt-get >/dev/null 2>&1; then
        "${privilege[@]}" apt-get update
        "${privilege[@]}" apt-get install -y "$apt_package"
    elif command -v dnf >/dev/null 2>&1; then
        "${privilege[@]}" dnf install -y "$rpm_package"
    elif command -v yum >/dev/null 2>&1; then
        "${privilege[@]}" yum install -y "$rpm_package"
    elif command -v apk >/dev/null 2>&1; then
        "${privilege[@]}" apk add --no-cache "$apk_package"
    else
        echo "[ERROR] No supported package manager found to install '$command_name'."
        exit 1
    fi
}

install_dependencies() {
    install_command curl curl curl curl
    install_command pkill procps procps-ng procps

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] 'uv' is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    echo "[INFO] Syncing Python environment with uv..."
    uv sync
}

# ===============================
# wait for server ready
# ===============================
wait_for_server() {
    echo "[INFO] Waiting for MCP Server to be ready at $SERVER_URL"
    for i in {1..30}; do
        if curl -s -o /dev/null "$SERVER_URL"; then
            echo "[INFO] MCP Server is ready"
            return 0
        fi
        echo "[INFO] Server not ready yet (attempt $i/30)..."
        sleep 1
    done

    echo "[ERROR] MCP Server did not become ready in time"
    exit 1
}

# ===============================
# start and daemonize server
# ===============================
start_server() {
    echo "[INFO] start MCP Server: $SERVER_URL"
    while true; do
        uv run python "$SERVER_SCRIPT" &
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

# install required commands and synchronize the Python environment
install_dependencies

# start server in background
start_server &

# wait until server is reachable before launching main script
wait_for_server

# ===============================
# start main script
# ===============================
uv run python "$MAIN_SCRIPT" \
    synthesizer=test \
    synthesizer.mcp_servers.ugreen_mcp.transport="$SERVER_URL" \
    synthesizer.query_generation.enable=True \
    synthesizer.query_generation.function_docs="examples/function_docs.example.json" \
    synthesizer.function_call_generation.enable=True \
    synthesizer.function_call_generation.function_dataset="data/tool_query" \
    synthesizer.critic.enable=True
