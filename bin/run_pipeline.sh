#!/usr/bin/env bash
set -e

# --- Step 1: 获取脚本所在目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Step 2: 项目根目录为父目录 ---
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Step 3: 进入根目录 ---
cd "$ROOT_DIR"

# --- Step 4: 激活虚拟环境 ---
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "❌ Error: Virtual environment not found at $ROOT_DIR/.venv/"
    exit 1
fi

# --- Step 3: 检查是否传入参数数组 ---
if [ "$#" -eq 0 ]; then
    echo "❌ Error: You must pass an array of synthesizer values (e.g. ./run_parallel.sh kk yy xx)"
    exit 1
fi

# --- Step 4: 并行执行 ---
echo "🚀 Running tasks in parallel..."

PIDS=()

for item in "$@"; do
    echo "▶️ Launching: python apps/main.py synthesizer=$item"
    python apps/main.py "synthesizer=$item" &

    PIDS+=($!)
done

# --- Step 5: 等待所有任务 ---
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "🎉 All tasks finished!"
