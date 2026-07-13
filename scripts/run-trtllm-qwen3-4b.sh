#!/bin/bash
# Start TensorRT-LLM trtllm-serve for Qwen3-4B-Instruct-2507.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
MODEL_DIR="${MODEL_DIR:-/home/andrewh/models/qwen3-4b-instruct}"
PORT=8003
PIDFILE="/tmp/toks-bench-trtllm-qwen3-4b.pid"

cd "$WORKDIR"
source .venv/bin/activate

echo "=== Stopping other GPU servers to free the GB10 GPU ==="
if [ -f "$PIDFILE" ]; then
  old_pid=$(cat "$PIDFILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "=== Stopping tracked trtllm-serve (PID $old_pid) ==="
    kill "$old_pid" 2>/dev/null || true
    sleep 5
  fi
  rm -f "$PIDFILE"
fi
source "$WORKDIR/scripts/manage-default-llama-server.sh"
default_server_stop
docker stop vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
docker rm vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
sleep 5

export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/opt/open-mpi/lib:$WORKDIR/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
echo "=== LD_LIBRARY_PATH=$LD_LIBRARY_PATH ==="
echo "=== Setting library paths for OpenMPI, torch, and TensorRT ==="
export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/opt/open-mpi/lib:$(python -c 'import torch; print(torch.__file__.replace("__init__.py","lib"))'):$(python -c 'import tensorrt_llm; import pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent / "libs" / ".." / "tensorrt_libs")'):${LD_LIBRARY_PATH}"

echo "=== Starting trtllm-serve on port $PORT ==="
# SECURITY: Bind the native server to loopback so it is not exposed to the
# local network. Port forwarding is not used for this native process.
trtllm-serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max_seq_len 8192 \
  2>&1 | tee /tmp/trtllm-qwen3-4b-serve.log &
echo $! > "$PIDFILE"
