#!/bin/bash
# Start TensorRT-LLM trtllm-serve for Qwen3-4B-Instruct-2507.
set -euo pipefail

cd /home/andrewh/spark-vllm-docker/toks-bench
source .venv/bin/activate

MODEL_DIR="/home/andrewh/models/qwen3-4b-instruct"
PORT=8003

echo "=== Stopping other GPU servers to free the GB10 GPU ==="
pkill -f "llama-server.*--port 8080" 2>/dev/null || true
pkill -f "llama-server.*--port 8081" 2>/dev/null || true
pkill -f "llama-server.*--port 808[3-9]" 2>/dev/null || true
docker stop vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
docker rm vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
sleep 5

export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/opt/open-mpi/lib:/home/andrewh/spark-vllm-docker/toks-bench/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
echo "=== LD_LIBRARY_PATH=$LD_LIBRARY_PATH ==="
echo "=== Setting library paths for OpenMPI, torch, and TensorRT ==="
export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/opt/open-mpi/lib:$(python -c 'import torch; print(torch.__file__.replace(\"__init__.py\",\"lib\"))'):$(python -c 'import tensorrt_llm; import pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent / \"libs\" / \"..\" / \"tensorrt_libs\")'):${LD_LIBRARY_PATH}"

echo "=== Starting trtllm-serve on port $PORT ==="
exec trtllm-serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max_seq_len 8192 \
  2>&1 | tee /tmp/trtllm-qwen3-4b-serve.log
