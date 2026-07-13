#!/bin/bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
MODEL_DIR="${MODEL_DIR:-/home/andrewh/models/nematron-3-nano-4b-fp8}"
# SECURITY: Do not run a vLLM binary from /tmp (world-writable).  Use a
# properly installed virtual-environment binary instead.
VLLM_BIN="${VLLM_BIN:-$WORKDIR/.venv/bin/vllm}"

cd "$WORKDIR"
source .venv/bin/activate

if [ ! -x "$VLLM_BIN" ]; then
  echo "ERROR: vLLM binary not found or not executable: $VLLM_BIN" >&2
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name nemotron-3-nano-4b-fp8 \
  --port 8001 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --quantization fp8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-nematron-8001.log
