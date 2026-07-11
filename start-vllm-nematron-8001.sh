#!/bin/bash
set -e
cd /home/andrewh/spark-vllm-docker/toks-bench
source .venv/bin/activate
exec /tmp/vllm-bench/bin/vllm serve /home/andrewh/models/nemotron-3-nano-4b-fp8 \
  --served-model-name nemotron-3-nano-4b-fp8 \
  --port 8001 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --quantization fp8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-nematron-8001.log
