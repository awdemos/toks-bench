#!/bin/bash
# Launch vLLM serving Qwen3-4B-Instruct-2507 via Docker and benchmark it.
set -euo pipefail

WORKDIR="/home/andrewh/spark-vllm-docker/toks-bench"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/vllm-qwen3-4b-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=900
PORT=8004
CONTAINER_NAME="vllm-qwen3-4b-8004"
VLLM_IMAGE="vllm-node:latest"
MODEL_DIR="/models/qwen3-4b-instruct"

cd "$WORKDIR"
source .venv/bin/activate
source "$WORKDIR/scripts/manage-default-llama-server.sh"
mkdir -p "$RESULTS"
exec > >(tee -a "$LOG")
exec 2>&1

log() { echo "=== $(date -Iseconds) === $*"; }

health_check() {
  local url=$1
  local wait=${2:-600}
  local deadline=$(($(date +%s) + wait))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -o /dev/null --max-time 3 "$url/v1/models"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_other_gpu() {
  log "Stopping other GPU servers to free the GB10 GPU"
  default_server_stop
  pkill -f "llama-server.*--port 8081" 2>/dev/null || true
  pkill -f "llama-server.*--port 8083" 2>/dev/null || true
  pkill -f "llama-server.*--port 8084" 2>/dev/null || true
  pkill -f "llama-server.*--port 8085" 2>/dev/null || true
  pkill -f "llama-server.*--port 8086" 2>/dev/null || true
  pkill -f "llama-server.*--port 8087" 2>/dev/null || true
  docker stop vllm-nematron-8001 vllm-qwen-8000 "$CONTAINER_NAME" 2>/dev/null || true
  docker rm vllm-nematron-8001 vllm-qwen-8000 "$CONTAINER_NAME" 2>/dev/null || true
  sleep 5
}

run_bench() {
  local provider=$1
  local prompt=$2
  local out="$RESULTS/${provider}-${prompt}.json"
  log "Benchmarking $provider / $prompt -> $out"
  timeout "$TIMEOUT" toks-bench --provider "$provider" --prompt "$prompt" \
    --runs "$RUNS" --format json --output "$out" || true
}

stop_other_gpu

log "Starting vLLM Qwen3-4B-Instruct-2507 on port $PORT"
docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p "$PORT:$PORT" \
  -v /home/andrewh/models:/models:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL_DIR" \
    --served-model-name qwen3-4b-instruct-2507 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --port "$PORT" --host 0.0.0.0 \
    2>&1 | tee /tmp/vllm-qwen3-4b-serve.log &

if ! health_check "http://localhost:$PORT"; then
  log "vLLM Qwen3-4B failed to start"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  exit 1
fi

for prompt in $PROMPTS; do
  run_bench "vllm-qwen3-4b-instruct-2507" "$prompt"
done

docker stop "$CONTAINER_NAME" 2>/dev/null || true

log "vLLM Qwen3-4B sweep complete. Aggregating..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
toks-bench-profile --csv "$WORKDIR/results/aggregate.csv" --results-dir "$RESULTS" --output "$WORKDIR/results/profiling-report.md"
toks-bench-chart --csv "$WORKDIR/results/aggregate.csv" --output "$WORKDIR/results/charts"
log "Done. Aggregate, profiling report, and charts regenerated."

# Restore the permanent default llama-server on port 8080
log "Restoring permanent default llama-server on port 8080"
default_server_start || true
