#!/bin/bash
# Launch Triton Inference Server with vLLM backend and OpenAI-compatible frontend
# for the local Qwen3-4B-Instruct-2507 checkpoint, then benchmark it.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/triton-vllm-qwen3-4b-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=900
PORT=9000
CONTAINER_NAME="triton-vllm-qwen3-4b"
# SECURITY: Pin this image to a digest to avoid mutable-tag supply-chain attacks.
# Example: TRITON_IMAGE="nvcr.io/nvidia/tritonserver@sha256:..."
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3}"
MODEL_REPO="$WORKDIR/triton_model_repository"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/andrewh/models/qwen3-4b-instruct}"
HOST_MODELS_ROOT="${HOST_MODELS_ROOT:-/home/andrewh/models}"

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
  pkill -f "llama-server.*--port 808[3-7]" 2>/dev/null || true
  docker stop vllm-nematron-8001 vllm-qwen-8000 vllm-qwen3-4b-8004 "$CONTAINER_NAME" 2>/dev/null || true
  docker rm vllm-nematron-8001 vllm-qwen-8000 vllm-qwen3-4b-8004 "$CONTAINER_NAME" 2>/dev/null || true
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

log "Pulling/launching Triton+vLLM container for Qwen3-4B on port $PORT"
docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  -p "$PORT:$PORT" \
  -v "$HOST_MODELS_ROOT:/models:ro" \
  -v "$HOST_MODEL_DIR:/models/qwen3-4b-instruct:ro" \
  -v "$MODEL_REPO:/triton_model_repository:ro" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  "$TRITON_IMAGE" \
  bash -c "cd /opt/tritonserver/python/openai && \
    python3 openai_frontend/main.py \
      --model-repository /triton_model_repository \
      --tokenizer /models/qwen3-4b-instruct \
      --chat-template /triton_model_repository/qwen3-4b-instruct-2507/qwen3-chat-template.jinja \
      --openai-port $PORT \
      --host 0.0.0.0" \
  2>&1 | tee /tmp/triton-vllm-qwen3-4b-serve.log &

if ! health_check "http://localhost:$PORT"; then
  log "Triton+vLLM failed to start"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  exit 1
fi

for prompt in $PROMPTS; do
  run_bench "triton-vllm-qwen3-4b-instruct-2507" "$prompt"
done

docker stop "$CONTAINER_NAME" 2>/dev/null || true

log "Triton+vLLM Qwen3-4B sweep complete. Regenerating reports..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
toks-bench-profile --csv "$WORKDIR/results/aggregate.csv" --results-dir "$RESULTS" --output "$WORKDIR/results/profiling-report.md"
toks-bench-chart --csv "$WORKDIR/results/aggregate.csv" --output "$WORKDIR/results/charts"
log "Done. Aggregate, profiling report, and charts regenerated."

# Restore the permanent default llama-server on port 8080
log "Restoring permanent default llama-server on port 8080"
default_server_start || true
