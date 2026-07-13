#!/bin/bash
# Sequential GPU benchmark sweep.
# Runs one inference server at a time to avoid overcommitting the GB10 GPU.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/sequential-gpu-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=900
# SECURITY: Pin this image to a digest to avoid mutable-tag supply-chain attacks.
# Example: VLLM_IMAGE="vllm-node@sha256:..."
VLLM_IMAGE="${VLLM_IMAGE:-vllm-node:latest}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/andrewh/llama.cpp/build/bin/llama-server}"
DEFAULT_MODEL="${DEFAULT_MODEL:-/home/andrewh/models/lowbit-2026/Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf}"

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

# Wait until the model is actually ready to serve chat completions.
# llama-server returns 200 from /v1/models while still loading the model,
# so we probe /v1/chat/completions with a tiny request.
model_ready_check() {
  local url=$1
  local wait=${2:-600}
  local deadline=$(($(date +%s) + wait))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
       -H 'Content-Type: application/json' \
       -d '{"model":"dummy","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0.0}' \
       "$url/v1/chat/completions" | grep -qE '^2'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_vllm() {
  docker stop vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
  docker rm vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
  sleep 3
}

stop_llama() {
  default_server_stop
  pkill -f "llama-server.*--port 8081" 2>/dev/null || true
  sleep 3
}

run_vllm_container_named() {
  local name=$1
  local port=$2
  shift 2
  docker run --rm \
    --name "$name" \
    --gpus all \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    -p "$port:$port" \
    -v /home/andrewh/models:/models:ro \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$VLLM_IMAGE" \
    vllm serve "$@" \
      --port "$port" --host 0.0.0.0
}

run_bench() {
  local provider=$1
  local prompt=$2
  local out="$RESULTS/${provider}-${prompt}.json"
  log "Benchmarking $provider / $prompt -> $out"
  timeout "$TIMEOUT" toks-bench --provider "$provider" --prompt "$prompt" \
    --runs "$RUNS" --format json --output "$out" || true
}

run_provider_sweep() {
  local provider=$1
  for prompt in $PROMPTS; do
    run_bench "$provider" "$prompt"
  done
}

START_AT="${START_AT:-1}"

if [ "$START_AT" -le 1 ]; then
  # 1) llama-server-qwen (permanent default: 2.17-bit 35B MoE on 8080)
  log "Starting temporary llama-server-qwen on port 8080 with default model"
  default_server_stop
  "$LLAMA_SERVER" \
    -m "$DEFAULT_MODEL" \
    --host 127.0.0.1 --port 8080 -ngl 99 -c 131072 \
    --cont-batching --flash-attn on --temp 0.7 --jinja \
    2>&1 | tee /tmp/llama-server-qwen-8080.log &
  health_check "http://localhost:8080" || { log "llama-server-qwen not healthy"; exit 1; }
  model_ready_check "http://localhost:8080" || { log "llama-server-qwen model not ready"; exit 1; }
  run_provider_sweep "llama-server-qwen"
fi

# 2) Stop llama-server, start vllm-nematron on 8001
stop_llama
log "Starting vllm-nematron-fp8 on port 8001"
run_vllm_container_named vllm-nematron-8001 8001 \
  /models/nemotron-3-nano-4b-fp8 \
  --served-model-name nemotron-3-nano-4b-fp8 \
  --trust-remote-code --quantization fp8 --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.6 --enforce-eager \
  2>&1 | tee /tmp/vllm-nematron-8001.log &
health_check "http://localhost:8001" || { log "vllm-nematron failed to start"; stop_vllm; exit 1; }
run_provider_sweep "vllm-nematron-fp8-docker"
stop_vllm

# 3) Start vllm-qwen-fp8 on port 8000
log "Starting vllm-qwen-fp8 on port 8000"
run_vllm_container_named vllm-qwen-8000 8000 \
  /models/qwen3.6-27b-fp8 \
  --served-model-name qwen3.6-27b-fp8 \
  --trust-remote-code --language-model-only \
  --quantization fp8 --dtype float16 \
  --max-model-len 8192 --gpu-memory-utilization 0.6 \
  2>&1 | tee /tmp/vllm-qwen-8000.log &
health_check "http://localhost:8000" || { log "vllm-qwen failed to start"; stop_vllm; exit 1; }
run_provider_sweep "vllm-qwen-fp8"
stop_vllm

# 4) Start llama-server-ornith on port 8081
log "Starting llama-server-ornith on port 8081"
$LLAMA_SERVER \
  -m /home/andrewh/models/ornith-1.0-35b/ornith-1.0-35b-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081 -ngl 99 -c 8192 \
  --cont-batching --flash-attn on --temp 0.7 \
  2>&1 | tee /tmp/llama-server-ornith-8081.log &
  health_check "http://localhost:8081" || { log "llama-server-ornith failed to start"; stop_llama; exit 1; }
  model_ready_check "http://localhost:8081" || { log "llama-server-ornith model not ready"; stop_llama; exit 1; }
  run_provider_sweep "llama-server-ornith"
stop_llama

# 5) Restore the permanent default llama-server on port 8080
log "Restoring permanent default llama-server on port 8080"
default_server_start || true

log "Sequential GPU sweep complete. Aggregating..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
log "Done. Aggregate written to $WORKDIR/results/aggregate.csv and aggregate-report.md"
