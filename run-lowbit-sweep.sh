#!/bin/bash
# Sequential benchmark sweep for 2026 low-bit llama-server models.
# Runs one model at a time on the GB10 GPU to avoid overcommit.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/lowbit-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=900
LLAMA_SERVER="${LLAMA_SERVER:-/home/andrewh/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-/home/andrewh/models/lowbit-2026}"

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

stop_all_llama() {
  default_server_stop
  pkill -f "llama-server.*--port 808[1-9]" 2>/dev/null || true
  sleep 3
}

run_llama_server() {
  local port=$1
  local model=$2
  local ctx=${3:-8192}
  shift 3
  "$LLAMA_SERVER" \
    -m "$model" \
    --host 127.0.0.1 --port "$port" -ngl 99 -c "$ctx" \
    --cont-batching --flash-attn on --temp 0.7 \
    --jinja "$@" \
    2>&1 | tee "/tmp/llama-server-lowbit-${port}.log" &
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

# Free the GPU from other llama-server / vllm containers.
stop_all_llama
docker stop vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
docker rm vllm-nematron-8001 vllm-qwen-8000 2>/dev/null || true
sleep 5

# Provider definitions: name port gguf_file [extra llama-server args]
providers=(
  "llama-server-qwen-q2k:8083:qwen3.6-27b-cerebellum-imatrix-Q2_K.gguf"
  "llama-server-qwen-autoround-q2k:8084:Qwen3.6-27B-Q2_K_MIXED.gguf"
  "llama-server-qwen36-a3b-iq2:8085:Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf"
  "llama-server-qwen3-4b-q2k:8086:Qwen3-4B-Instruct-2507-Q2_K.gguf"
  "llama-server-ternary-bonsai-8b:8087:Ternary-Bonsai-8B-Q2_0.gguf"
)

for entry in "${providers[@]}"; do
  IFS=':' read -r provider port gguf <<< "$entry"
  model_path="$MODEL_DIR/$gguf"

  if [ ! -f "$model_path" ]; then
    log "Skipping $provider: model not found at $model_path"
    continue
  fi

  log "Starting $provider on port $port with $model_path"
  run_llama_server "$port" "$model_path" 8192

  if ! health_check "http://localhost:${port}"; then
    log "$provider failed health check; stopping and continuing"
    stop_all_llama
    continue
  fi
  if ! model_ready_check "http://localhost:${port}"; then
    log "$provider model not ready; stopping and continuing"
    stop_all_llama
    continue
  fi

  run_provider_sweep "$provider"
  stop_all_llama
  sleep 5
done

log "Low-bit sweep complete. Aggregating..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
log "Done. Aggregate written to $WORKDIR/results/aggregate.csv and aggregate-report.md"

# Restore the permanent default llama-server on 8080.
log "Restoring permanent default llama-server on port 8080"
default_server_start || true
log "llama-server-qwen restored."
