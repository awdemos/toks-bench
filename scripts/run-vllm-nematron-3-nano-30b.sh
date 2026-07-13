#!/bin/bash
# Launch vLLM serving NVIDIA Nemotron-3-Nano-30B-A3B-NVFP4 via Docker and benchmark it.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/vllm-nematron-3-nano-30b-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=1200
PORT=8010
CONTAINER_NAME="vllm-nematron-30b-8010"
# SECURITY: Pin this image to a digest to avoid mutable-tag supply-chain attacks.
# Example: VLLM_IMAGE="vllm-node@sha256:..."
VLLM_IMAGE="${VLLM_IMAGE:-vllm-node:latest}"
MODEL_DIR="/models/nemotron-3-nano-30b-a3b-nvfp4"

# Pinned plugin download.  If upstream updates the file, update this hash.
REASONING_PARSER_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py"
REASONING_PARSER_SHA256="aafb12208054504f619cbdd01837e1532a482ad937ed987bfe9a13fb812ae2b7"
REASONING_PARSER_NAME="nano_v3_reasoning_parser.py"

cd "$WORKDIR"
source .venv/bin/activate
source "$WORKDIR/scripts/manage-default-llama-server.sh"
mkdir -p "$RESULTS"
exec > >(tee -a "$LOG")
exec 2>&1

log() { echo "=== $(date -Iseconds) === $*"; }

# Verify a file against a SHA-256 hash.  Exits non-zero on mismatch.
verify_sha256() {
  local file=$1 expected=$2
  if ! command -v sha256sum >/dev/null 2>&1; then
    log "ERROR: sha256sum not available"
    return 1
  fi
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    log "ERROR: checksum mismatch for $file (expected $expected, got $actual)"
    return 1
  fi
  log "Verified SHA-256 of $file"
}

# Download a URL to a file and verify its SHA-256 hash.
download_verified() {
  local url=$1 output=$2 expected=$3
  local tmp
  tmp=$(mktemp)
  trap 'rm -f "$tmp"' RETURN
  if command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$tmp"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL --max-time 60 "$url" -o "$tmp"
  else
    log "ERROR: wget or curl required"
    return 1
  fi
  verify_sha256 "$tmp" "$expected"
  mv "$tmp" "$output"
}

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

HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/andrewh/models/nemotron-3-nano-30b-a3b-nvfp4}"
if [ ! -d "$HOST_MODEL_DIR" ] || [ -z "$(find "$HOST_MODEL_DIR" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  log "Model not found at $HOST_MODEL_DIR; cannot continue"
  exit 1
fi

# Download and verify the reasoning-parser plugin before mounting it into the container.
PLUGIN_DIR="$WORKDIR/.plugins"
mkdir -p "$PLUGIN_DIR"
PLUGIN_FILE="$PLUGIN_DIR/$REASONING_PARSER_NAME"
if [ ! -f "$PLUGIN_FILE" ]; then
  log "Downloading verified reasoning parser plugin"
  download_verified "$REASONING_PARSER_URL" "$PLUGIN_FILE" "$REASONING_PARSER_SHA256"
else
  verify_sha256 "$PLUGIN_FILE" "$REASONING_PARSER_SHA256" || {
    log "Existing plugin failed verification; re-downloading"
    rm -f "$PLUGIN_FILE"
    download_verified "$REASONING_PARSER_URL" "$PLUGIN_FILE" "$REASONING_PARSER_SHA256"
  }
fi

log "Starting vLLM Nemotron-3-Nano-30B-A3B-NVFP4 on port $PORT"
docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  -p "$PORT:$PORT" \
  -v "$HOST_MODEL_DIR:$MODEL_DIR:ro" \
  -v "$PLUGIN_FILE:/opt/$REASONING_PARSER_NAME:ro" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -w "$MODEL_DIR" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL_DIR" \
    --moe-backend cutlass \
    --max-model-len 8192 \
    --port "$PORT" --host 0.0.0.0 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "/opt/$REASONING_PARSER_NAME" \
    --reasoning-parser nano_v3 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --gpu-memory-utilization 0.7 \
  2>&1 | tee /tmp/vllm-nematron-30b-serve.log &

if ! health_check "http://localhost:$PORT"; then
  log "vLLM Nemotron-3-Nano failed to start"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  exit 1
fi

for prompt in $PROMPTS; do
  run_bench "vllm-nematron-3-nano-30b-a3b-nvfp4" "$prompt"
done

docker stop "$CONTAINER_NAME" 2>/dev/null || true

log "Nemotron-3-Nano sweep complete. Regenerating reports..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
toks-bench-profile --csv "$WORKDIR/results/aggregate.csv" --results-dir "$RESULTS" --output "$WORKDIR/results/profiling-report.md"
toks-bench-chart --csv "$WORKDIR/results/aggregate.csv" --output "$WORKDIR/results/charts"
python "$WORKDIR/scripts/generate_dashboard.py" --csv "$WORKDIR/results/aggregate.csv" --charts "$WORKDIR/results/charts" --output "$WORKDIR/results/dashboard.html"
log "Done. Aggregate, profiling report, charts, and dashboard regenerated."

log "Restoring permanent default llama-server on port 8080"
default_server_start || true
