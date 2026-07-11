#!/bin/bash
# Benchmark the two ready Spark-compatible models via vLLM Docker.
# gemma4-26b-a4b-nvfp4 and diffusiongemma-26b-a4b-it
set -euo pipefail

WORKDIR="/home/andrewh/code/spark-vllm-docker/toks-bench"
RESULTS="$WORKDIR/results/full"
LOG="$RESULTS/spark-models-sweep-$(date +%Y%m%d-%H%M%S).log"
PROMPTS="short medium long code tool"
RUNS=3
TIMEOUT=900
VLLM_IMAGE="vllm-node:latest"

cd "$WORKDIR"
source .venv/bin/activate
mkdir -p "$RESULTS"
exec > >(tee -a "$LOG")
exec 2>&1

log() { echo "=== $(date -Iseconds) === $*"; }

source "$WORKDIR/scripts/manage-default-llama-server.sh"

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

stop_other_vllm() {
  docker stop vllm-gemma4-8005 vllm-diffusiongemma-8006 2>/dev/null || true
  docker rm vllm-gemma4-8005 vllm-diffusiongemma-8006 2>/dev/null || true
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

default_server_stop
stop_other_vllm

# 1) Gemma4-26B-A4B-NVFP4 on port 8005
log "Starting vLLM Gemma4-26B-A4B-NVFP4 on port 8005"
docker run --rm \
  --name vllm-gemma4-8005 \
  --gpus all \
  --ipc=host \
  -p 8005:8005 \
  -v /home/andrewh/models:/models:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$VLLM_IMAGE" \
  vllm serve /models/gemma4-26b-a4b-nvfp4 \
    --served-model-name gemma4-26b-a4b-nvfp4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --port 8005 --host 0.0.0.0 \
    --load-format instanttensor \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 \
    2>&1 | tee /tmp/vllm-gemma4-8005.log &

if health_check "http://localhost:8005"; then
  run_provider_sweep "vllm-gemma4-26b-a4b-nvfp4"
else
  log "Gemma4-26B-A4B-NVFP4 failed to start; skipping"
fi

docker stop vllm-gemma4-8005 2>/dev/null || true
sleep 5

# 2) DiffusionGemma-26B-A4B-IT (BF16) on port 8006
log "Starting vLLM DiffusionGemma-26B-A4B-IT on port 8006"
docker run --rm \
  --name vllm-diffusiongemma-8006 \
  --gpus all \
  --ipc=host \
  -p 8006:8006 \
  -v /home/andrewh/models:/models:ro \
  -v "/home/andrewh/code/spark-vllm-docker/mods/diffusiongemma:/diffusiongemma-mod:ro" \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$VLLM_IMAGE" \
  bash -c "bash /diffusiongemma-mod/run.sh && vllm serve /models/diffusiongemma-26b-a4b-it \
    --served-model-name diffusiongemma-26b-a4b-it \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --port 8006 --host 0.0.0.0 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --load-format fastsafetensors \
    --attention-backend TRITON_ATTN \
    --max-num-seqs 10 \
    --enable-prefix-caching \
    --diffusion-config '{\"canvas_length\":256}' \
    --override-generation-config '{\"max_new_tokens\": null}' \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{\"enable_thinking\": false}' \
    --moe-backend triton" \
  2>&1 | tee /tmp/vllm-diffusiongemma-8006.log &

if health_check "http://localhost:8006"; then
  run_provider_sweep "vllm-diffusiongemma-26b-a4b-it"
else
  log "DiffusionGemma-26B-A4B-IT failed to start; skipping"
fi

docker stop vllm-diffusiongemma-8006 2>/dev/null || true

log "Spark model sweep complete. Regenerating reports..."
toks-bench-aggregate "$RESULTS" --csv "$WORKDIR/results/aggregate.csv" --report "$WORKDIR/results/aggregate-report.md"
toks-bench-profile --csv "$WORKDIR/results/aggregate.csv" --results-dir "$RESULTS" --output "$WORKDIR/results/profiling-report.md"
toks-bench-chart --csv "$WORKDIR/results/aggregate.csv" --output "$WORKDIR/results/charts"
python3 "$WORKDIR/scripts/generate_dashboard.py" --csv "$WORKDIR/results/aggregate.csv" --charts "$WORKDIR/results/charts" --output "$WORKDIR/results/dashboard.html"
log "Done. Aggregate, profiling report, charts, and dashboard regenerated."

default_server_start
log "Default llama-server-qwen restored."
