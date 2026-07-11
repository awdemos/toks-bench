#!/usr/bin/env bash
# Run toks-bench against the inference servers that are currently live.
# Live probing (curl) on 2026-07-09 showed:
#   - port 8000 (vllm-qwen-fp8 / vllm-qwen-fp8-docker) -> up
#   - port 11434 (ollama-qwen / ollama-ornith) -> up
#   - port 8080 (llama-server-qwen) -> down
#   - port 8001 (vllm-nematron-fp8-docker) -> down
#   - port 8002 (tensorrt-nematron-fp8) -> down
#
# Default config uses 512 output tokens and 10 runs; the vLLM qwen endpoint
# needs ~75 s per 512-token run, so we use --runs 3 and a generous outer
# timeout to keep total sweep time manageable while still producing useful
# statistics.

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

RESULTS_DIR="results/full"
mkdir -p "$RESULTS_DIR"

OUTER_TIMEOUT=900
RUNS=3
PROMPTS=(short medium long code tool)

declare -a PROVIDERS=(
    vllm-qwen-fp8
    ollama-qwen
    ollama-ornith
)

LOG="$RESULTS_DIR/live-sweep-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG")
exec 2>&1

echo "=== starting live sweep at $(date -Iseconds) ==="
echo "results dir: $RESULTS_DIR"
echo "outer timeout per command: ${OUTER_TIMEOUT}s"
echo "runs per prompt: $RUNS"
echo "providers: ${PROVIDERS[*]}"
echo "prompts: ${PROMPTS[*]}"

FAILED=0
for provider in "${PROVIDERS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        out="$RESULTS_DIR/${provider}-${prompt}.json"
        echo "=== $(date -Iseconds) === $provider / $prompt ==="
        if timeout "$OUTER_TIMEOUT" toks-bench --provider "$provider" --prompt "$prompt" \
            --runs "$RUNS" --format json --output "$out"; then
            echo "OK on $provider / $prompt -> $out"
        else
            rc=$?
            echo "FAILED ($rc) on $provider / $prompt"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "=== $(date -Iseconds) === sweep complete ==="
echo "failed combinations: $FAILED"

echo "=== aggregating results ==="
toks-bench-aggregate "$RESULTS_DIR"

echo "=== done at $(date -Iseconds) ==="
