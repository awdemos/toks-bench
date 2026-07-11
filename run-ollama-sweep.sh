#!/usr/bin/env bash
# Run toks-bench against the Ollama providers only.
# Ollama is CPU-bound on this host because the only GPU is occupied by the
# root-owned vLLM qwen server, so 512-token runs exceed the wall-clock timeout.
# This script completes the remaining Ollama combinations after the live sweep
# was stopped to avoid wasting time on the already-verified vLLM qwen results.

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

RESULTS_DIR="results/full"
mkdir -p "$RESULTS_DIR"

OUTER_TIMEOUT=900
RUNS=3
PROMPTS=(short medium long code tool)

declare -a PROVIDERS=(
    ollama-qwen
    ollama-ornith
)

LOG="$RESULTS_DIR/ollama-sweep-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG")
exec 2>&1

echo "=== starting ollama sweep at $(date -Iseconds) ==="
echo "results dir: $RESULTS_DIR"
echo "outer timeout per command: ${OUTER_TIMEOUT}s"
echo "runs per prompt: $RUNS"
echo "providers: ${PROVIDERS[*]}"
echo "prompts: ${PROMPTS[*]}"
echo "NOTE: Ollama is CPU-bound because the only GPU is in use by vLLM qwen (root process)."
echo "      Expect ReadTimeout results for 512-token runs."

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

echo "=== $(date -Iseconds) === ollama sweep complete ==="
echo "failed combinations: $FAILED"

echo "=== aggregating results ==="
toks-bench-aggregate "$RESULTS_DIR"

echo "=== done at $(date -Iseconds) ==="
