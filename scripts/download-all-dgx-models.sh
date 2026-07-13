#!/bin/bash
# Download all recommended DGX Spark benchmark models in parallel.
set -euo pipefail

MODELS_BASE="${MODELS_BASE:-/home/andrewh/models}"
mkdir -p "$MODELS_BASE"

generate_manifest() {
  local dir="$1"
  local manifest="$MODELS_BASE/$dir.sha256"
  find "$MODELS_BASE/$dir" -type f -exec sha256sum {} \; | sort > "$manifest"
  echo "Recorded manifest at $manifest"
}

verify_manifest() {
  local dir="$1" expected="$2"
  local manifest="$MODELS_BASE/$dir.sha256"
  if [ ! -f "$expected" ]; then
    echo "WARNING: expected manifest not found: $expected" >&2
    return 0
  fi
  if [ ! -f "$manifest" ]; then
    echo "WARNING: generated manifest not found: $manifest" >&2
    return 0
  fi
  if ! diff -q "$expected" "$manifest" >/dev/null 2>&1; then
    echo "ERROR: manifest mismatch for $dir" >&2
    diff "$expected" "$manifest" >&2 || true
    return 1
  fi
  echo "Verified manifest for $dir"
}

download_model() {
  local repo="$1"
  local dir="$2"
  local expected_manifest="${3:-}"
  local target_dir="$MODELS_BASE/$dir"
  echo "=== Starting $repo -> $target_dir ==="
  mkdir -p "$target_dir"
  hf download "$repo" --local-dir "$target_dir" 2>&1 | tee "/tmp/download-$(basename "$target_dir").log"
  generate_manifest "$dir"
  if [ -n "$expected_manifest" ]; then
    verify_manifest "$dir" "$expected_manifest"
  fi
  echo "=== Finished $repo ==="
}

# 1. Nemotron-3-Nano-30B-A3B-NVFP4 (already downloading elsewhere, but include for completeness)
download_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 nemotron-3-nano-30b-a3b-nvfp4 &

# 2. DiffusionGemma 26B-A4B (text-to-image reasoning) - BF16 variant
download_model google/diffusiongemma-26B-A4B-it diffusiongemma-26b-a4b-it &

# 3. OpenAI GPT-OSS-120B (MXFP4 recipe exists)
download_model openai/gpt-oss-120b gpt-oss-120b &

# 4. GLM-4.7-Flash-AWQ (4-bit AWQ, solo recipe)
download_model cyankiwi/GLM-4.7-Flash-AWQ-4bit glm-4.7-flash-awq &

# 5. Qwen3.6-35B-A3B-NVFP4 (NVIDIA NVFP4 variant)
download_model nvidia/Qwen3.6-35B-A3B-NVFP4 qwen3.6-35b-a3b-nvfp4 &

wait
echo "=== All DGX Spark benchmark models downloaded ==="
ls -lh "$MODELS_BASE"
