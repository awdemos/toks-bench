#!/bin/bash
# Download Spark-compatible (single-GB10) models from recipe-derived HF IDs.
set -euo pipefail

MODEL_BASE="${MODEL_BASE:-/home/andrewh/models}"
mkdir -p "$MODEL_BASE"

generate_manifest() {
  local dir="$1"
  local manifest="$MODEL_BASE/$dir.sha256"
  find "$MODEL_BASE/$dir" -type f -exec sha256sum {} \; | sort > "$manifest"
  echo "Recorded manifest at $manifest"
}

verify_manifest() {
  local dir="$1" expected="$2"
  local manifest="$MODEL_BASE/$dir.sha256"
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
  echo "=== Downloading $repo into $dir ==="
  mkdir -p "$MODEL_BASE/$dir"
  (cd "$MODEL_BASE/$dir" && hf download "$repo" --local-dir . 2>&1 | tee "/tmp/download-${dir}.log")
  generate_manifest "$dir"
  if [ -n "$expected_manifest" ]; then
    verify_manifest "$dir" "$expected_manifest"
  fi
}

# NVIDIA Nemotron-3-Nano-30B-A3B-NVFP4 (already downloading in another session; include for completeness)
download_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 nemotron-3-nano-30b-a3b-nvfp4 &

# OpenAI GPT-OSS-120B with MXFP4 (solo recipe)
download_model openai/gpt-oss-120b openai-gpt-oss-120b &

# Gemma-4-26B-A4B-NVFP4 (solo recipe)
download_model nvidia/Gemma-4-26B-A4B-NVFP4 gemma4-26b-a4b-nvfp4 &

# DiffusionGemma 26B NVFP4 (solo recipe)
download_model nvidia/diffusiongemma-26B-A4B-it-NVFP4 diffusiongemma-26b-a4b-it-nvfp4 &

# DiffusionGemma 26B BF16 (solo recipe, larger but fits per recipe)
download_model google/diffusiongemma-26B-A4B-it diffusiongemma-26b-a4b-it &

wait
echo "=== All Spark-compatible model downloads complete ==="
