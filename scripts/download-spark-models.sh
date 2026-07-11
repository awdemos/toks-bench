#!/bin/bash
# Download Spark-compatible (single-GB10) models from recipe-derived HF IDs.
set -euo pipefail

MODEL_BASE="/home/andrewh/models"
mkdir -p "$MODEL_BASE"

download_model() {
  local repo="$1"
  local dir="$2"
  echo "=== Downloading $repo into $dir ==="
  mkdir -p "$MODEL_BASE/$dir"
  (cd "$MODEL_BASE/$dir" && hf download "$repo" --local-dir . 2>&1 | tee "/tmp/download-${dir}.log")
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
