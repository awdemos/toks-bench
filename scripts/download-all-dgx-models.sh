#!/bin/bash
# Download all recommended DGX Spark benchmark models in parallel.
set -euo pipefail

MODELS_BASE="/home/andrewh/models"
mkdir -p "$MODELS_BASE"

download_model() {
  local repo="$1"
  local target_dir="$MODELS_BASE/$2"
  echo "=== Starting $repo -> $target_dir ==="
  mkdir -p "$target_dir"
  hf download "$repo" --local-dir "$target_dir" 2>&1 | tee "/tmp/download-$(basename "$target_dir").log"
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
