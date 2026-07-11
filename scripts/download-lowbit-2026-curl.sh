#!/bin/bash
# Download the five selected 2026 low-bit models using direct HF URLs.
set -euo pipefail

TARGET_DIR="/home/andrewh/models/lowbit-2026"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

download_file() {
  local repo="$1"
  local file="$2"
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  echo "=== Downloading ${repo}/${file} ==="
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=10 --timeout=60 --show-progress "${url}" -O "${file}"
  else
    echo "wget not found; trying curl"
    curl -L --retry 10 --retry-delay 5 --connect-timeout 60 --max-time 0 --progress-bar \
      -o "${file}" "${url}"
  fi
}

# 1. 2-bit 27B dense (Cerebellum Q2_K imatrix)
download_file deucebucket/Qwen3.6-27B-Cerebellum-Q2K-GGUF qwen3.6-27b-cerebellum-imatrix-Q2_K.gguf

# 2. 2-bit 27B dense (AutoRound Q2_K mixed)
download_file sphaela/Qwen3.6-27B-AutoRound-GGUF Qwen3.6-27B-Q2_K_MIXED.gguf

# 3. 2.17-bit 35B MoE A3B (ByteShape IQ2_S)
download_file byteshape/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf

# 4. 2-bit 4B small baseline
download_file DhruvalLabs/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q2_K.gguf

# 5. 1.58-bit ternary 8B (experimental, Q2_0 container)
download_file prism-ml/Ternary-Bonsai-8B-gguf Ternary-Bonsai-8B-Q2_0.gguf

echo "=== All low-bit models downloaded to $(pwd) ==="
ls -lh ./*.gguf
