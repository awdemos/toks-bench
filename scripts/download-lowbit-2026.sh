#!/bin/bash
# Download the five selected 2026 low-bit models for toks-bench profiling.
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/home/andrewh/models/lowbit-2026}"
mkdir -p "$TARGET_DIR"

verify_sha256() {
  local file="$1" expected="${2:-}"
  if [ -z "$expected" ]; then
    return 0
  fi
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: checksum mismatch for $file (expected $expected, got $actual)" >&2
    return 1
  fi
  echo "Verified SHA-256 of $file"
}

download_file() {
  local repo="$1"
  local file="$2"
  local expected_sha256="${3:-}"
  local target="$TARGET_DIR/$file"
  echo "=== Downloading $repo / $file ==="
  hf download "$repo" "$file" --local-dir "$TARGET_DIR"
  if [ -n "$expected_sha256" ]; then
    verify_sha256 "$target" "$expected_sha256"
  else
    echo "Record this SHA-256 and pass it as the third argument to verify on subsequent runs:"
    sha256sum "$target"
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

echo "=== All low-bit models downloaded to $TARGET_DIR ==="
ls -lh "$TARGET_DIR"/*.gguf
