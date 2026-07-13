#!/bin/bash
# Install TensorRT-LLM and PyTorch CUDA 13.0 in the toks-bench venv.
set -euo pipefail

WORKDIR="${WORKDIR:-/home/andrewh/spark-vllm-docker/toks-bench}"
# SECURITY: Pin these versions and verify hashes.  Do not install unpinned
# packages in CI or production.  Adjust the versions to match your platform.
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TENSORRT_LLM_VERSION="${TENSORRT_LLM_VERSION:-}"

cd "$WORKDIR"
source .venv/bin/activate

echo "=== Installing PyTorch CUDA 13.0 ==="
pip install --upgrade pip setuptools wheel
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu130

echo "=== Installing TensorRT-LLM ==="
if [ -n "$TENSORRT_LLM_VERSION" ]; then
  pip install "tensorrt_llm==${TENSORRT_LLM_VERSION}"
else
  echo "WARNING: TENSORRT_LLM_VERSION is not set; installing latest tensorrt_llm." >&2
  echo "Set a pinned version to avoid supply-chain drift." >&2
  pip install tensorrt_llm
fi

echo "=== Verifying installation ==="
trtllm-serve --help | head -n 20
python -c 'import tensorrt_llm; print(tensorrt_llm.__version__)'

echo "=== Done ==="
