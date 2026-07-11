#!/bin/bash
# Install TensorRT-LLM and PyTorch CUDA 13.0 in the toks-bench venv.
set -euo pipefail

cd /home/andrewh/spark-vllm-docker/toks-bench
source .venv/bin/activate

echo "=== Installing PyTorch CUDA 13.0 ==="
pip install --upgrade pip setuptools wheel
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

echo "=== Installing TensorRT-LLM ==="
pip install tensorrt_llm

echo "=== Verifying installation ==="
trtllm-serve --help | head -n 20
python -c 'import tensorrt_llm; print(tensorrt_llm.__version__)'

echo "=== Done ==="
