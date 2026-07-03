#!/bin/bash
# Install vLLM and test dependencies in llm_vit conda environment

CONDA_BIN="/home/cuong/anaconda3/bin/conda"
ENV_NAME="llm_vit"

echo "Installing packages in $ENV_NAME..."

$CONDA_BIN run -n $ENV_NAME pip install \
  "openai>=1.0.0" \
  "aiohttp>=3.9.0" \
  "requests>=2.31.0" \
  "httpx>=0.25.0" \
  "tqdm>=4.66.0" \
  "numpy>=1.24.0" \
  2>&1

echo "Done! Test packages installed."
echo "To run benchmark: conda activate $ENV_NAME && python test_server.py"
