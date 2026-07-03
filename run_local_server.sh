#!/bin/bash
# Run vLLM server locally using conda llm env (has vLLM 0.21.1)
# For local testing on RTX A6000

MODEL_PATH="Qwen/Qwen3.5-2B"
HF_HOME="/home/cuong/.cache/huggingface"

echo "Starting vLLM server with model: $MODEL_PATH"
echo "Using conda env: llm"

/home/cuong/anaconda3/bin/conda run -n llm \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3.5-2B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --num-scheduler-steps 10 \
  --disable-log-requests \
  2>&1 | tee /home/cuong/Downloads/thi_llm/server.log
