#!/bin/bash
pkill -9 -f 'vllm.entrypoints' 2>/dev/null
sleep 2

nohup /home/cuong/anaconda3/bin/conda run -n llm \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-2B \
  --served-model-name Qwen3.5-2B \
  --host 0.0.0.0 \
  --port 7500 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.54 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --spec-method mtp \
  --spec-tokens 3 \
  --performance-mode throughput \
  --no-enable-log-requests \
  > /home/cuong/Downloads/thi_llm/server.log 2>&1 &

echo "vLLM started PID: $! on port 7500 (gpu-util=0.54)"
