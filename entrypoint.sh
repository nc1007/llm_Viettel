#!/bin/bash
INTERNAL_PORT=8001
EXTERNAL_PORT=8000

# Build args: replace --port=8000 with --port=8001
VLLM_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --port=*) VLLM_ARGS+=("--port=$INTERNAL_PORT") ;;
        --port)   ;;
        *)        VLLM_ARGS+=("$arg") ;;
    esac
done

echo "[entry] Starting vLLM on internal port $INTERNAL_PORT..."
python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# Wait for vLLM to be ready on internal port
echo "[entry] Waiting for vLLM to start..."
until curl -sf http://localhost:$INTERNAL_PORT/health > /dev/null 2>&1; do
    sleep 3
done
echo "[entry] vLLM healthy! Running warmup..."

# Warmup: compile Triton/CUDA kernels with varying request sizes
python3 << 'PYEOF'
import requests, time, sys

BASE = "http://localhost:8001/v1/chat/completions"

warmups = [
    ("What is 2+2?", 10),
    ("Name a planet.", 20),
    ("What color is the sky?", 30),
    ("Name a country in Asia.", 40),
    ("What is the speed of light?", 50),
    ("List the planets in order from the sun.", 60),
    ("What is quantum mechanics? Explain briefly.", 80),
    ("Describe the process of photosynthesis.", 100),
    ("What are the main differences between Python and Java?", 120),
    ("Explain Newton's three laws of motion.", 150),
]

print("[warmup] Sending 10 warmup requests to compile kernels...", flush=True)
for i, (prompt, max_tok) in enumerate(warmups):
    try:
        t0 = time.time()
        r = requests.post(BASE, json={
            "model": "Qwen3.5-2B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tok, "stream": False, "temperature": 0
        }, timeout=180)
        ttft = (time.time() - t0) * 1000
        print(f"[warmup] {i+1}/10: status={r.status_code} ttft={ttft:.0f}ms", flush=True)
    except Exception as e:
        print(f"[warmup] {i+1}/10: ERROR {e}", flush=True)

print("[warmup] Kernel warmup complete!", flush=True)
PYEOF

# Open external port via socat (more reliable than Python proxy)
echo "[entry] Warmup done! Opening port $EXTERNAL_PORT via socat..."
socat TCP-LISTEN:$EXTERNAL_PORT,fork,reuseaddr TCP:127.0.0.1:$INTERNAL_PORT &

echo "[entry] Server ready on port $EXTERNAL_PORT"
wait $VLLM_PID
