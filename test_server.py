"""
Test script for vLLM server - simulate competition benchmark.
Tests TTFT and TBT latency with 20 concurrent requests.
Run with: conda activate llm_vit && python test_server.py
"""
import asyncio
import time
import json
import statistics
from openai import AsyncOpenAI

SERVER_PORT = 7500
client = AsyncOpenAI(base_url=f"http://localhost:{SERVER_PORT}/v1", api_key="none")

TEST_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France and why?",
    "Write a Python function to sort a list using quicksort.",
    "What are the main causes of climate change?",
    "Explain the theory of relativity briefly.",
    "What is machine learning and how does it work?",
    "Describe the process of photosynthesis in plants.",
    "How do neural networks learn patterns in data?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of entropy in thermodynamics.",
]

async def send_request(prompt: str, request_id: int, max_tokens: int = 150) -> dict:
    start = time.perf_counter()
    first_token_time = None
    token_times = []
    last_time = start
    token_count = 0

    try:
        stream = await client.chat.completions.create(
            model="Qwen3.5-2B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
        )
        async for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                if first_token_time is None:
                    first_token_time = now
                    ttft = (first_token_time - start) * 1000
                else:
                    tbt = (now - last_time) * 1000
                    token_times.append(tbt)
                last_time = now
                token_count += 1

        ttft_val = ttft if first_token_time else 9999
        tbt_val = statistics.median(token_times) if token_times else 0
        effective = first_token_time is not None and ttft_val <= 2000 and tbt_val <= 200
        
        return {
            "id": request_id,
            "ttft_ms": ttft_val,
            "tbt_median_ms": tbt_val,
            "tokens": token_count,
            "success": True,
            "effective": effective,
        }
    except Exception as e:
        return {
            "id": request_id,
            "ttft_ms": 9999,
            "tbt_median_ms": 9999,
            "tokens": 0,
            "success": False,
            "effective": False,
            "error": str(e),
        }


async def run_benchmark(n_requests: int = 20):
    print(f"Starting benchmark with {n_requests} concurrent requests on port {SERVER_PORT}")
    
    tasks = []
    for i in range(n_requests):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        tasks.append(send_request(prompt, i, max_tokens=150))

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    effective = [r for r in results if r["effective"]]
    successful = [r for r in results if r["success"]]
    erc = len(effective) / len(results)

    ttfts = [r["ttft_ms"] for r in successful]
    tbts = [r["tbt_median_ms"] for r in successful if r["tbt_median_ms"] > 0]

    print(f"\n{'='*55}")
    print(f"BENCHMARK RESULTS ({n_requests} concurrent requests)")
    print(f"{'='*55}")
    print(f"Total wall time : {total_time:.2f}s")
    print(f"Effective (ERC) : {erc*100:.1f}%  ({len(effective)}/{len(results)})")
    
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        p95_idx = min(int(0.95 * len(ttfts)), len(ttfts)-1)
        print(f"TTFT  p50={statistics.median(ttfts):.0f}ms  p95={ttfts_sorted[p95_idx]:.0f}ms  max={max(ttfts):.0f}ms")
    
    if tbts:
        tbts_sorted = sorted(tbts)
        p95_idx = min(int(0.95 * len(tbts)), len(tbts)-1)
        print(f"TBT   p50={statistics.median(tbts):.0f}ms  p95={tbts_sorted[p95_idx]:.0f}ms  max={max(tbts):.0f}ms")
    
    violations_ttft = sum(1 for r in results if r["ttft_ms"] > 2000)
    violations_tbt = sum(1 for r in results if r["tbt_median_ms"] > 200 and r["success"])
    print(f"SLO violations  : TTFT>{violations_ttft}  TBT>{violations_tbt}")
    print(f"Estimated score : {100 * erc:.1f}")
    print(f"{'='*55}")
    
    # Show individual results
    print("\nIndividual Results:")
    for r in results:
        status = "OK" if r["effective"] else "FAIL"
        if r["success"]:
            print(f"  [{status}] req#{r['id']:02d}: TTFT={r['ttft_ms']:.0f}ms  TBT={r['tbt_median_ms']:.0f}ms  tokens={r['tokens']}")
        else:
            print(f"  [ERR] req#{r['id']:02d}: {r.get('error', 'unknown error')}")

if __name__ == "__main__":
    asyncio.run(run_benchmark(n_requests=20))
