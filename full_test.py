import asyncio, time, statistics
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:7500/v1", api_key="none")

PROMPTS = [
    "What is machine learning?", "Explain neural networks briefly.", "What is Python programming language?",
    "Describe quantum computing.", "What is the internet?", "Explain DNA structure.",
    "What causes earthquakes?", "How does WiFi work?", "What is inflation?",
    "Describe photosynthesis.", "What is gravity?", "Explain blockchain technology.",
    "What is artificial intelligence?", "How do vaccines work?", "What is democracy?",
    "Explain climate change.", "What is a computer processor?", "How does the heart work?",
    "What is philosophy?", "Explain evolution theory.", "What is mathematics?",
    "How does language acquisition work?", "What is consciousness?", "Explain theory of relativity.",
]

async def send_request(prompt, request_id, max_tokens=80):
    start = time.perf_counter()
    first_token_time = None
    token_times = []
    last_time = start
    token_count = 0
    try:
        stream = await client.chat.completions.create(
            model="Qwen3.5-2B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, stream=True, temperature=0
        )
        async for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                if first_token_time is None:
                    first_token_time = now
                    ttft = (first_token_time - start) * 1000
                else:
                    token_times.append((now - last_time) * 1000)
                last_time = now
                token_count += 1
        ttft_val = ttft if first_token_time else 9999
        tbt_val = statistics.median(token_times) if token_times else 0
        effective = first_token_time is not None and ttft_val <= 2000 and tbt_val <= 200
        return {"id": request_id, "ttft": ttft_val, "tbt": tbt_val, "tokens": token_count, "ok": effective}
    except Exception as e:
        return {"id": request_id, "ttft": 9999, "tbt": 9999, "tokens": 0, "ok": False, "err": str(e)}

async def run_test(n_requests, label):
    tasks = [send_request(PROMPTS[i % len(PROMPTS)], i) for i in range(n_requests)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0

    effective = [r for r in results if r["ok"]]
    erc = len(effective) / n_requests
    ttfts = sorted([r["ttft"] for r in results])
    tbts = [r["tbt"] for r in results if 0 < r["tbt"] < 9999]

    p50_ttft = statistics.median(ttfts)
    p95_ttft = ttfts[min(int(0.95 * n_requests), n_requests - 1)]
    p99_ttft = ttfts[min(int(0.99 * n_requests), n_requests - 1)]
    max_ttft = ttfts[-1]

    p50_tbt = statistics.median(tbts) if tbts else 0
    p95_tbt = sorted(tbts)[min(int(0.95 * len(tbts)), len(tbts)-1)] if tbts else 0

    vio_ttft = sum(1 for r in results if r["ttft"] > 2000)
    vio_tbt  = sum(1 for r in results if 0 < r["tbt"] < 9999 and r["tbt"] > 200)
    total_tok = sum(r["tokens"] for r in results)

    return {
        "label": label, "n": n_requests, "wall": wall, "erc": erc,
        "effective": len(effective),
        "p50_ttft": p50_ttft, "p95_ttft": p95_ttft, "p99_ttft": p99_ttft, "max_ttft": max_ttft,
        "p50_tbt": p50_tbt, "p95_tbt": p95_tbt,
        "vio_ttft": vio_ttft, "vio_tbt": vio_tbt,
        "total_tokens": total_tok,
        "throughput": total_tok / wall,
    }

def print_table(runs):
    SEP = "+" + "-"*22 + "+" + ("-"*12 + "+")*len(runs)
    HDR = "+" + "="*22 + "+" + ("="*12 + "+")*len(runs)

    def row(label, vals, fmt="{}", slo=None):
        cells = ""
        for i, v in enumerate(vals):
            txt = fmt.format(v)
            # Mark SLO violations
            if slo and isinstance(v, (int, float)):
                txt = txt + (" ✓" if v <= slo else " ✗")
            cells += f" {txt:>9} |"
        print(f"| {label:<20} |{cells}")

    print()
    print(HDR)
    print("| {:^20} |".format("METRIC") + "".join(f" {'Test '+r['label']:^10} |" for r in runs))
    print(HDR)

    row("Concurrent Requests", [r["n"] for r in runs], "{:,}")
    row("Wall Time (s)", [r["wall"] for r in runs], "{:.2f}s")
    print(SEP)
    row("ERC (Effective %)", [r["erc"]*100 for r in runs], "{:.1f}%")
    row("Effective / Total", [f"{r['effective']}/{r['n']}" for r in runs])
    print(SEP)
    row("TTFT p50 (ms)", [r["p50_ttft"] for r in runs], "{:.0f}", slo=2000)
    row("TTFT p95 (ms)", [r["p95_ttft"] for r in runs], "{:.0f}", slo=2000)
    row("TTFT p99 (ms)", [r["p99_ttft"] for r in runs], "{:.0f}", slo=2000)
    row("TTFT max (ms)", [r["max_ttft"] for r in runs], "{:.0f}", slo=2000)
    row("TTFT violations", [r["vio_ttft"] for r in runs], "{:,}")
    print(SEP)
    row("TBT p50 (ms)", [r["p50_tbt"] for r in runs], "{:.0f}", slo=200)
    row("TBT p95 (ms)", [r["p95_tbt"] for r in runs], "{:.0f}", slo=200)
    row("TBT violations", [r["vio_tbt"] for r in runs], "{:,}")
    print(SEP)
    row("Total tokens", [r["total_tokens"] for r in runs], "{:,}")
    row("Throughput (tok/s)", [r["throughput"] for r in runs], "{:.0f}")
    print(HDR)
    row("SCORE (est.)", [r["erc"]*100 for r in runs], "{:.1f}")
    print(HDR)
    print()
    print("SLO: TTFT ≤ 2000ms  |  TBT_median ≤ 200ms  |  ✓ Pass  ✗ Fail")

async def main():
    print("\n" + "="*60)
    print("  vLLM Benchmark — Qwen3.5-2B | MTP spec | FP8 KV cache")
    print("="*60)
    print("Server: localhost:7500 | Model: Qwen3.5-2B (BF16 weights)")
    print("Config: chunked-prefill | prefix-cache | mtp spec-tokens=3")
    print()

    runs = []
    for n, label in [(20, "20 req"), (60, "60 req"), (120, "120 req")]:
        print(f"Running {n} concurrent requests...")
        result = await run_test(n, label)
        runs.append(result)
        print(f"  → ERC={result['erc']*100:.1f}%  TTFT_p50={result['p50_ttft']:.0f}ms  TBT_p50={result['p50_tbt']:.0f}ms")

    print_table(runs)

asyncio.run(main())
