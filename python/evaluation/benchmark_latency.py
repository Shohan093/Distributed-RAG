import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
import requests

ROUTER_URL = "http://192.168.66.242:8000/query"  # worker1
N_QUERIES = 100


def load_sample_queries():
    """Pull real query titles from the eval sets across all three epochs,
    take a mixed sample -- real text, not synthetic."""
    all_titles = []
    for epoch in [1, 2, 3]:
        with open(f"data/eval/generation{epoch}/queries.json") as f:
            records = json.load(f)
        all_titles.extend([r["title"] for r in records])

    rng = np.random.default_rng(42)
    sample = rng.choice(all_titles, size=min(N_QUERIES, len(all_titles)), replace=False)
    return list(sample)


def benchmark():
    queries = load_sample_queries()
    print(f"Benchmarking {len(queries)} real queries against {ROUTER_URL}...")

    latencies_ms = []
    failures = 0

    for i, query_text in enumerate(queries):
        start = time.perf_counter()
        try:
            resp = requests.post(ROUTER_URL, json={"query": query_text, "k": 10}, timeout=15)
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
        except requests.RequestException as e:
            print(f"Query {i} failed: {e}")
            failures += 1

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(queries)} done...")

    latencies_ms = np.array(latencies_ms)
    result = {
        "n_queries": len(queries),
        "n_successful": len(latencies_ms),
        "n_failures": failures,
        "mean_ms": float(latencies_ms.mean()),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": float(latencies_ms.min()),
        "max_ms": float(latencies_ms.max()),
        "raw_latencies_ms": latencies_ms.tolist(),
    }

    print(f"\nMean: {result['mean_ms']:.1f}ms  Median: {result['median_ms']:.1f}ms  "
          f"P95: {result['p95_ms']:.1f}ms  P99: {result['p99_ms']:.1f}ms")

    os.makedirs("~/drift-rag/results/tables", exist_ok=True)
    with open("~/drift-rag/results/tables/latency_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved ~/drift-rag/results/tables/latency_benchmark.json")


if __name__ == "__main__":
    benchmark()