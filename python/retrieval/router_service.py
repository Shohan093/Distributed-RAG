import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
import json
from retrieval.router import Router
from concurrent.futures import ThreadPoolExecutor, as_completed

# Real shard IPs -- worker1 has no SSH-alias resolution for "worker2" etc,
# so we use the raw IPs directly, same as our earlier reachability test.
SHARD_URLS = [
    "http://192.168.64.34:8000",   # worker2, epoch 1
    "http://192.168.66.113:8000",  # worker3, epoch 2
    "http://192.168.67.240:8000",  # worker4, epoch 3
]

TIMING_LOG_PATH = "$HOME/drift-rag/results/tables/timing_log.jsonl"

app = FastAPI()
router = Router()


class QueryRequest(BaseModel):
    query: str
    k: int = 10

def call_shard(url, query_emb_list, k):
    try:
        resp = requests.post(f"{url}/search", json={"embedding": query_emb_list, "k": k}, timeout=10)
        resp.raise_for_status()
        return resp.json()["results"]
    except requests.RequestException as e:
        print(f"Shard {url} failed: {e}")
        return None

@app.post("/query")
def query(request: QueryRequest):
    t0 = time.perf_counter()
    query_emb = router.encode_query(request.query)
    t1 = time.perf_counter()
    query_emb_list = query_emb.tolist()

    results_per_shard = []
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as executor:
        futures = [executor.submit(call_shard, url, query_emb_list, request.k) for url in SHARD_URLS]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results_per_shard.append(result)
    t2 = time.perf_counter()

    fused = router.fuse_results(results_per_shard, k=request.k)
    t3 = time.perf_counter()

    timing = {
        "encode_ms": 1000 * (t1 - t0),
        "shard_fanout_ms": 1000 * (t2 - t1),
        "fuse_ms": 1000 * (t3 - t2),
        "total_ms": 1000 * (t3 - t0),
    }
    print(f"[TIMING] encode={timing['encode_ms']:.1f}ms  shard_fanout={timing['shard_fanout_ms']:.1f}ms  "
          f"fuse={timing['fuse_ms']:.1f}ms  total={timing['total_ms']:.1f}ms")

    os.makedirs(os.path.dirname(TIMING_LOG_PATH), exist_ok=True)
    with open(TIMING_LOG_PATH, "a") as f:
        f.write(json.dumps(timing) + "\n")

    return {"query": request.query, "results": fused}


@app.get("/health")
def health():
    return {"status": "ok"}