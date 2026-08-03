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
from retrieval.router import Router

# Real shard IPs -- worker1 has no SSH-alias resolution for "worker2" etc,
# so we use the raw IPs directly, same as our earlier reachability test.
SHARD_URLS = [
    "http://192.168.64.34:8000",   # worker2, epoch 1
    "http://192.168.66.113:8000",  # worker3, epoch 2
    "http://192.168.67.240:8000",  # worker4, epoch 3
]

app = FastAPI()
router = Router()


class QueryRequest(BaseModel):
    query: str
    k: int = 10


@app.post("/query")
def query(request: QueryRequest):
    query_emb = router.encode_query(request.query)

    results_per_shard = []
    for url in SHARD_URLS:
        try:
            resp = requests.post(f"{url}/search", json={"embedding": query_emb.tolist(), "k": request.k}, timeout=10)
            resp.raise_for_status()
            results_per_shard.append(resp.json()["results"])
        except requests.RequestException as e:
            print(f"Shard {url} failed: {e}")
            # Continue with whichever shards DID respond, rather than failing the whole query
            continue

    fused = router.fuse_results(results_per_shard, k=request.k)
    return {"query": request.query, "results": fused}


@app.get("/health")
def health():
    return {"status": "ok"}