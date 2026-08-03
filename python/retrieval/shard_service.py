import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from retrieval.shard import Shard

# Which epoch this specific VM's shard service represents.
# Read from an environment variable so the SAME code runs on every shard VM,
# just configured differently -- no need for three separate copies of this file.
EPOCH = int(os.environ.get("SHARD_EPOCH", "1"))
INDEX_TYPE = os.environ.get("SHARD_INDEX_TYPE", "flat")

app = FastAPI()
shard = Shard(epoch=EPOCH, index_type=INDEX_TYPE)


class SearchRequest(BaseModel):
    embedding: list[float]
    k: int = 10


@app.post("/search")
def search(request: SearchRequest):
    query_embedding = np.array(request.embedding, dtype="float32")
    results = shard.search(query_embedding, k=request.k)
    return {"epoch": EPOCH, "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "epoch": EPOCH, "index_type": INDEX_TYPE}