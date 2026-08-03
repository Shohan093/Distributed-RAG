import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import load_config


class Router:
    """Encodes incoming queries with the current production model (v3)
    and fuses results returned by shards. Holds no adapters and no
    FAISS indexes itself -- those live entirely on the shards."""

    def __init__(self):
        config = load_config("configs/embedding.yaml")
        v3_name = config["models"]["v3"]["name"]
        self.encoder = SentenceTransformer(v3_name)

    def encode_query(self, query_text):
        emb = self.encoder.encode([query_text], normalize_embeddings=True)
        return emb[0]

    def fuse_results(self, results_per_shard, k=10, rrf_k=60):
        """Reciprocal Rank Fusion across heterogeneous shard result lists.
        results_per_shard: list of lists, each shard's results already
        ranked best-first (as returned by Shard.search)."""
        scores = {}
        doc_info = {}

        for shard_results in results_per_shard:
            for rank, result in enumerate(shard_results):
                doc_id = result["doc_id"]
                rrf_contribution = 1.0 / (rrf_k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution
                if doc_id not in doc_info:
                    doc_info[doc_id] = result

        ranked_ids = sorted(scores.keys(), key=lambda d: -scores[d])
        fused = []
        for doc_id in ranked_ids[:k]:
            entry = dict(doc_info[doc_id])
            entry["fused_score"] = scores[doc_id]
            fused.append(entry)
        return fused