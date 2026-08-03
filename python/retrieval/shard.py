import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
import faiss

EPOCH_TO_MODEL = {1: "v1", 2: "v2", 3: "v3"}


class Shard:
    """Represents one epoch's retrieval service: loads its own adapter
    (if it's not the native v3 generation) and its own FAISS index,
    and answers queries handed to it as raw v3 embeddings."""

    def __init__(self, epoch, index_type="flat"):
        self.epoch = epoch
        self.model_key = EPOCH_TO_MODEL[epoch]
        self.index_type = index_type

        index_path = f"indexes/generation{epoch}/{index_type}.index"
        self.index = faiss.read_index(index_path)

        if self.model_key == "v3":
            self.adapter = None  # native generation, no compensation needed
        else:
            adapter_path = f"checkpoints/adapters/R_v3_to_{self.model_key}.npy"
            self.adapter = np.load(adapter_path)

        cleaned_path = f"data/generation{epoch}/cleaned"
        import pandas as pd
        df = pd.read_parquet(cleaned_path)
        self.doc_lookup = {row["row_num"]: (row["title"], row["text"]) for _, row in df.iterrows()}

    def search(self, query_v3_embedding, k=10):
        """query_v3_embedding: a single (1024,) numpy array, already
        normalized, from the router's V3 encoder."""
        q = query_v3_embedding.reshape(1, -1)

        if self.adapter is not None:
            q = q @ self.adapter

        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        q = q.astype("float32")

        distances, doc_ids = self.index.search(q, k)

        results = []
        for score, doc_id in zip(distances[0], doc_ids[0]):
            if doc_id == -1:
                continue  # FAISS pads with -1 if fewer than k results exist
            title, text = self.doc_lookup.get(int(doc_id), ("", ""))
            results.append({
                "doc_id": int(doc_id),
                "score": float(score),
                "title": title,
                "text": text[:300],  # preview only, not the full body
            })
        return results