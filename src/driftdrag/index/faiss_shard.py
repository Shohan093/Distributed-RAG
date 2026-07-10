"""Per-shard FAISS index build/query (Stage 0/3).

FAISS is deliberately kept out of the Spark executors: PySpark writes
embeddings to Parquet (see ingest/spark_embed_job.py), and this module -- run
as a small standalone indexing step, one process per shard -- builds/updates
the ANN structure from that Parquet output. That split mirrors the intended
production architecture (Section 2 Stage 0): Spark/Kafka do ingest+embed at
data-lake scale, a lighter-weight indexing service builds the per-shard
HNSW/IVF-PQ structure. Milvus (see docker-compose.milvus.yml) is the
drop-in, natively-distributed replacement referenced in the plan for the
100M+/multi-VM scale-out experiments (E5/E6); this module is what backs the
pilot-scale (10M) single-VM experiments and unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissShard:
    def __init__(self, dim: int, index_type: str = "hnsw", hnsw_m: int = 32):
        self.dim = dim
        self.index_type = index_type
        if index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m)
        elif index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"unsupported index_type: {index_type}")
        self.ids: list[str] = []
        self.model_version: str | None = None

    def add(self, embeddings: np.ndarray, doc_ids: list[str], model_version: str) -> None:
        if self.ids and self.model_version and self.model_version != model_version:
            raise ValueError(
                f"shard already indexed under model version {self.model_version!r}; "
                f"got {model_version!r} -- re-embed the shard or use the compensation path, "
                "do not mix versions inside one FAISS index"
            )
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        self.index.add(normed.astype(np.float32))
        self.ids.extend(doc_ids)
        self.model_version = model_version

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        q = query / (np.linalg.norm(query) + 1e-12)
        scores, idx = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        meta = {"ids": self.ids, "model_version": self.model_version, "dim": self.dim, "index_type": self.index_type}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FaissShard":
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        shard = cls(dim=meta["dim"], index_type=meta["index_type"])
        shard.index = faiss.read_index(str(path.with_suffix(".faiss")))
        shard.ids = meta["ids"]
        shard.model_version = meta["model_version"]
        return shard


def build_shard_from_parquet(parquet_path: str | Path, out_path: str | Path, embedding_col: str = "embedding", id_col: str = "doc_id", version_col: str = "model_version") -> FaissShard:
    """Batch-builds one shard's FAISS index from the Parquet output of the
    PySpark embedding job. Run once per shard per re-embed/compensate epoch,
    outside the Spark cluster.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    embeddings = np.vstack(df[embedding_col].to_numpy())
    shard = FaissShard(dim=embeddings.shape[1])
    shard.add(embeddings, df[id_col].astype(str).tolist(), str(df[version_col].iloc[0]))
    shard.save(out_path)
    return shard
