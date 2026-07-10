"""Stage 3 (query-time compensation), parallelized across shards with Ray.

For every shard with >=2 model versions in GCS, computes both the global
and per-cluster drift vectors (driftdrag.compensate.drift_vectors) and
writes them to the Drift State Ledger, so the query router can apply them
without re-embedding the shard. Same precondition as
parallel_drift_detection.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import ray

GCS_BUCKET = os.environ["GCS_BUCKET"]
REDIS_HOST = os.environ["REDIS_HOST"]
EMBEDDINGS_ROOT = f"gs://{GCS_BUCKET}/embeddings"
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", "8"))

DRIFTDRAG_PKG = str(Path.home() / "driftdrag_pkg" / "driftdrag")


def list_shard_versions() -> dict[str, list[str]]:
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    shard_versions: dict[str, list[str]] = {}
    for shard_dir in fs.ls(EMBEDDINGS_ROOT):
        m = re.search(r"shard_id=(\d+)", shard_dir)
        if not m:
            continue
        shard_id = m.group(1)
        versions = [re.search(r"model_version=(\w+)", d).group(1) for d in fs.ls(shard_dir) if "model_version=" in d]
        shard_versions[shard_id] = sorted(versions)
    return shard_versions


@ray.remote
def compute_shard_compensation(shard_id: str, v_from: str, v_to: str) -> dict:
    from driftdrag.compensate.drift_vectors import global_drift_vector, per_cluster_drift_vectors
    from driftdrag.ledger.drift_ledger import DriftLedger, connect

    old_df = pd.read_parquet(f"{EMBEDDINGS_ROOT}/shard_id={shard_id}/model_version={v_from}")
    new_df = pd.read_parquet(f"{EMBEDDINGS_ROOT}/shard_id={shard_id}/model_version={v_to}")
    old_embeddings = np.vstack(old_df["embedding"].to_numpy())
    new_embeddings = np.vstack(new_df["embedding"].to_numpy())
    n = min(len(old_embeddings), len(new_embeddings))
    old_embeddings, new_embeddings = old_embeddings[:n], new_embeddings[:n]

    global_vec = global_drift_vector(old_embeddings, new_embeddings)
    _, cluster_vecs = per_cluster_drift_vectors(old_embeddings, new_embeddings, n_clusters=min(N_CLUSTERS, n))

    client = connect(host=REDIS_HOST, port=6379)
    ledger = DriftLedger(client, history_path="/tmp/ledger_history.jsonl")
    ledger.update_drift_vector(f"shard-{shard_id}", v_from, v_to, global_vec.tolist())
    for cluster_id, vec in cluster_vecs.items():
        ledger.update_drift_vector(f"shard-{shard_id}", v_from, v_to, vec.tolist(), cluster_id=str(cluster_id))

    return {"shard_id": shard_id, "v_from": v_from, "v_to": v_to, "n_clusters": len(cluster_vecs)}


def main() -> None:
    ray.init(address="auto", runtime_env={"py_modules": [DRIFTDRAG_PKG]})

    shard_versions = list_shard_versions()
    tasks = [
        compute_shard_compensation.remote(shard_id, versions[0], versions[-1])
        for shard_id, versions in shard_versions.items()
        if len(versions) >= 2
    ]
    if not tasks:
        print("No shard has >=2 model versions yet -- nothing to compensate for.")
        return

    results = ray.get(tasks)
    for r in sorted(results, key=lambda r: int(r["shard_id"])):
        print(f"shard {r['shard_id']}: global + {r['n_clusters']}-cluster drift vectors written to ledger")
    print(f"\n{len(results)} shards compensated.")


if __name__ == "__main__":
    main()
