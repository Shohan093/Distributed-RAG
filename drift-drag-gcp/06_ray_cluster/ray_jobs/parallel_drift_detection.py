"""Stage 2 (Dual Drift Detector), parallelized across shards with Ray.

Reads each shard's embeddings from GCS, computes the anchor-residual and
data-drift signals with the *same* driftdrag.detect functions used in the
local scaffold, and writes the results into the Drift State Ledger (Redis)
-- exactly the plan in PLAN.md Stage 2/E3, just fanned out with Ray instead
of a Python for-loop so it scales to hundreds of shards.

Precondition: at least one shard under gs://$GCS_BUCKET/embeddings has
Parquet output for two different model_version values (i.e. you've run the
Dataproc embed job twice with --model-version=v0 then --model-version=v1
after a real re-embed/fine-tune step -- see drift-drag/PLAN.md E0/E1). If
only one version exists yet, this reports zero drift for every shard
(there is nothing to compare against) rather than failing.

Run via ../run_ray_job.sh parallel_drift_detection.py (sets GCS_BUCKET /
REDIS_HOST env vars and executes this on the Ray head node).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import ray

GCS_BUCKET = os.environ["GCS_BUCKET"]
REDIS_HOST = os.environ["REDIS_HOST"]
EMBEDDINGS_ROOT = f"gs://{GCS_BUCKET}/embeddings"

DRIFTDRAG_PKG = str(Path.home() / "driftdrag_pkg" / "driftdrag")


def list_shard_versions() -> dict[str, list[str]]:
    """Returns {shard_id: [model_version, ...]} by listing GCS prefixes."""
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
def compute_shard_drift(shard_id: str, v_from: str, v_to: str) -> dict:
    import driftdrag.detect.anchor_residual as anchor_mod
    import driftdrag.detect.data_drift as drift_mod
    import numpy as np
    from driftdrag.ledger.drift_ledger import DriftLedger, connect

    old_df = pd.read_parquet(f"{EMBEDDINGS_ROOT}/shard_id={shard_id}/model_version={v_from}")
    new_df = pd.read_parquet(f"{EMBEDDINGS_ROOT}/shard_id={shard_id}/model_version={v_to}")

    old_embeddings = np.vstack(old_df["embedding"].to_numpy())
    new_embeddings = np.vstack(new_df["embedding"].to_numpy())
    n = min(len(old_embeddings), len(new_embeddings))

    residual = anchor_mod.anchor_residual(new_embeddings[:n], old_embeddings[:n])
    data_drift = drift_mod.mmd2_unbiased(old_embeddings[:n], new_embeddings[:n])

    client = connect(host=REDIS_HOST, port=6379)
    ledger = DriftLedger(client, history_path="/tmp/ledger_history.jsonl")
    ledger.update_drift_scores(f"shard-{shard_id}", v_from, v_to, anchor_residual=residual, data_drift_score=data_drift)

    return {"shard_id": shard_id, "v_from": v_from, "v_to": v_to, "anchor_residual": residual, "data_drift_score": data_drift}


def main() -> None:
    ray.init(address="auto", runtime_env={"py_modules": [DRIFTDRAG_PKG]})

    shard_versions = list_shard_versions()
    tasks = []
    for shard_id, versions in shard_versions.items():
        if len(versions) < 2:
            print(f"shard {shard_id}: only version(s) {versions} present, skipping (need >=2 to compute drift)")
            continue
        v_from, v_to = versions[0], versions[-1]
        tasks.append(compute_shard_drift.remote(shard_id, v_from, v_to))

    if not tasks:
        print("No shard has >=2 model versions yet -- nothing to compute. "
              "Re-run the Dataproc embed job with a different --model-version "
              "to produce a second version, then re-run this job.")
        return

    results = ray.get(tasks)
    for r in sorted(results, key=lambda r: int(r["shard_id"])):
        print(f"shard {r['shard_id']} ({r['v_from']}->{r['v_to']}): "
              f"anchor_residual={r['anchor_residual']:.4f} data_drift={r['data_drift_score']:.4f}")
    print(f"\n{len(results)} shards scored and written to the Drift State Ledger.")


if __name__ == "__main__":
    main()
