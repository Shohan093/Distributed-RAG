"""End-to-end local smoke test: no Docker, no Kafka, no real Redis.

Exercises the full Stage 0 -> Stage 1 -> Stage 3 path on one machine before
anything is split across VMs:

  1. write sample documents as a file "stream" (scripts/produce_sample_docs)
  2. run the PySpark embedding job in local[*] mode, file source
     (ingest/spark_embed_job) -> Parquet, partitioned by shard/version
  3. build a per-shard FAISS index from that Parquet output (index/faiss_shard)
  4. round-trip a query through it and confirm the nearest neighbor is itself
  5. write drift-ledger bookkeeping for that shard via fakeredis
     (ledger/drift_ledger) and export the ledger history to Parquet

Swapping infra=local for infra=docker/cluster (see configs/infra/) replays
the same five steps against real Kafka/Redis/Spark cluster/MinIO -- this
script's job is to prove the *logic* is right first, cheaply.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from driftdrag.index.faiss_shard import FaissShard  # noqa: E402
from driftdrag.ingest import spark_embed_job  # noqa: E402
from driftdrag.ledger.drift_ledger import DriftLedger, connect  # noqa: E402

STREAM_DIR = REPO_ROOT / "data" / "stream_in"
LAKE_DIR = REPO_ROOT / "data" / "lake"
EMBEDDINGS_PATH = LAKE_DIR / "embeddings"
CHECKPOINT_PATH = LAKE_DIR / "_checkpoints" / "spark_embed_job"
INDEX_PATH = LAKE_DIR / "indexes" / "shard-smoke"
NUM_SHARDS = 4
MODEL_VERSION = "v0"
EMBEDDING_DIM = 32


def reset_workspace() -> None:
    for path in [STREAM_DIR, EMBEDDINGS_PATH, CHECKPOINT_PATH, INDEX_PATH.parent]:
        if path.exists():
            shutil.rmtree(path)


def produce_docs() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import produce_sample_docs

    for batch_id in range(3):
        produce_sample_docs.write_batch(STREAM_DIR, batch_id, n_docs=20, source="smoke-test")


def run_embed_job() -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").appName("drift-drag-smoke-test").config(
        "spark.ui.enabled", "false"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    cfg = spark_embed_job.EmbedJobConfig(
        source="file",
        file_path=str(STREAM_DIR),
        max_files_per_trigger=1,
        output_path=str(EMBEDDINGS_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        num_shards=NUM_SHARDS,
        model_version=MODEL_VERSION,
        backend_name="hashing",
        embedding_dim=EMBEDDING_DIM,
        trigger_once=True,
    )
    query = spark_embed_job.run(spark, cfg)
    query.awaitTermination()
    spark.stop()


def verify_embeddings() -> pd.DataFrame:
    df = pd.read_parquet(EMBEDDINGS_PATH)
    assert len(df) > 0, "embedding job produced no rows"
    assert {"doc_id", "chunk_id", "shard_id", "embedding", "model_version"} <= set(df.columns)
    print(f"embeddings: {len(df)} chunks across {df['shard_id'].nunique()} shards")
    print(df.groupby("shard_id", observed=True).size().rename("chunks").to_string())
    return df


def build_and_query_faiss_shard(df: pd.DataFrame) -> None:
    shard_id = df["shard_id"].value_counts().idxmax()
    sub = df[df["shard_id"] == shard_id].reset_index(drop=True)

    embeddings = np.vstack(sub["embedding"].to_numpy()).astype(np.float32)
    ids = sub["doc_id"].astype(str).tolist()
    version = str(sub["model_version"].iloc[0])

    shard = FaissShard(dim=embeddings.shape[1], index_type="flat")
    shard.add(embeddings, ids, version)
    shard.save(INDEX_PATH)

    reloaded = FaissShard.load(INDEX_PATH)
    probe = 0
    results = reloaded.search(embeddings[probe], k=1)
    assert results[0][0] == ids[probe], f"expected nearest neighbor of itself, got {results}"
    print(f"FAISS shard {shard_id}: {len(ids)} vectors, self-query OK -> {results}")


def exercise_ledger(df: pd.DataFrame, tmp_history_path: Path) -> None:
    client = connect(use_fake=True)
    ledger = DriftLedger(client, history_path=tmp_history_path)

    for shard_id in df["shard_id"].unique():
        ledger.mark_reembedded(f"shard-{shard_id}", "v-1", MODEL_VERSION, MODEL_VERSION)
        ledger.update_drift_scores(f"shard-{shard_id}", "v-1", MODEL_VERSION, anchor_residual=0.0, data_drift_score=0.0)

    records = ledger.all_shards()
    assert len(records) == df["shard_id"].nunique()
    out = ledger.export_history_parquet(LAKE_DIR / "ledger_history.parquet")
    print(f"drift ledger: {len(records)} shard records, history exported to {out}")


def main() -> None:
    print("== 1/5 resetting workspace ==")
    reset_workspace()
    print("== 2/5 producing sample doc stream ==")
    produce_docs()
    print("== 3/5 running PySpark embedding job (local[*], file source) ==")
    run_embed_job()
    print("== 4/5 verifying Parquet output + FAISS shard build/query ==")
    df = verify_embeddings()
    build_and_query_faiss_shard(df)
    print("== 5/5 exercising Drift State Ledger (fakeredis) ==")
    exercise_ledger(df, LAKE_DIR / "ledger_history.jsonl")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
