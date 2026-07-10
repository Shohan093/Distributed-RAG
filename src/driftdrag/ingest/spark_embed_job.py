"""PySpark Structured Streaming embedding job (Stage 0, Python half).

Consumes the same doc stream as the Scala shard partitioner
(scala/shard-partitioner) independently, chunks each document, embeds each
chunk with a pluggable backend (embeddings.py), tags every row with the
embedding model version and shard id, and appends the result to the
Parquet data lake, partitioned by (shard_id, model_version). A separate,
non-Spark step (index/faiss_shard.py) later builds the per-shard ANN index
from this Parquet output -- see that module's docstring for why FAISS is
kept out of the executors.

Runs three ways with the same code, switched by --source / config:
  local[*] + file source   -> no Docker needed, used by scripts/smoke_test
  spark://spark-master:7077 + kafka source, single VM -> docker-compose.yml
  spark://spark-master:7077 + kafka source, N VMs     -> + docker-compose.cluster.yml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, LongType, StringType, StructField, StructType

from driftdrag.ingest.chunking import chunk_text
from driftdrag.ingest.embeddings import get_backend
from driftdrag.ingest.sharding import shard_for_doc
from driftdrag.ingest.stream_source import read_doc_stream

RESULT_SCHEMA = StructType(
    [
        StructField("doc_id", StringType()),
        StructField("chunk_id", StringType()),
        StructField("shard_id", IntegerType()),
        StructField("text", StringType()),
        StructField("embedding", ArrayType(FloatType())),
        StructField("model_version", StringType()),
        StructField("timestamp", LongType()),
    ]
)


@dataclass
class EmbedJobConfig:
    source: str = "file"
    file_path: str = "data/stream_in"
    max_files_per_trigger: int = 1
    bootstrap_servers: str = "localhost:9092"
    topic: str = "docs-raw"
    output_path: str = "data/lake/embeddings"
    checkpoint_path: str = "data/lake/_checkpoints/spark_embed_job"
    num_shards: int = 32
    model_version: str = "v0"
    backend_name: str = "hashing"
    embedding_dim: int = 128
    window_chars: int = 800
    overlap_chars: int = 100
    trigger_once: bool = True


def _make_embed_partitions(cfg: EmbedJobConfig):
    def embed_partitions(pdf_iter):
        backend = get_backend(cfg.backend_name, dim=cfg.embedding_dim)
        columns = [f.name for f in RESULT_SCHEMA.fields]
        for pdf in pdf_iter:
            rows = []
            for _, row in pdf.iterrows():
                for i, chunk in enumerate(chunk_text(row["text"], cfg.window_chars, cfg.overlap_chars)):
                    rows.append(
                        {
                            "doc_id": row["doc_id"],
                            "chunk_id": f"{row['doc_id']}::{i}",
                            "shard_id": shard_for_doc(row["doc_id"], cfg.num_shards),
                            "text": chunk,
                            "model_version": cfg.model_version,
                            "timestamp": row["timestamp"],
                        }
                    )
            if not rows:
                yield pd.DataFrame(columns=columns)
                continue
            out = pd.DataFrame(rows)
            embeddings = backend.embed(out["text"].tolist(), cfg.model_version)
            out["embedding"] = [row.tolist() for row in embeddings]
            yield out[columns]

    return embed_partitions


def process_batch(batch_df: DataFrame, cfg: EmbedJobConfig) -> None:
    if not batch_df.take(1):
        return
    embedded = batch_df.mapInPandas(_make_embed_partitions(cfg), schema=RESULT_SCHEMA)
    embedded.write.mode("append").partitionBy("shard_id", "model_version").parquet(cfg.output_path)


def run(spark: SparkSession, cfg: EmbedJobConfig):
    if cfg.source == "file":
        source_kwargs = {"path": cfg.file_path, "max_files_per_trigger": cfg.max_files_per_trigger}
    elif cfg.source == "kafka":
        source_kwargs = {"bootstrap_servers": cfg.bootstrap_servers, "topic": cfg.topic}
    else:
        raise ValueError(f"unknown source: {cfg.source}")

    docs = read_doc_stream(spark, cfg.source, **source_kwargs)

    writer = docs.writeStream.option("checkpointLocation", cfg.checkpoint_path).foreachBatch(
        lambda batch_df, _batch_id: process_batch(batch_df, cfg)
    )

    query = writer.trigger(availableNow=True).start() if cfg.trigger_once else writer.start()
    return query


def parse_args(argv: list[str] | None = None) -> EmbedJobConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="file", choices=["file", "kafka"])
    parser.add_argument("--file-path", default="data/stream_in")
    parser.add_argument("--max-files-per-trigger", type=int, default=1)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="docs-raw")
    parser.add_argument("--output-path", default="data/lake/embeddings")
    parser.add_argument("--checkpoint-path", default="data/lake/_checkpoints/spark_embed_job")
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--model-version", default="v0")
    parser.add_argument("--backend-name", default="hashing")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--trigger-once", action="store_true", default=True)
    args = parser.parse_args(argv)
    cfg = EmbedJobConfig(
        source=args.source,
        file_path=args.file_path,
        max_files_per_trigger=args.max_files_per_trigger,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        num_shards=args.num_shards,
        model_version=args.model_version,
        backend_name=args.backend_name,
        embedding_dim=args.embedding_dim,
        trigger_once=args.trigger_once,
    )
    return cfg, args.master


def main(argv: list[str] | None = None) -> None:
    cfg, master = parse_args(argv)
    spark = SparkSession.builder.master(master).appName("drift-drag-embed-job").getOrCreate()
    query = run(spark, cfg)
    query.awaitTermination()


if __name__ == "__main__":
    main()
