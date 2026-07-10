"""Streaming source abstraction (Stage 0).

The same downstream job code must run three ways as this project moves
through its lifecycle: a local file-based "stream" for smoke tests with no
Docker/Kafka available, a single docker-compose Kafka broker for one-VM dev,
and a multi-broker Kafka/Redpanda cluster spread across VMs for the real
E5/E6 runs. Switching between them is one config field
(``infra.stream_source``), never a code change -- that is what "shift to
multiple virtual machines later" requires architecturally.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import LongType, StringType, StructField, StructType

DOC_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("text", StringType(), False),
        StructField("timestamp", LongType(), True),
        StructField("source", StringType(), True),
    ]
)


def read_doc_stream(spark: SparkSession, source: str, **kwargs) -> DataFrame:
    if source == "kafka":
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kwargs["bootstrap_servers"])
            .option("subscribe", kwargs["topic"])
            .option("startingOffsets", kwargs.get("starting_offsets", "latest"))
            .load()
        )
        return raw.select(from_json(col("value").cast("string"), DOC_SCHEMA).alias("d")).select("d.*")
    if source == "file":
        return (
            spark.readStream.format("json")
            .schema(DOC_SCHEMA)
            .option("maxFilesPerTrigger", kwargs.get("max_files_per_trigger", 1))
            .load(kwargs["path"])
        )
    raise ValueError(f"unknown stream source: {source!r} (expected 'kafka' or 'file')")
