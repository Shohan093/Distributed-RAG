"""Writes sample documents as newline-delimited JSON files under
data/stream_in/, which Structured Streaming's file source (see
ingest/stream_source.py, infra=local) treats as a stream: one file per
micro-batch. In docker/cluster infra this same role is played by a Kafka
producer publishing to the docs-raw topic instead -- see
scripts/produce_to_kafka.py in DEPLOYMENT.md for the equivalent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SAMPLE_TEXTS = [
    "Large-scale retrieval systems must cope with embedding models that are periodically retrained.",
    "Streaming analytics pipelines ingest documents continuously from Kafka topics into Spark.",
    "Distributed vector indexes are sharded across many machines to scale beyond a single node.",
    "Concept drift in data streams can be detected with kernel two-sample tests such as MMD.",
    "A budgeted scheduler decides which shards to re-embed given a fixed GPU-hour allowance.",
    "Log anomaly detection benefits from retrieval-augmented context drawn from historical incidents.",
    "Query drift compensation avoids re-indexing an entire corpus after a model upgrade.",
    "Apache Spark Structured Streaming processes micro-batches with exactly-once semantics.",
]


def write_batch(out_dir: Path, batch_id: int, n_docs: int, source: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch-{batch_id:04d}.json"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n_docs):
            text = SAMPLE_TEXTS[(batch_id * n_docs + i) % len(SAMPLE_TEXTS)]
            doc = {
                "doc_id": f"doc-{batch_id:04d}-{i:04d}",
                "text": text,
                "timestamp": int(time.time()),
                "source": source,
            }
            fh.write(json.dumps(doc) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/stream_in")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--docs-per-batch", type=int, default=20)
    parser.add_argument("--source", default="smoke-test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for batch_id in range(args.batches):
        path = write_batch(out_dir, batch_id, args.docs_per_batch, args.source)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
