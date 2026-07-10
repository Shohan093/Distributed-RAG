"""Real-model (re-)embedding, run on the GPU VM only -- this is the one
step in the whole GCP deployment that needs a GPU, matching the validated
local run (drift-drag/scripts/validate_real_pipeline.py) but reading/
writing GCS instead of local disk, and producing a NEW model_version for
existing shards instead of a one-off BEIR validation.

Reuses the same chunked `text` column the Dataproc embed job already wrote
for model_version=SOURCE_VERSION -- no need to re-fetch or re-chunk raw
documents, just re-embed the identical text with a different model/version.
This is exactly the "model upgrade v_from -> v_to" event Stage 0/1 (PLAN.md)
is built around: same content, new embedding space, which is what lets
06_ray_cluster/ray_jobs/parallel_drift_detection.py compute real drift
once this has run.

Usage (see run_embedding_job.sh):
  python embed_shards_gpu.py --num-shards 32 --source-version v0 \
    --target-version v1 --model-name nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

DRIFTDRAG_PKG_PARENT = str(Path.home() / "driftdrag_pkg")
sys.path.insert(0, DRIFTDRAG_PKG_PARENT)

from driftdrag.ingest.embeddings import SentenceTransformerBackend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET"))
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--model-name", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    if not args.gcs_bucket:
        raise SystemExit("set --gcs-bucket or the GCS_BUCKET env var")

    backend = SentenceTransformerBackend(
        checkpoints={args.target_version: args.model_name},
        dim=args.dim,
        trust_remote_code=True,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    root = f"gs://{args.gcs_bucket}/embeddings"
    processed = 0
    for shard_id in range(args.num_shards):
        src_path = f"{root}/shard_id={shard_id}/model_version={args.source_version}"
        try:
            df = pd.read_parquet(src_path)
        except FileNotFoundError:
            continue

        embeddings = backend.embed_documents(df["text"].tolist(), args.target_version)
        out = df.copy()
        out["embedding"] = list(embeddings)
        out["model_version"] = args.target_version

        dst_path = f"{root}/shard_id={shard_id}/model_version={args.target_version}"
        out.to_parquet(f"{dst_path}/part-0.parquet", index=False)
        print(f"shard {shard_id}: re-embedded {len(out)} chunks -> {dst_path}")
        processed += 1

    print(f"\nDone. {processed}/{args.num_shards} shards re-embedded as {args.target_version}.")
    print("Run 06_ray_cluster's parallel_drift_detection.py / parallel_compensation.py next --")
    print(f"they activate automatically now that shards have both {args.source_version} and {args.target_version}.")


if __name__ == "__main__":
    main()
