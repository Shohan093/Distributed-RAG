#!/usr/bin/env bash
# STEP 9b. Submits the PySpark half of Stage 0 (drift-drag/src/driftdrag/
# ingest/spark_embed_job.py -- reused unmodified from the main project,
# not forked/duplicated here, so the two deployment targets never drift
# apart). References the local file directly; `gcloud dataproc jobs submit
# pyspark` auto-uploads local paths to a staging bucket, so this always
# ships the current version of the job.
#
# Writes first to HDFS (fast, cluster-internal) as scratch, matching
# Google's own recommended Dataproc pattern of "HDFS for intermediate,
# GCS for durable output" -- see publish_hdfs_to_gcs.sh for the promotion
# step that must run before delete_cluster.sh, or this run's output is
# lost when the cluster (and its HDFS) is deleted.
#
# Default embedding backend is "hashing" (dependency-free, CPU-fast,
# known drift ground truth -- see drift-drag/PLAN.md E0) so this ingestion
# pipeline stays cheap to run at scale on CPU-only Dataproc workers. Real
# nomic-embed-text-v1.5 embedding happens on the dedicated GPU VM
# (08_gpu_embedding/) instead -- attaching GPUs to Dataproc workers is
# possible but adds real cost for a step this project handles more cheaply
# elsewhere.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
DRIFTDRAG_SRC="$SCRIPT_DIR/../../drift-drag/src/driftdrag/ingest/spark_embed_job.py"

KAFKA_IP="$(gcloud compute instances describe "$KAFKA_VM" --zone "$ZONE" --project "$PROJECT_ID" --format='value(networkInterfaces[0].networkIP)')"

gcloud dataproc jobs submit pyspark \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --cluster "$DATAPROC_CLUSTER" \
  --py-files="gs://${GCS_BUCKET}/code/driftdrag.zip" \
  --properties="spark.jars.packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3" \
  "$DRIFTDRAG_SRC" \
  -- \
  --source=kafka \
  --bootstrap-servers="${KAFKA_IP}:9092" \
  --topic="${KAFKA_TOPIC_DOCS}" \
  --output-path="hdfs:///driftdrag/embeddings" \
  --checkpoint-path="gs://${GCS_BUCKET}/_checkpoints/spark_embed_job" \
  --num-shards="${NUM_SHARDS}" \
  --model-version=v0 \
  --backend-name=hashing \
  --embedding-dim="${EMBEDDING_DIM}"

echo ""
echo "Embeddings staged in HDFS at hdfs:///driftdrag/embeddings on this cluster."
echo "Run publish_hdfs_to_gcs.sh next to promote them to GCS BEFORE deleting the cluster."
