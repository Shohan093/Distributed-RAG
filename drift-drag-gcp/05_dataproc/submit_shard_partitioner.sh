#!/usr/bin/env bash
# STEP 9a. Submits the Scala/Spark half of Stage 0 (see
# drift-drag/scala/shard-partitioner) to the Dataproc cluster. Writes
# straight to GCS (shard assignment metadata is tiny -- no need to stage
# through HDFS first, unlike the embeddings below) and touches the Redis
# ledger VM directly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KAFKA_IP="$(gcloud compute instances describe "$KAFKA_VM" --zone "$ZONE" --project "$PROJECT_ID" --format='value(networkInterfaces[0].networkIP)')"
REDIS_IP="$(gcloud compute instances describe "$REDIS_VM" --zone "$ZONE" --project "$PROJECT_ID" --format='value(networkInterfaces[0].networkIP)')"

gcloud dataproc jobs submit spark \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --cluster "$DATAPROC_CLUSTER" \
  --class com.driftdrag.ingest.ShardPartitionerJob \
  --jars "gs://${GCS_BUCKET}/code/shard-partitioner-assembly-0.1.0.jar" \
  --properties="spark.jars.packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3" \
  -- \
  --source=kafka \
  --bootstrap-servers="${KAFKA_IP}:9092" \
  --topic="${KAFKA_TOPIC_DOCS}" \
  --output-path="gs://${GCS_BUCKET}/shard_assignments" \
  --checkpoint-path="gs://${GCS_BUCKET}/_checkpoints/shard_partitioner" \
  --num-shards="${NUM_SHARDS}" \
  --model-version=v0 \
  --redis-host="${REDIS_IP}" \
  --redis-port=6379 \
  --trigger-once=true
