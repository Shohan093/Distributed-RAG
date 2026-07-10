#!/usr/bin/env bash
# STEP 9c. MUST run after submit_embed_job.sh and BEFORE delete_cluster.sh
# -- copies the embeddings from this cluster's HDFS to the durable GCS
# bucket using `hadoop distcp` (the standard tool for bulk HDFS<->GCS data
# movement on Dataproc, itself a distributed/parallel Spark-less MapReduce
# copy job -- not a single-threaded gsutil cp). Once this succeeds, the
# cluster's HDFS is disposable again.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud dataproc jobs submit hadoop \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --cluster "$DATAPROC_CLUSTER" \
  --class org.apache.hadoop.tools.DistCp \
  -- \
  "hdfs:///driftdrag/embeddings" "gs://${GCS_BUCKET}/embeddings"

echo "Published. Verify with:"
echo "  gcloud storage ls -r gs://${GCS_BUCKET}/embeddings/ | head -20"
