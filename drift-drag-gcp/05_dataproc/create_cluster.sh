#!/usr/bin/env bash
# STEP 8 (after Kafka/Redis are up and build_and_upload_package.sh has run
# once). Creates the EPHEMERAL Dataproc cluster -- Spark + YARN + HDFS come
# bundled in the image, this is where "Apache Spark" and "HDFS" both
# actually run. This cluster should exist only for the duration of a job
# run: create -> submit_shard_partitioner.sh -> submit_embed_job.sh ->
# publish_hdfs_to_gcs.sh -> delete_cluster.sh. See COST_BUDGET.md -- this
# is the single biggest cost lever in the whole project.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

# Cheapest possible option for a quick smoke test (single VM, no distributed
# story -- fine for validating the job logic, not for the E5/E6 distributed
# experiments themselves): replace the --num-workers/--worker-* lines below
# with a single `--single-node` flag.

gcloud dataproc clusters create "$DATAPROC_CLUSTER" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image-version "$DATAPROC_IMAGE_VERSION" \
  --master-machine-type "$DATAPROC_MASTER_MACHINE" \
  --master-boot-disk-size=50GB \
  --num-workers "$DATAPROC_NUM_WORKERS" \
  --worker-machine-type "$DATAPROC_WORKER_MACHINE" \
  --worker-boot-disk-size=50GB \
  --initialization-actions="gs://${GCS_BUCKET}/code/init_actions.sh" \
  --properties="spark:spark.jars.packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3" \
  --network="$NETWORK"

echo ""
echo "Cluster created. HDFS is available cluster-internally at hdfs:///..."
echo "Verify with:"
echo "  gcloud dataproc clusters describe ${DATAPROC_CLUSTER} --region ${REGION} --project ${PROJECT_ID}"
echo ""
echo "REMINDER: this cluster costs money every hour it exists (see COST_BUDGET.md)."
echo "Run delete_cluster.sh as soon as your job(s) finish."
