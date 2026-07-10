#!/usr/bin/env bash
# STEP 9d. Run this as soon as submit_*.sh + publish_hdfs_to_gcs.sh are
# done. Deleting the cluster destroys its HDFS -- anything not yet
# published to GCS is gone after this runs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

read -p "Confirm embeddings were published to GCS (publish_hdfs_to_gcs.sh ran successfully)? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  echo "Aborting -- run publish_hdfs_to_gcs.sh first."
  exit 1
fi

gcloud dataproc clusters delete "$DATAPROC_CLUSTER" --region "$REGION" --project "$PROJECT_ID" --quiet
echo "Cluster deleted. Dataproc no longer billing."
