#!/usr/bin/env bash
# STEP 3. Creates the GCS bucket that plays the role drift-drag/data/lake
# (local) or MinIO (drift-drag docker-compose) played before: the durable
# data lake that survives after an ephemeral Dataproc/GPU VM is deleted.
# HDFS (via Dataproc, see 05_dataproc/) is used for *scratch* data during a
# job run; GCS is where results and the driftdrag wheel/jar land so the
# next job (possibly on a brand new cluster) can read them.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  || echo "Bucket already exists or name is taken globally -- if the latter, edit GCS_BUCKET in ../env.sh and re-run."

echo "Creating the lake's top-level layout (empty marker objects, for clarity in the console)..."
for prefix in code embeddings shard_assignments indexes ledger_history beir_datasets mlflow; do
  echo "" | gcloud storage cp - "gs://${GCS_BUCKET}/${prefix}/.keep"
done

echo "Done. Layout:"
gcloud storage ls "gs://${GCS_BUCKET}/"
