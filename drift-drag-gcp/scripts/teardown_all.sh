#!/usr/bin/env bash
# The "stop spending money" button. Deletes the Dataproc cluster and GPU VM
# if they exist (nothing worth preserving on either), and STOPS (does not
# delete) Kafka/Redis/Airflow/Ray so their state/disks survive for next
# time. Safe to run even if some resources don't exist yet -- every
# command below tolerates "not found".
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "=== Deleting Dataproc cluster (if any) ==="
gcloud dataproc clusters delete "$DATAPROC_CLUSTER" --region "$REGION" --project "$PROJECT_ID" --quiet 2>&1 || echo "(none found)"

echo "=== Deleting GPU VM (if any) ==="
gcloud compute instances delete "$GPU_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet 2>&1 || echo "(none found)"

echo "=== Deleting Ray head (if any) ==="
gcloud compute instances delete "$RAY_HEAD_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet 2>&1 || echo "(none found)"

for vm in "$KAFKA_VM" "$REDIS_VM" "$AIRFLOW_VM"; do
  echo "=== Stopping ${vm} (if running) ==="
  gcloud compute instances stop "$vm" --zone "$ZONE" --project "$PROJECT_ID" 2>&1 || echo "(not running or doesn't exist)"
done

echo ""
echo "Done. Run scripts/status.sh to confirm nothing billable is left running."
