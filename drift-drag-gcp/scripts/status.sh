#!/usr/bin/env bash
# Run this at the start and end of every work session -- see
# COST_BUDGET.md's "one rule that matters". Shows every billable resource
# this project might have created, so nothing is left running by accident.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "=== Compute Engine instances ==="
gcloud compute instances list --project "$PROJECT_ID" \
  --format="table(name,zone.basename(),machineType.basename(),status,scheduling.automaticRestart)"

echo ""
echo "=== Dataproc clusters (should be EMPTY between runs) ==="
gcloud dataproc clusters list --region "$REGION" --project "$PROJECT_ID" 2>&1 || echo "(dataproc API not enabled yet, or no clusters)"

echo ""
echo "=== GCS bucket size (informational -- storage is cheap, not a cost risk) ==="
gcloud storage du -s "gs://${GCS_BUCKET}" 2>&1 || echo "(bucket not created yet)"
