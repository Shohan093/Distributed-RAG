#!/usr/bin/env bash
# Stops (default) or deletes (--delete) the Kafka VM. Stopping preserves
# the boot disk (and Kafka's on-disk log data) for ~$1/month; deleting
# removes it entirely -- use --delete only if you're done with this
# project or want a totally clean restart.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

if [[ "${1:-}" == "--delete" ]]; then
  gcloud compute instances delete "$KAFKA_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet
else
  gcloud compute instances stop "$KAFKA_VM" --zone "$ZONE" --project "$PROJECT_ID"
  echo "Stopped (not deleted). Restart with:"
  echo "  gcloud compute instances start ${KAFKA_VM} --zone ${ZONE} --project ${PROJECT_ID}"
fi
