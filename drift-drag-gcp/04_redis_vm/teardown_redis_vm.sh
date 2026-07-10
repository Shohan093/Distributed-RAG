#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

if [[ "${1:-}" == "--delete" ]]; then
  gcloud compute instances delete "$REDIS_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet
else
  gcloud compute instances stop "$REDIS_VM" --zone "$ZONE" --project "$PROJECT_ID"
  echo "Stopped (not deleted). Restart with:"
  echo "  gcloud compute instances start ${REDIS_VM} --zone ${ZONE} --project ${PROJECT_ID}"
fi
