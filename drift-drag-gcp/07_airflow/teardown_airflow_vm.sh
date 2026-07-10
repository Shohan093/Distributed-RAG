#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

if [[ "${1:-}" == "--delete" ]]; then
  gcloud compute instances delete "$AIRFLOW_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet
else
  gcloud compute instances stop "$AIRFLOW_VM" --zone "$ZONE" --project "$PROJECT_ID"
  echo "Stopped (not deleted) -- DAG history/state on its local SQLite DB is preserved."
fi
