#!/usr/bin/env bash
# Copies the DAG file into Airflow's dags/ folder on the VM. Airflow's
# scheduler polls that folder (default every ~30s) and will pick up
# drift_drag_pipeline automatically -- no restart needed. Re-run any time
# the DAG file changes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute scp "$SCRIPT_DIR/dags/drift_drag_pipeline_dag.py" \
  "${AIRFLOW_VM}:/tmp/drift_drag_pipeline_dag.py" \
  --zone "$ZONE" --project "$PROJECT_ID"

gcloud compute ssh "$AIRFLOW_VM" --zone "$ZONE" --project "$PROJECT_ID" --command \
  "sudo mv /tmp/drift_drag_pipeline_dag.py /opt/airflow/dags/drift_drag_pipeline_dag.py"

echo "Synced. It should appear in the Airflow UI's DAG list within ~30 seconds."
