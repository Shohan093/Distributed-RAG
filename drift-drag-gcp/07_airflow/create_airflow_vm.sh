#!/usr/bin/env bash
# STEP 11. --scopes=cloud-platform lets Airflow's Google provider operators
# (Dataproc create/submit/delete) authenticate as this VM's default service
# account via Application Default Credentials, no key file needed. Fine for
# a trial/dev project; a production setup would use a dedicated,
# narrowly-scoped service account instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances create "$AIRFLOW_VM" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$SMALL_MACHINE_TYPE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --scopes=cloud-platform \
  --network="$NETWORK" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup-script.sh" \
  --tags=driftdrag-airflow

echo ""
echo "VM created. Airflow takes ~2-5 min to finish initializing on first boot."
echo "Get the admin password:"
echo "  gcloud compute ssh ${AIRFLOW_VM} --zone ${ZONE} --project ${PROJECT_ID} --command='sudo cat /opt/airflow/standalone_admin_password.txt'"
echo ""
echo "Open the UI (after 01_network/create_firewall_rules.sh has run):"
echo "  http://\$(gcloud compute instances describe ${AIRFLOW_VM} --zone ${ZONE} --project ${PROJECT_ID} --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):8080"
echo "  username: admin, password: (from the command above)"
