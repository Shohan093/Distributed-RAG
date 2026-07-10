#!/usr/bin/env bash
# STEP 2 of the runbook. Uses the project's auto-created "default" VPC
# (every new GCP project gets one) instead of a custom VPC -- one less
# thing to provision/tear down for a trial-budget project. The default
# network already ships with a "default-allow-internal" rule permitting
# all traffic between instances in the same network/region, which is all
# Kafka<->Dataproc<->Ray<->Redis<->Airflow need (see COST_BUDGET.md -- this
# script does not add anything for that). All this script adds is external
# access, from YOUR machine only, to the two web UIs (Airflow, Ray
# dashboard) -- never opened to 0.0.0.0/0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

MY_IP="$(curl -s ifconfig.me)"
if [[ -z "$MY_IP" ]]; then
  echo "Could not auto-detect your public IP. Find it manually (e.g. https://ifconfig.me)"
  echo "and re-run: MY_IP=<your.ip.here> bash create_firewall_rules.sh"
  exit 1
fi
echo "Detected your public IP as ${MY_IP} -- opening the admin UIs to this address only."

gcloud compute firewall-rules create driftdrag-allow-admin-ui \
  --project "$PROJECT_ID" \
  --network "$NETWORK" \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:8080,tcp:8265 \
  --source-ranges="${MY_IP}/32" \
  --description="Airflow webserver (8080) and Ray dashboard (8265), your IP only" \
  || echo "Rule already exists, skipping (re-run 02b_update_my_ip.sh below if your IP changed)."

echo "Done. Verify with:"
echo "  gcloud compute firewall-rules list --project ${PROJECT_ID}"
echo ""
echo "NOTE: if your public IP changes (e.g. you switch networks), re-run this script --"
echo "it will fail with ALREADY_EXISTS; delete the old rule first:"
echo "  gcloud compute firewall-rules delete driftdrag-allow-admin-ui --project ${PROJECT_ID}"
