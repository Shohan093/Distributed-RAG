#!/usr/bin/env bash
# STEP 1 of the runbook (see ../README.md). Enables every GCP API this
# project touches. Safe to re-run -- enabling an already-enabled API is a
# no-op. Enabling APIs does not itself cost anything; billing only starts
# once you create VMs/clusters in the later steps.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

echo "Verifying billing is enabled on project ${PROJECT_ID}..."
BILLING_ENABLED="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)')"
if [[ "$BILLING_ENABLED" != "True" ]]; then
  echo "ERROR: billing is not enabled on project ${PROJECT_ID}. Link a billing account first:"
  echo "  gcloud billing projects link ${PROJECT_ID} --billing-account=<ACCOUNT_ID>"
  echo "  (find ACCOUNT_ID with: gcloud billing accounts list)"
  exit 1
fi
echo "Billing OK."

echo "Enabling required APIs (this can take 1-2 minutes)..."
gcloud services enable \
  compute.googleapis.com \
  dataproc.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project "$PROJECT_ID"

echo "Done. Enabled APIs:"
gcloud services list --enabled --project "$PROJECT_ID" \
  --filter="config.name:(compute.googleapis.com OR dataproc.googleapis.com OR storage.googleapis.com)"
