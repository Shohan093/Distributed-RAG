#!/usr/bin/env bash
# STEP 12 (only run this when you're about to actively embed -- see
# COST_BUDGET.md, this is the single most expensive resource in the
# project). --maintenance-policy=TERMINATE is required for GPU-attached
# instances (they can't live-migrate). Uses your project's T4 quota=1 --
# if a previous GPU VM wasn't deleted, this will fail with a quota error.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances create "$GPU_VM" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$GPU_MACHINE_TYPE" \
  --accelerator="$GPU_ACCELERATOR" \
  --maintenance-policy=TERMINATE \
  --image-family="$GPU_IMAGE_FAMILY" \
  --image-project="$GPU_IMAGE_PROJECT" \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --network="$NETWORK" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup-script.sh" \
  --tags=driftdrag-gpu

echo ""
echo "VM created. First boot installs a few extra pip packages (~1-2 min)."
echo "Next: bash sync_driftdrag_package.sh && bash run_embedding_job.sh"
echo ""
echo "COST REMINDER: run delete_gpu_vm.sh the moment you're done -- see COST_BUDGET.md."
