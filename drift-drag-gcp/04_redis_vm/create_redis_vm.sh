#!/usr/bin/env bash
# STEP 5.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances create "$REDIS_VM" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$SMALL_MACHINE_TYPE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-standard \
  --network="$NETWORK" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup-script.sh" \
  --tags=driftdrag-redis

echo ""
echo "VM created. Redis installs in ~30-60 seconds."
echo "Get its internal IP (needed by every other component -- this is REDIS_HOST):"
echo "  gcloud compute instances describe ${REDIS_VM} --zone ${ZONE} --project ${PROJECT_ID} --format='value(networkInterfaces[0].networkIP)'"
