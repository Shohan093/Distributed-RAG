#!/usr/bin/env bash
# STEP 10. Ephemeral like Dataproc/GPU: start when you're about to run a
# Ray job (drift detection, compensation, or feeding the scheduler), stop
# when done -- see COST_BUDGET.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances create "$RAY_HEAD_VM" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$RAY_MACHINE_TYPE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --network="$NETWORK" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup-script.sh" \
  --tags=driftdrag-ray

echo ""
echo "Ray head installing (~1-2 min). Dashboard (after firewall + your IP, see 01_network/):"
RAY_IP_CMD="gcloud compute instances describe ${RAY_HEAD_VM} --zone ${ZONE} --project ${PROJECT_ID} --format='value(networkInterfaces[0].accessConfigs[0].natIP)'"
echo "  http://\$(${RAY_IP_CMD}):8265"
echo ""
echo "To add worker nodes later (scale out the parallel per-shard work):"
echo "  gcloud compute instances create driftdrag-ray-worker-1 --zone ${ZONE} --project ${PROJECT_ID} \\"
echo "    --machine-type ${RAY_MACHINE_TYPE} --image-family=debian-12 --image-project=debian-cloud \\"
echo "    --metadata=ray-head-ip=<RAY_HEAD_INTERNAL_IP> \\"
echo "    --metadata-from-file startup-script=worker-startup-script.sh"
