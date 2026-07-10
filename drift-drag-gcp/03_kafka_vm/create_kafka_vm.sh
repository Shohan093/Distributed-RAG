#!/usr/bin/env bash
# STEP 4. Creates the Kafka VM and lets the startup script (previous file)
# install/configure/start Kafka automatically. Takes 2-3 minutes after the
# instance is RUNNING for Kafka itself to finish installing -- watch
# progress with the journalctl command printed at the end.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances create "$KAFKA_VM" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$SMALL_MACHINE_TYPE" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard \
  --network="$NETWORK" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup-script.sh" \
  --tags=driftdrag-kafka

echo ""
echo "VM created. Kafka is installing in the background (~2-3 min)."
echo "Watch progress:"
echo "  gcloud compute ssh ${KAFKA_VM} --zone ${ZONE} --project ${PROJECT_ID} --command='sudo tail -f /var/log/driftdrag-kafka-setup.log'"
echo ""
echo "Get its internal IP (needed by every other component):"
echo "  gcloud compute instances describe ${KAFKA_VM} --zone ${ZONE} --project ${PROJECT_ID} --format='value(networkInterfaces[0].networkIP)'"
