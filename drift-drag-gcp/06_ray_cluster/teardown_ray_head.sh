#!/usr/bin/env bash
# Ray has no state worth preserving between sessions (unlike Kafka/Redis) --
# default to --delete, not stop, to save the ~$0.15/hr machine cost entirely
# rather than just pausing it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances delete "$RAY_HEAD_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet
echo "Ray head deleted. Re-create any time with create_ray_head.sh + sync_driftdrag_package.sh."
