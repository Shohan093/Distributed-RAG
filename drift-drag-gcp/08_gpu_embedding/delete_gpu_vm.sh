#!/usr/bin/env bash
# Always --delete (never just stop) -- a stopped GPU VM still reserves the
# GPU against your quota=1 in some configurations and buys you nothing
# (there's no state on this VM worth preserving; embeddings already landed
# in GCS). Run this the moment embed_shards_gpu.py finishes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute instances delete "$GPU_VM" --zone "$ZONE" --project "$PROJECT_ID" --quiet
echo "GPU VM deleted. Quota freed."
