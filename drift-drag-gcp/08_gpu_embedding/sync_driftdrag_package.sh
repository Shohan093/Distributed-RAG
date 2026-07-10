#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute scp --recurse \
  "$SCRIPT_DIR/../../drift-drag/src/driftdrag" \
  "${GPU_VM}:~/driftdrag_pkg/" \
  --zone "$ZONE" --project "$PROJECT_ID"

gcloud compute scp \
  "$SCRIPT_DIR/embed_shards_gpu.py" \
  "${GPU_VM}:~/embed_shards_gpu.py" \
  --zone "$ZONE" --project "$PROJECT_ID"

echo "Synced. Run with: bash run_embedding_job.sh"
