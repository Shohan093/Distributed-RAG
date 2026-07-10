#!/usr/bin/env bash
# Copies the (unmodified) driftdrag Python package and this directory's
# ray_jobs/ scripts onto the Ray head node's home directory, so
# ray.init(runtime_env={"py_modules": [...]}) in each job script can find
# it locally. Re-run this any time drift-drag/src/driftdrag changes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

gcloud compute scp --recurse \
  "$SCRIPT_DIR/../../drift-drag/src/driftdrag" \
  "${RAY_HEAD_VM}:~/driftdrag_pkg/" \
  --zone "$ZONE" --project "$PROJECT_ID"

gcloud compute scp --recurse \
  "$SCRIPT_DIR/ray_jobs" \
  "${RAY_HEAD_VM}:~/driftdrag_pkg/" \
  --zone "$ZONE" --project "$PROJECT_ID"

echo "Synced. Run a job with:"
echo "  bash run_ray_job.sh parallel_drift_detection.py"
