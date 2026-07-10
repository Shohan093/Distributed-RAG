#!/usr/bin/env bash
# Runs one of ray_jobs/*.py on the Ray head node over SSH, inside the venv
# startup-script.sh created. Usage: bash run_ray_job.sh parallel_drift_detection.py
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

JOB_FILE="${1:?usage: run_ray_job.sh <job_file.py> [extra args...]}"
shift || true

# Resolved HERE (on your machine, which has gcloud) not on the remote VM
# (which doesn't have the gcloud SDK installed) -- then passed as a plain
# env var into the SSH command.
REDIS_IP="$(gcloud compute instances describe "$REDIS_VM" --zone "$ZONE" --project "$PROJECT_ID" --format='value(networkInterfaces[0].networkIP)')"

gcloud compute ssh "$RAY_HEAD_VM" --zone "$ZONE" --project "$PROJECT_ID" --command \
  "cd ~/driftdrag_pkg && GCS_BUCKET=${GCS_BUCKET} REDIS_HOST=${REDIS_IP} /opt/driftdrag-venv/bin/python ray_jobs/${JOB_FILE} $*"
