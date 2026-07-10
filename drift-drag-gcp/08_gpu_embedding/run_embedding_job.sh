#!/usr/bin/env bash
# STEP 13. Re-embeds every shard's existing chunks (produced by the
# Dataproc PySpark job with --model-version=v0 --backend-name=hashing)
# with a real model, as a new version -- e.g. v1 -- so real drift exists
# between v0 and v1 for the Ray jobs to measure. Override SOURCE_VERSION/
# TARGET_VERSION/MODEL_NAME as env vars to produce v2, v3... for the full
# 5-transition sequence PLAN.md's E0 describes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

SOURCE_VERSION="${SOURCE_VERSION:-v0}"
TARGET_VERSION="${TARGET_VERSION:-v1}"
MODEL_NAME="${MODEL_NAME:-nomic-ai/nomic-embed-text-v1.5}"

gcloud compute ssh "$GPU_VM" --zone "$ZONE" --project "$PROJECT_ID" --command \
  "GCS_BUCKET=${GCS_BUCKET} /opt/conda/bin/python ~/embed_shards_gpu.py \
    --num-shards ${NUM_SHARDS} --source-version ${SOURCE_VERSION} \
    --target-version ${TARGET_VERSION} --model-name '${MODEL_NAME}'"
