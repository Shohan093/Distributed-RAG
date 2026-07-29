#!/usr/bin/env bash

EPOCH=$1
MAX_RETRIES=50

for attempt in $(seq 1 $MAX_RETRIES); do
  echo "=== Attempt $attempt for epoch $EPOCH ==="
  ~/.master/bin/python3 -u embeddings/embed_corpus.py --epoch $EPOCH

  if [ -f "../embeddings/generation${EPOCH}/v3/doc_embeddings.npy" ] || \
     [ -f "../embeddings/generation${EPOCH}/v2/doc_embeddings.npy" ] || \
     [ -f "../embeddings/generation${EPOCH}/v1/doc_embeddings.npy" ]; then
    echo "Detected completion. Verifying..."
    break
  fi

  echo "Process ended without completing (likely OOM kill). Waiting 10s, then retrying..."
  sleep 10
done
