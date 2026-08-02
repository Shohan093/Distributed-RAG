#!/usr/bin/env bash
set -e

declare -A EPOCH_TO_VM=( [1]="worker2" [2]="worker3" [3]="worker4" )
declare -A EPOCH_TO_MODEL=( [1]="v1" [2]="v2" [3]="v3" )

echo "Watching for all three shard embeddings to finish..."

while true; do
  all_done=true
  for epoch in 1 2 3; do
    vm="${EPOCH_TO_VM[$epoch]}"
    model="${EPOCH_TO_MODEL[$epoch]}"
    if ! ssh $vm "test -f ~/drift-rag/embeddings/generation${epoch}/${model}/doc_embeddings.npy"; then
      all_done=false
    fi
  done

  if $all_done; then
    echo "All three epochs finished. Firing calibration pipeline..."
    ./scripts/run_calibration_pipeline.sh
    break
  fi

  echo "$(date +%H:%M:%S) — not all finished yet, checking again in 5 minutes..."
  sleep 300
done