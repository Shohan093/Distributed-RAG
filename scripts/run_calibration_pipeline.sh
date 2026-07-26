#!/usr/bin/env bash
set -e

PY="~/.master/bin/python3"
REMOTE_DIR="~/drift-rag"

declare -A EPOCH_TO_VM=( [1]="worker2" [2]="worker3" [3]="worker4" )
declare -A EPOCH_TO_MODEL=( [1]="v1" [2]="v2" [3]="v3" )

echo "=== Selecting anchors on each shard, in its own embedding space ==="
for epoch in 1 2 3; do
  vm="${EPOCH_TO_VM[$epoch]}"
  model="${EPOCH_TO_MODEL[$epoch]}"
  echo "Selecting anchors for epoch $epoch on $vm..."
  ssh $vm "cd $REMOTE_DIR/python && mkdir -p ../checkpoints/calibration_ids && \
    $PY embeddings/calibration_sampler.py \
      --doc-embeddings ../embeddings/generation${epoch}/${model}/doc_embeddings.npy \
      --doc-ids ../embeddings/generation${epoch}/${model}/doc_ids.json \
      --n-clusters 1333 --n-centroid 1333 --n-random 500 --n-outlier 500 \
      --output ../checkpoints/calibration_ids/epoch${epoch}.json"
done

echo "=== Pulling anchor id files back to this machine, then pushing to worker5 ==="
mkdir -p /tmp/calibration_ids
for epoch in 1 2 3; do
  vm="${EPOCH_TO_VM[$epoch]}"
  scp "${vm}:${REMOTE_DIR}/checkpoints/calibration_ids/epoch${epoch}.json" "/tmp/calibration_ids/"
done
for epoch in 1 2 3; do
  ssh worker5 "mkdir -p ${REMOTE_DIR}/checkpoints/calibration_ids"
  scp "/tmp/calibration_ids/epoch${epoch}.json" "worker5:${REMOTE_DIR}/checkpoints/calibration_ids/"
done

echo "=== Building the combined calibration text set on worker5 ==="
ssh worker5 "cd $REMOTE_DIR/python && $PY embeddings/build_calibration_set.py"

echo "=== Embedding the combined calibration set with all three models on worker5 ==="
ssh worker5 "cd $REMOTE_DIR/python && nohup $PY -u embeddings/embed_calibration.py > embed_calibration_log.txt 2>&1 &"

echo "Calibration pipeline launched. Check progress with: ssh worker5 'tail -f ~/drift-rag/python/embed_calibration_log.txt'"