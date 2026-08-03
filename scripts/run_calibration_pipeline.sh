#!/usr/bin/env bash

set -euo pipefail

PY="$HOME/.master/bin/python3"
REMOTE_DIR="$HOME/drift-rag"

declare -A EPOCH_TO_VM=(
    [1]="worker2"
    [2]="worker3"
    [3]="worker4"
)

declare -A EPOCH_TO_MODEL=(
    [1]="v1"
    [2]="v2"
    [3]="v3"
)

echo "=================================================="
echo " Step 1: Selecting calibration anchors"
echo "=================================================="

for epoch in 1 2 3; do
    vm="${EPOCH_TO_VM[$epoch]}"
    model="${EPOCH_TO_MODEL[$epoch]}"

    echo "Epoch ${epoch} (${model}) -> ${vm}"

    ssh "$vm" "
        set -e
        cd \"$REMOTE_DIR/python\"

        mkdir -p \"$REMOTE_DIR/checkpoints/calibration_ids\"

        \"$PY\" embeddings/calibration_sampler.py \
            --doc-embeddings \"$REMOTE_DIR/embeddings/generation${epoch}/${model}/doc_embeddings.npy\" \
            --doc-ids \"$REMOTE_DIR/embeddings/generation${epoch}/${model}/doc_ids.json\" \
            --n-clusters 1333 \
            --n-centroid 1333 \
            --n-random 500 \
            --n-outlier 500 \
            --output \"$REMOTE_DIR/checkpoints/calibration_ids/epoch${epoch}.json\"
    "
done

echo
echo "=================================================="
echo " Step 2: Collecting anchor IDs onto worker5"
echo "=================================================="

mkdir -p "$REMOTE_DIR/checkpoints/calibration_ids"

for epoch in 1 2 3; do
    vm="${EPOCH_TO_VM[$epoch]}"

    echo "Copying epoch${epoch}.json from ${vm}"

    scp \
        "${vm}:${REMOTE_DIR}/checkpoints/calibration_ids/epoch${epoch}.json" \
        "$REMOTE_DIR/checkpoints/calibration_ids/"
done

echo
echo "=================================================="
echo " Step 3: Building combined calibration dataset"
echo "=================================================="

cd "$REMOTE_DIR/python"

"$PY" embeddings/build_calibration_set.py

echo
echo "=================================================="
echo " Step 4: Embedding calibration dataset"
echo "=================================================="

nohup \
    "$PY" -u embeddings/embed_calibration.py \
    > embed_calibration_log.txt 2>&1 &

echo
echo "Pipeline launched successfully!"
echo
echo "Monitor progress with:"
echo
echo "tail -f $REMOTE_DIR/python/embed_calibration_log.txt"