#!/usr/bin/env bash

set -euo pipefail

VM_USER="ubuntu"
LOCAL_DIR="$(dirname "$(realpath "$0")")/../"
REMOTE_DIR="~/drift-rag/"

declare -A EPOCH_TO_VM=(
    [1]="worker2"
    [2]="worker3"
    [3]="worker4"
)

for epoch in 1 2 3; do
  vm="${EPOCH_TO_VM[$epoch]}"
  echo "Syncing generation${epoch}/cleaned to $vm..."
  rsync -avz \
    "${LOCAL_DIR}data/generation${epoch}/cleaned/" \
    "${VM_USER}@${vm}:${REMOTE_DIR}data/generation${epoch}/cleaned/"
  echo "Done: $vm"
done

echo "Data deployment complete!"