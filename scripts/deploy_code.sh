#!/usr/bin/env bash

set -e

VM_USER="ubuntu"
LOCAL_DIR="$(dirname "$(realpath "$0")")/../"
REMOTE_DIR="~/drift-rag/"

# --- Excludes for CODE sync (goes to every VM) ---
CODE_RSYNC_OPTS=(
    -avz

    # Git
    --exclude "/.git/"
    --exclude "/.gitignore"

    # VS Code
    --exclude "/.vscode/"

    # Python virtual environments
    --exclude "/venv/"
    --exclude "/.venv/"
    --exclude "/env/"
    --exclude "/.env/"
    --exclude "/.conda/"

    # Python cache / compiled files
    --exclude "__pycache__/"
    --exclude "*.pyc"
    --exclude "*.pyo"

    # Local project data
    --exclude "data/raw/"
    --exclude "data/generation*/"

    # Generated ML artifacts
    --exclude "/embeddings/"
    --exclude "indexes/"
    --exclude "checkpoints/"
    --exclude "results/"

    # Spark build artifacts
    --exclude "spark/target/"
    --exclude ".metals/"

)

deploy_code () {
    VM=$1
    echo "Deploying code to $VM..."
    rsync "${CODE_RSYNC_OPTS[@]}" "$LOCAL_DIR" "$VM_USER@$VM:$REMOTE_DIR"
    echo "Done: $VM"
}

for VM in worker1 worker2 worker3 worker4 worker5; do
    deploy_code "$VM" &
done
wait

echo "Code deployment complete!"