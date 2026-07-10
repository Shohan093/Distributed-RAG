#!/usr/bin/env bash
# Dataproc initialization action: runs on every node (master + all workers)
# during cluster creation, BEFORE any job is submitted. Installs the
# driftdrag package's pure-Python dependencies into the cluster's system
# Python so `--py-files driftdrag.zip` (see submit_embed_job.sh) only needs
# to ship our own code, not third-party wheels. Must live in GCS -- Dataproc
# initialization actions cannot be a local path (see build_and_upload_package.sh,
# which uploads this file alongside the driftdrag.zip/jar).
set -euo pipefail
exec > >(tee -a /var/log/driftdrag-init-action.log) 2>&1

/opt/conda/bin/pip install --quiet \
  numpy scipy scikit-learn pandas pyarrow redis faiss-cpu

echo "driftdrag dependencies installed."
