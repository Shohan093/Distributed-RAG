#!/usr/bin/env bash
# Installs Python 3 + Ray + driftdrag's dependencies and starts a Ray head
# node. Ray is used here for the embarrassingly-parallel per-shard work in
# Stages 2-4 (anchor-residual + MMD drift scores, per-cluster compensation
# vectors, per-shard features for the scheduler's quality-drop predictor)
# -- Spark/Dataproc already owns bulk data-lake-scale ingestion, so Ray's
# job is fine-grained Python-native parallelism across shards, not another
# copy of what Spark does.
set -euo pipefail
exec > >(tee /var/log/driftdrag-ray-setup.log) 2>&1

INTERNAL_IP="$(curl -s -H 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/ip')"

apt-get update -y
apt-get install -y python3-pip python3-venv

python3 -m venv /opt/driftdrag-venv
source /opt/driftdrag-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet "ray[default]" numpy scipy scikit-learn pandas pyarrow redis gcsfs faiss-cpu

cat > /etc/systemd/system/ray-head.service <<EOF
[Unit]
Description=Ray head node
After=network.target

[Service]
Type=forking
User=root
Environment=PATH=/opt/driftdrag-venv/bin:/usr/bin:/bin
ExecStart=/opt/driftdrag-venv/bin/ray start --head --port=6380 --dashboard-host=0.0.0.0 --dashboard-port=8265 --node-ip-address=${INTERNAL_IP}
ExecStop=/opt/driftdrag-venv/bin/ray stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
# NOTE: Ray's GCS port defaults to 6379, same default port as Redis. We use
# --port=6380 here specifically so a Ray head and the Redis ledger VM could
# even share a machine without conflict, and so there's no ambiguity in
# firewall rules/logs between "Redis 6379" and "Ray 6379".

systemctl daemon-reload
systemctl enable ray-head
systemctl start ray-head

echo "Ray head started at ${INTERNAL_IP}:6380, dashboard on :8265"
