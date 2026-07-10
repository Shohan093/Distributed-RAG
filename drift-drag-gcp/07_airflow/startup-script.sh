#!/usr/bin/env bash
# Installs Apache Airflow (standalone mode: webserver+scheduler+SQLite in
# one process, appropriate for this project's scale, not a production HA
# setup) plus the Google provider, which is what lets the DAG create/submit
# to/delete the Dataproc cluster directly via Airflow operators instead of
# shelling out to gcloud.
set -euo pipefail
exec > >(tee /var/log/driftdrag-airflow-setup.log) 2>&1

apt-get update -y
apt-get install -y python3-pip python3-venv apt-transport-https ca-certificates gnupg curl

# gcloud CLI: the DAG shells out to `gcloud compute ssh` to run Ray jobs
# (see dags/drift_drag_pipeline_dag.py) instead of configuring a separate
# Airflow SSH connection -- one less thing to set up by hand. Auth is
# automatic via this VM's attached service account (--scopes=cloud-platform
# in create_airflow_vm.sh), no `gcloud auth login` needed.
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
apt-get update -y
apt-get install -y google-cloud-cli

python3 -m venv /opt/airflow-venv
source /opt/airflow-venv/bin/activate
pip install --quiet --upgrade pip

AIRFLOW_VERSION="2.9.3"
PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install --quiet "apache-airflow==${AIRFLOW_VERSION}" \
  "apache-airflow-providers-google" \
  "apache-airflow-providers-ssh" \
  --constraint "$CONSTRAINT_URL"

export AIRFLOW_HOME=/opt/airflow
mkdir -p "$AIRFLOW_HOME/dags"

cat > /etc/systemd/system/airflow.service <<EOF
[Unit]
Description=Apache Airflow (standalone)
After=network.target

[Service]
Type=simple
User=root
Environment=AIRFLOW_HOME=/opt/airflow
Environment=PATH=/opt/airflow-venv/bin:/usr/bin:/bin
ExecStart=/opt/airflow-venv/bin/airflow standalone
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable airflow
systemctl start airflow

echo "Airflow starting -- first boot takes a few minutes (DB init + admin user creation)."
echo "Find the generated admin password with:"
echo "  sudo cat /opt/airflow/standalone_admin_password.txt"
