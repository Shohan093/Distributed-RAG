# Drift-DRAG on GCP — step-by-step runbook

This directory is a **separate, GCP-specific deployment** of the same project in `../drift-drag/` (the local/Docker-Compose/Ansible scaffold — see its `PLAN.md`, `README.md`, `DEPLOYMENT.md`). Nothing in `../drift-drag/` is duplicated here except where GCP genuinely needs a different artifact (job-submission scripts, VM startup scripts) — the actual pipeline logic (`driftdrag` Python package, the Scala shard-partitioner) is referenced directly from `../drift-drag/`, never copied, so the two deployments never drift apart.

**Read [COST_BUDGET.md](COST_BUDGET.md) before starting anything.** You have $300 in trial credit; this runbook is written so the persistent pieces cost cents/hour and the expensive pieces (Dataproc, GPU) exist only for the duration of a single job.

## Architecture: which big-data framework does what

| Framework | Role here | Where |
|---|---|---|
| **Kafka** | Streaming ingestion bus (docs-raw, logs-raw, shard-assignments topics) | `03_kafka_vm/` — one persistent VM, KRaft mode |
| **Apache Spark** | Stage 0 ingestion: Scala shard-partitioner + PySpark embedding job, both reused unmodified from `../drift-drag/` | `05_dataproc/` — ephemeral Dataproc cluster |
| **HDFS** | Fast intra-cluster scratch for the embedding job's output during a Dataproc cluster's lifetime; promoted to GCS before the cluster is deleted | `05_dataproc/publish_hdfs_to_gcs.sh` (uses `hadoop distcp`) |
| **Ray** | Parallel, per-shard Stage 2-4 work (anchor-residual + MMD drift detection, per-cluster compensation vectors, scheduler feature computation) across potentially hundreds of shards | `06_ray_cluster/` — one VM (add workers to scale out) |
| **Airflow** | Orchestrates one full pipeline run (create cluster -> submit jobs -> publish -> delete cluster -> Ray stages) as a single DAG, with a guaranteed cluster-delete step even on failure | `07_airflow/` — one persistent VM |
| GCS | Durable data lake (embeddings, shard assignments, ledger history, code artifacts) — the GCP equivalent of `../drift-drag/data/lake` or its MinIO service | `02_storage/` |
| GPU VM (T4) | The one step that needs a GPU: real `nomic-embed-text-v1.5` (re-)embedding, ephemeral | `08_gpu_embedding/` |
| Redis | Drift State Ledger (same `driftdrag.ledger.drift_ledger` code as local) | `04_redis_vm/` |

## Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`), with a project that has billing enabled — this runbook assumes project **`drift-rag`**, region **`asia-south1`** (edit `env.sh` if yours differ).
- The `../drift-drag/` directory present as a sibling (several scripts here reference it directly, e.g. the Scala jar and the PySpark job file).
- `zip`, `curl` available locally (used by `scripts/build_and_upload_package.sh` and the firewall script).

Every script in this directory does `source ".../env.sh"` first — **edit `env.sh` once** (project ID, region, machine sizes) rather than any individual script.

---

## Step-by-step

### Step 1 — Enable APIs and confirm billing
```bash
bash 00_prereqs/enable_apis.sh
```
Enables Compute Engine, Dataproc, Cloud Storage, IAM. Fails loudly with instructions if billing isn't linked yet. Free — enabling an API costs nothing by itself.

### Step 2 — Firewall rules for the admin UIs
```bash
bash 01_network/create_firewall_rules.sh
```
Auto-detects your public IP and opens Airflow's webserver (8080) and Ray's dashboard (8265) to **that IP only** — never to the whole internet. Re-run this if your IP changes (e.g. different network/VPN).

### Step 3 — Create the GCS data lake bucket
```bash
bash 02_storage/create_gcs_bucket.sh
```
If the bucket name is taken (GCS names are global across all of GCP, not just your project), edit `GCS_BUCKET` in `env.sh` and re-run.

### Step 4 — Kafka VM
```bash
bash 03_kafka_vm/create_kafka_vm.sh
```
Watch it finish installing (~2-3 min):
```bash
gcloud compute ssh driftdrag-kafka --zone asia-south1-a --project drift-rag \
  --command='sudo tail -f /var/log/driftdrag-kafka-setup.log'
```
Wait for `Kafka setup complete.`. Note the internal IP it prints — you'll need it again, but every later script re-fetches it automatically via `gcloud compute instances describe`, so you don't need to copy it down yourself.

### Step 5 — Redis VM (Drift State Ledger)
```bash
bash 04_redis_vm/create_redis_vm.sh
```
Installs in under a minute (much simpler than Kafka — a straight `apt install redis-server` plus a config edit).

### Step 6 — Sanity-check Kafka + Redis before spending on anything bigger
```bash
KAFKA_IP=$(gcloud compute instances describe driftdrag-kafka --zone asia-south1-a --project drift-rag --format='value(networkInterfaces[0].networkIP)')
gcloud compute ssh driftdrag-kafka --zone asia-south1-a --project drift-rag \
  --command="/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list"
# expect: docs-raw, logs-raw, shard-assignments

REDIS_IP=$(gcloud compute instances describe driftdrag-redis --zone asia-south1-a --project drift-rag --format='value(networkInterfaces[0].networkIP)')
gcloud compute ssh driftdrag-redis --zone asia-south1-a --project drift-rag --command="redis-cli ping"
# expect: PONG
```

### Step 7 — Build and upload the driftdrag package + Scala jar
```bash
bash scripts/build_and_upload_package.sh
```
This is the step that ties `../drift-drag/` into this deployment: zips the Python package, builds (or reuses) the Scala fat jar via the same `sbt`/JDK already set up under `drift-drag/.tools/` from local development, and uploads everything to `gs://<bucket>/code/`. **Re-run this any time you change driftdrag's Python code or the Scala job.**

### Step 8 — Produce some sample documents into Kafka
You need *something* in the `docs-raw` topic before Stage 0 has anything to ingest. Simplest option, from your own machine:
```bash
pip install kafka-python
python - <<'PY'
import json, time
from kafka import KafkaProducer
KAFKA_IP = "REPLACE_WITH_OUTPUT_OF: gcloud compute instances describe driftdrag-kafka --zone asia-south1-a --project drift-rag --format='value(networkInterfaces[0].networkIP)'"
# Kafka's advertised.listener is its INTERNAL ip -- you can only reach it from
# inside the VPC. Easiest path: SSH-tunnel or just run this producer via
# `gcloud compute ssh driftdrag-kafka --command="python3 ..."` instead, or
# open a temporary firewall rule for 9092 from your IP if you'd rather run
# it locally. The tunnel approach (no extra firewall rule) is recommended:
#   gcloud compute ssh driftdrag-kafka --zone asia-south1-a --project drift-rag -- -L 9092:localhost:9092
# then point this script at bootstrap_servers="localhost:9092" instead.
PY
```
(See "Troubleshooting -> reaching Kafka from your laptop" below for the exact tunnel command.)

### Step 9 — Run Stage 0 on Dataproc (ephemeral — cost concentrates here)
```bash
bash 05_dataproc/create_cluster.sh          # ~2 min to provision
bash 05_dataproc/submit_shard_partitioner.sh   # Scala job
bash 05_dataproc/submit_embed_job.sh           # PySpark job, writes to HDFS
bash 05_dataproc/publish_hdfs_to_gcs.sh        # HDFS -> GCS, MUST run before deleting
bash 05_dataproc/delete_cluster.sh             # asks for confirmation
```
Verify:
```bash
gcloud storage ls gs://<bucket>/embeddings/
gcloud storage ls gs://<bucket>/shard_assignments/
```
You should see `shard_id=0/model_version=v0/`, `shard_id=1/model_version=v0/`, ... partitions.

### Step 10 — Ray: drift detection + compensation (needs >=2 model versions to do anything real)
At this point only `model_version=v0` exists (produced by the cheap `hashing` backend on Dataproc), so drift detection has nothing to compare yet — that's expected; skip to Step 11 first if you want real numbers, or run this now to confirm the plumbing works (it will correctly report "nothing to compute yet" for every shard).
```bash
bash 06_ray_cluster/create_ray_head.sh
bash 06_ray_cluster/sync_driftdrag_package.sh
bash 06_ray_cluster/run_ray_job.sh parallel_drift_detection.py
bash 06_ray_cluster/run_ray_job.sh parallel_compensation.py
bash 06_ray_cluster/run_ray_job.sh run_scheduler.py
```

### Step 11 — GPU VM: produce a real second model version (v1)
```bash
bash 08_gpu_embedding/create_gpu_vm.sh
bash 08_gpu_embedding/sync_driftdrag_package.sh
bash 08_gpu_embedding/run_embedding_job.sh    # SOURCE_VERSION=v0 TARGET_VERSION=v1 by default
bash 08_gpu_embedding/delete_gpu_vm.sh        # DO THIS IMMEDIATELY AFTER
```
Re-embeds every shard's existing chunks with real `nomic-embed-text-v1.5` as `model_version=v1`. Repeat with `SOURCE_VERSION=v1 TARGET_VERSION=v2 MODEL_NAME=...` (a fine-tuned checkpoint, once you have one — see `../drift-drag/PLAN.md` E0) to build out the full v0..v5 sequence.

### Step 12 — Re-run Ray drift detection/compensation now that v0 AND v1 exist
```bash
bash 06_ray_cluster/run_ray_job.sh parallel_drift_detection.py
bash 06_ray_cluster/run_ray_job.sh parallel_compensation.py
bash 06_ray_cluster/run_ray_job.sh run_scheduler.py
```
This time every shard should print real `anchor_residual`/`data_drift_score` numbers instead of "skipping".

### Step 13 — Airflow: run the whole Stage 0-4 pipeline as one DAG
```bash
bash 07_airflow/create_airflow_vm.sh
# wait ~2-5 min, then:
gcloud compute ssh driftdrag-airflow --zone asia-south1-a --project drift-rag --command='sudo cat /opt/airflow/standalone_admin_password.txt'
bash 07_airflow/sync_dag.sh
```
Open `http://<airflow-vm-external-ip>:8080`, log in as `admin` / (the password above), find `drift_drag_pipeline` in the DAG list, and trigger it manually (it's `schedule=None` — not a recurring cron job). Watch the graph view; `delete_dataproc_cluster` is configured to run even if an earlier task fails, so a bad run still cleans up its Dataproc cluster.

### Step 14 — Always, at the end of a session
```bash
bash scripts/status.sh        # see what's running
bash scripts/teardown_all.sh  # delete Dataproc/GPU (if any), stop Kafka/Redis/Airflow/Ray
```

---

## Troubleshooting

**Reaching Kafka (or any internal-only service) from your laptop.** Everything in this deployment listens on internal VPC IPs only (no public Kafka/Redis endpoints, deliberately — see the security note in `04_redis_vm/startup-script.sh`). To run a one-off producer/consumer from your own machine instead of `gcloud compute ssh ... --command=`, open an SSH tunnel:
```bash
gcloud compute ssh driftdrag-kafka --zone asia-south1-a --project drift-rag -- -L 9092:localhost:9092
# in another terminal, now localhost:9092 on YOUR machine reaches Kafka
```

**Dataproc cluster creation fails with a quota error.** Check `gcloud compute regions describe asia-south1 --project drift-rag` for your `CPUS` quota/usage — the default cluster config here uses 3 x n2-standard-4 (12 vCPUs); reduce `DATAPROC_NUM_WORKERS` in `env.sh` or use smaller machine types if you're close to your limit.

**GPU VM creation fails with a quota error.** You have `NVIDIA_T4_GPUS` quota = 1 in this project/region — if a previous GPU VM is still around (check `scripts/status.sh`), delete it first; only one can exist at a time.

**Airflow DAG's Dataproc/GCS tasks fail with a permissions error.** The Airflow VM authenticates as its attached service account (via `--scopes=cloud-platform` in `create_airflow_vm.sh`) — confirm that service account has the Dataproc Editor / Storage Admin roles (the GCE default service account has "Editor" on the project unless your organization has restricted it; if so, grant the specific roles via `gcloud projects add-iam-policy-binding`).

**"driftdrag" import errors on Ray/GPU jobs.** Re-run the relevant `sync_driftdrag_package.sh` — these copy `../drift-drag/src/driftdrag` onto the VM fresh; if you've edited the package locally since the last sync, the VM has a stale copy.

## How this maps back to PLAN.md's experiments

- **E0** (drift-simulation protocol): Step 11's GPU re-embedding loop, run repeatedly with successive `SOURCE_VERSION`/`TARGET_VERSION` pairs, is how you build the v1..v5 sequence at GCP scale instead of on your local machine.
- **E1-E4** (retrieval-quality comparisons): pull the Parquet from `gs://<bucket>/embeddings` locally (`gcloud storage cp -r ...`) and run them through `../drift-drag/src/driftdrag/eval/metrics.py` exactly as the local validation did — no need to keep GCP resources up for this part (see the earlier conversation: quality metrics are hardware-independent, run these locally).
- **E5-E6** (distributed system latency/throughput, GPU-hours cost curves): this is the part that specifically needs the multi-VM Dataproc + Kafka + Ray setup here — a local machine cannot produce a genuine multi-node latency measurement.
