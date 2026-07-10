# Deployment: from one machine to multiple VMs

This is the operational companion to [PLAN.md §8](PLAN.md#8-multi-vm-deployment--portability-design), which explains *why* the project is structured this way. This document is the *how*.

## The three profiles, recap

| You are here | Command |
|---|---|
| Local smoke test | `python scripts/run_smoke_test.py` (see README.md) |
| Single VM, everything in Docker | `docker compose up` |
| N VMs | `infra/ansible/site.yml` against `inventory.ini`, then each VM runs `docker compose -f docker-compose.yml -f docker-compose.cluster.yml --profile <its role> up -d` |

Nothing in `src/driftdrag/`, `scala/shard-partitioner/`, or the job code changes between rows. Only `.env` / `configs/infra/*.yaml` change.

## Step 1: single-VM Docker (do this before multi-VM)

```bash
cp .env.example .env
docker compose up -d
docker compose ps                      # kafka, spark-master, spark-worker, redis, minio, minio-init, mlflow
```

Submit the Python job against it:

```bash
python -m driftdrag.ingest.spark_embed_job \
  --master spark://localhost:7077 --source kafka \
  --bootstrap-servers localhost:9092 --topic docs-raw \
  --output-path s3a://driftdrag-lake/embeddings \
  --checkpoint-path s3a://driftdrag-lake/_checkpoints/spark_embed_job
```

(or drive it through Hydra with `+infra=docker`, which fills in these values from `configs/infra/docker.yaml`).

Submit the Scala job against the same broker:

```bash
cd scala/shard-partitioner && sbt assembly
spark-submit --master spark://localhost:7077 \
  --class com.driftdrag.ingest.ShardPartitionerJob \
  target/scala-2.12/shard-partitioner-assembly-0.1.0.jar \
  --source=kafka --bootstrap-servers=localhost:9092 --topic=docs-raw \
  --output-path=s3a://driftdrag-lake/shard_assignments --num-shards=32 \
  --redis-host=localhost --redis-port=6379
```

Produce test traffic into Kafka (any producer works; a minimal one):

```bash
python - <<'PY'
import json, time
from kafka import KafkaProducer
p = KafkaProducer(bootstrap_servers="localhost:9092", value_serializer=lambda v: json.dumps(v).encode())
for i in range(20):
    p.send("docs-raw", {"doc_id": f"doc-{i}", "text": "sample streaming document", "timestamp": int(time.time()), "source": "manual-test"})
p.flush()
PY
```

## Step 2: provision N VMs with Ansible

1. Create N Linux VMs (any provider). Make sure they can all reach each other on the ports in `.env.example` (9092 Kafka, 7077/8080 Spark, 6379 Redis, 9000/9001 MinIO, 5000 MLflow) — put them in the same VPC/subnet or open a security group between them.
2. `cp infra/ansible/inventory.ini.example infra/ansible/inventory.ini` and fill in real `ansible_host` IPs per role (see the role table in PLAN.md §8.2). A VM can be in more than one group for small deployments (e.g. put `storage` and `spark_master` on the same VM).
3. `ansible-playbook -i infra/ansible/inventory.ini infra/ansible/site.yml` — installs Docker, clones the repo, renders each VM's `.env` from its group membership (`infra/ansible/templates/env.j2`), and starts that VM's container profile(s).
4. Verify: `docker compose -f docker-compose.yml -f docker-compose.cluster.yml --profile <role> ps` on each VM.

Scaling a role out later (e.g. more Spark workers as the corpus grows toward the 100M headline run, E6) is: add a host to the `[spark_worker]` group in `inventory.ini`, re-run `site.yml` — nothing else changes.

## Step 3: point jobs at the cluster

Run the PySpark/Scala jobs from the `client` VM (or any machine with network access to the cluster and the repo checked out), using `+infra=cluster` so every hostname comes from `configs/infra/cluster.yaml` / the cluster `.env` instead of `localhost`:

```bash
python -m driftdrag.ingest.spark_embed_job --master spark://$SPARK_MASTER_HOST:7077 --source kafka \
  --bootstrap-servers $KAFKA_BOOTSTRAP_SERVERS --topic docs-raw \
  --output-path s3a://driftdrag-lake/embeddings --checkpoint-path s3a://driftdrag-lake/_checkpoints/spark_embed_job
```

## Things that only matter once you're actually distributed

- **Shard assignment must be deterministic across languages and machines.** Both `driftdrag.ingest.sharding.shard_for_doc` (Python) and `ShardPartitioning.shardFor` (Scala) hash `doc_id` with CRC-32 over UTF-8 bytes — verified to agree exactly in this repo (`tests/test_sharding.py` and `scala/.../ShardPartitioningSpec.scala` share fixture values). If you ever add a third producer of shard assignments, it must use the same hash, not a language's built-in `hash()`.
- **The Drift State Ledger is a single logical Redis instance**, not per-VM local state — every Spark executor across every worker VM, and the Scala job, read/write the same `REDIS_HOST`. If Redis becomes a bottleneck at very high shard counts, PLAN.md §7 notes etcd as the stronger-consistency alternative; swapping it in means changing `ledger/drift_ledger.py::connect()`'s backend, not the ledger's call sites.
- **The data lake is MinIO/S3, not local disk**, once there is more than one Spark executor — `configs/infra/{docker,cluster}.yaml` set `lake.root: s3a://...` for exactly this reason. Don't let a job fall back to writing `data/lake/...` on a VM's local disk; another executor on another VM won't see it.
- **`network_mode: host`** in `docker-compose.cluster.yml` requires Linux hosts — this is why the overlay is separate from `docker-compose.yml` rather than the default.
- **Version alignment**: the PySpark client, the Scala job's `sparkVersion` in `build.sbt`, and the `apache/spark` image tag in `docker-compose.yml` must share a Spark major.minor once jobs submit to `spark://spark-master:7077` instead of running in `local[*]` — see the note at the top of `requirements.txt`.

## Scaling knobs as the corpus grows (10M -> 100M -> 1B extrapolated, E1/E6)

| Knob | Where | What it does |
|---|---|---|
| `num_shards` | `configs/config.yaml` / `.env` `NUM_SHARDS` | More, smaller shards -> finer-grained compensate/re-embed decisions (Stage 4) at the cost of more per-shard bookkeeping in the ledger |
| `spark_worker` group size | `inventory.ini` | More executors for the embedding job and the Scala shard partitioner |
| `KAFKA_NUM_PARTITIONS` | `.env` | Caps how parallel the Kafka consumers (both languages) can get |
| `index.backend: faiss\|milvus` (future work slot -- no code path reads this yet) | `configs/experiment/*.yaml` | FAISS-per-shard (this scaffold's default, `index/faiss_shard.py`) up to the point a single VM's shards stop fitting in memory, then Milvus (`docker compose -f docker-compose.yml -f docker-compose.milvus.yml up`) for a natively-distributed ANN layer |
