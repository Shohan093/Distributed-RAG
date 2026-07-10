# Drift-DRAG

Drift-compensated distributed RAG for streaming analytics. See [PLAN.md](PLAN.md) for the full experimental plan and methodology this repository implements, and [DEPLOYMENT.md](DEPLOYMENT.md) for moving it from one machine onto multiple VMs.

This repo is genuinely polyglot: **Scala + Spark Structured Streaming** does the Kafka-facing shard-assignment half of ingestion, **PySpark** does the embedding half (needs the Python ML stack), and both sit on top of Kafka, Redis, MinIO, FAISS/Milvus, and MLflow. See [PLAN.md §1.1](PLAN.md#11-big-data-framework-mapping-scala-pyspark-and-the-rest-of-the-stack) for why each piece is in the language it's in.

## Layout

```
configs/            Hydra configs (infra: local|docker|cluster; experiment: e1..e8)
src/driftdrag/       Python package: ingest, ledger, detect, compensate, schedule, index, eval
scala/shard-partitioner/   sbt project: the Scala/Spark half of Stage 0
scripts/             smoke test + sample data producer + local env helper
infra/ansible/        multi-VM provisioning (see DEPLOYMENT.md)
tests/                pytest suite for every Python module
```

## Requirements

- Python 3.10+ (this sandbox used 3.14) with `pip install -e .[dev]` (installs `driftdrag` + pytest/fakeredis)
- **JDK 17+** for PySpark 4.x, or JDK 8/11 if you pin `pyspark==3.5.3` (see the note in `requirements.txt` about matching the Spark version used by `docker-compose.yml` / `scala/shard-partitioner/build.sbt`)
- sbt 1.9+ and a JDK to build `scala/shard-partitioner` (only needed once you run it against real Kafka; not required for the Python-only smoke test)
- Docker + Docker Compose (only for the single-VM/multi-VM profiles, not the local smoke test)

### Windows note (no system-wide installs needed)

Spark's local filesystem writes need a real Java 17+ and, on Windows, a `winutils.exe`/`hadoop.dll` shim (Hadoop's native-IO layer). Rather than installing anything system-wide, this repo's `scripts/env.ps1` points at a **project-local, portable** JDK 17 + winutils shim under `.tools/` (gitignored — download it once, see below). Linux VMs (the real deployment target) don't need any of this; their JRE and native filesystem calls just work.

```powershell
# one-time setup (already done in this sandbox's .tools/, safe to re-run):
mkdir .tools
curl -L -o .tools/jdk17.zip "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk"
Expand-Archive .tools/jdk17.zip .tools/ ; rm .tools/jdk17.zip
mkdir .tools/hadoop/bin
curl -L -o .tools/hadoop/bin/winutils.exe "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/winutils.exe"
curl -L -o .tools/hadoop/bin/hadoop.dll "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/hadoop.dll"
```

## Local smoke test (no Docker, no Kafka)

Proves the ingest -> ledger -> ANN-index path works before anything touches a VM:

```powershell
. .\scripts\env.ps1        # Windows only; sets JAVA_HOME/HADOOP_HOME to .tools/
pip install -e ".[dev]"
python -m pytest tests/ -q                # 45 tests, pure-Python modules
python scripts\run_smoke_test.py          # PySpark local[*] + FAISS + fakeredis ledger, end to end
```

Expected tail of `run_smoke_test.py`:

```
embeddings: 60 chunks across 4 shards
FAISS shard 0: 15 vectors, self-query OK -> [('doc-0002-0001', 1.0)]
drift ledger: 4 shard records, history exported to .../data/lake/ledger_history.parquet

SMOKE TEST PASSED
```

What it actually exercises: `scripts/produce_sample_docs.py` writes a tiny file-based "stream"; `driftdrag.ingest.spark_embed_job` runs as a real (if trivially small) Spark Structured Streaming query in `local[*]` mode, chunking + embedding + writing partitioned Parquet; `driftdrag.index.faiss_shard` builds a real FAISS index from that Parquet output and round-trips a query; `driftdrag.ledger.drift_ledger` exercises the same Redis-shaped API against `fakeredis` and exports Parquet history. Swapping `infra=local` for `infra=docker`/`cluster` (see `configs/infra/`) replays the identical job code against real Kafka/Redis/MinIO — see DEPLOYMENT.md.

## Scala shard-partitioner

```powershell
cd scala\shard-partitioner
sbt compile   # verified in this repo: compiles cleanly under sbt 1.9.9 / Scala 2.12.18 / JDK 17
sbt test      # ScalaTest: CRC32 shard assignment matches the Python fixtures in tests/test_sharding.py
sbt assembly  # produces target/scala-2.12/shard-partitioner-assembly-0.1.0.jar for spark-submit
```

Run it against the docker-compose Kafka broker once that's up (see DEPLOYMENT.md):

```
spark-submit --class com.driftdrag.ingest.ShardPartitionerJob \
  target/scala-2.12/shard-partitioner-assembly-0.1.0.jar \
  --source=kafka --bootstrap-servers=localhost:9092 --topic=docs-raw \
  --output-path=s3a://driftdrag-lake/shard_assignments --num-shards=32 \
  --redis-host=localhost --redis-port=6379
```

## Single-VM Docker stack

```bash
cp .env.example .env    # edit if you changed any ports
docker compose up
```

Brings up Kafka (KRaft), Spark master+worker, Redis, MinIO, and MLflow. Point the same PySpark job / Scala jar at it with `+infra=docker` (Hydra) instead of `+infra=local`.

## Multi-VM

See [DEPLOYMENT.md](DEPLOYMENT.md).

## Running the experiments (E1-E8)

`configs/experiment/e1_degradation.yaml` is the template; `PLAN.md §3` describes what each of E1-E8 sweeps. Real runs need trained model-version checkpoints (Section 5 of PLAN.md) rather than the `HashingDriftBackend` smoke-test stand-in — swap `embedding_backend: hashing` for `sentence-transformer` once those checkpoints exist.
