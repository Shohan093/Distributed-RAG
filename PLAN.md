# Drift-Compensated Distributed RAG for Streaming Analytics
## Full Experimental Plan, Methodology Pipeline, and Project Structure for a Q1 Journal Submission

**Working system name:** Drift-DRAG (rename freely; a memorable acronym helps at review time)
**Core claim:** The first framework that jointly handles (a) embedding-model drift, (b) streaming data drift, and (c) distributed sharded retrieval under a unified, budget-aware controller — without global re-indexing of a 10⁸–10⁹-vector corpus.

This repository (`drift-drag/`) implements the scaffold described below: a runnable, polyglot (Scala + PySpark + Python) pipeline that starts on one machine and is designed, from the ground up, to be relocated onto multiple virtual machines without code changes — only configuration and `.env` values change. See [README.md](README.md) for how to run the local smoke test and [DEPLOYMENT.md](DEPLOYMENT.md) for how to move it onto N VMs.

---

## 1. Why This Is a *Big Data Analytics* Paper (and how to prove it to reviewers)

Q1 big-data venues (e.g., IEEE TBD, IEEE TKDE, VLDB Journal, ACM TKDD, JBD, FGCS, Information Sciences) will accept the RAG framing only if the *systems and data-scale contribution* is first-class. Anchor every experiment to the canonical dimensions:

| Big-data dimension | How your project instantiates it | Experiment that proves it |
|---|---|---|
| **Volume** | Sharded vector corpus, target 10⁸ vectors real + 10⁹ extrapolated (cost model) | E1, E6 (scalability sweep) |
| **Velocity** | Continuous document arrival (CC-News / LogHub streams), periodic model upgrades v₁→v_T | E2, E5 (streaming ingest + query latency under load) |
| **Variety** | Web news + QA + system logs + multilingual queries; heterogeneous shard versions | E1, E4 (cross-domain drift) |
| **Veracity / Value** | Retrieval quality (Recall/nDCG) and downstream analytics quality (log anomaly F1, QA EM) under drift | E3, E7 |
| **Distributed infrastructure** | Multi-node sharded FAISS/Milvus, streaming ingestion (Kafka + Spark/Flink), per-shard controllers | E5, E6 |

**Reviewer-proofing rule:** every plot should have either (i) corpus size, (ii) throughput/latency, or (iii) compute cost (GPU-hours, #vectors re-embedded, $-normalized) on one axis. That is what distinguishes a big-data-analytics paper from an IR paper.

---

## 1.1 Big-Data Framework Mapping (Scala, PySpark, and the rest of the stack)

Every stage below names a specific engine on purpose, not a generic "Spark job." The split follows one rule: **Scala/Spark owns anything JVM-native, high-throughput, or on the critical ingest path; PySpark owns anything that needs the Python ML ecosystem (embedding models, scikit-learn, numpy/scipy drift math); a handful of dedicated big-data services (Kafka, Redis, MinIO/S3, FAISS/Milvus, MLflow) sit underneath both.** This is also why the codebase (`scala/shard-partitioner/`, `src/driftdrag/ingest/spark_embed_job.py`) is genuinely polyglot rather than PySpark-only with a token Scala file.

| Stage (Section 2) | Framework(s) | Why this language/engine |
|---|---|---|
| Stage 0 — sharding assignment | **Scala + Spark Structured Streaming** (`scala/shard-partitioner`) | Consumes Kafka directly with no py4j hop; CRC32-based shard hashing must run identically to the Python side (see `ShardPartitioning.shardFor` / `driftdrag.ingest.sharding.shard_for_doc`) — cheapest to guarantee that by keeping the hot path JVM-native on both ends of the comparison |
| Stage 0 — chunking + embedding | **PySpark** (`src/driftdrag/ingest/spark_embed_job.py`) via `mapInPandas` | Needs the Python ML stack (HF/sentence-transformers backbones, Section 5); FAISS/Milvus SDKs are also Python-first |
| Stage 0 — streaming bus | **Kafka** (KRaft mode) / Redpanda as a drop-in swap | Standard durable pub/sub for Volume+Velocity; Flink is documented as an alternative low-latency engine for the E5 ultra-low-latency ingest variant, but Spark Structured Streaming is the default engine on both language sides so there is one execution model to reason about, not two |
| Stage 0 — data lake | **Parquet on MinIO (S3-compatible)** | Columnar storage for embeddings/anchors/ledger history; MinIO stands in for HDFS/S3 and is what both the Scala and PySpark jobs read/write via `s3a://` in `infra=docker/cluster` |
| Stage 1 — Drift State Ledger | **Redis** (`redis-py` in Python, `Jedis` in Scala) + Parquet history export | A KV store needs one client per language, not a shared library — this is the one place both jobs talk to the *same* running service directly (see `ledger/drift_ledger.py`, `ShardPartitionerJob.updateLedger`) |
| Stage 2 — anchor-residual / MMD / ADWIN detectors | **Python (numpy/scipy/scikit-learn)**, invoked as PySpark `mapInPandas`/UDFs at scale | Kernel two-sample tests and k-means anchor selection have no mature Scala/Spark-native equivalents worth reimplementing |
| Stage 3 — compensation operators | **Python (numpy/scikit-learn)** | Procrustes/low-rank-affine/residual-MLP are linear-algebra-heavy; run driver-side or as a UDF over per-shard batches |
| Stage 3 — ANN index | **FAISS** (per-shard, pilot/10M scale) → **Milvus** (distributed, 100M+/multi-VM scale, `docker-compose.milvus.yml`) | FAISS is deliberately kept out of Spark executors (see `index/faiss_shard.py` docstring); Milvus is the natively-distributed drop-in for E5/E6 |
| Stage 4 — budgeted scheduler | **Python (numpy/scikit-learn)**, orchestrated by **Hydra** multirun sweeps | Multiple-choice-knapsack DP + the quality-drop regressor are both small, driver-side computations — no benefit from Spark here |
| Stage 5 — downstream analytics | **PySpark** for LogHub/CC-News batch feature prep, **Python/PyTorch** for the generator (E7) | Bulk log/news ETL benefits from Spark's parallelism; the RAG generator is a standard HF/PyTorch model |
| Experiment tracking | **MLflow** | Every E-series script logs metrics/artifacts here regardless of which language produced them |
| Orchestration/config | **Hydra** (`configs/`) | Language-agnostic: the same `infra=local\|docker\|cluster` switch parametrizes both the PySpark job's CLI args and the Scala job's `--bootstrap-servers=...` style args |

**Consequence for reproducibility:** any number in the paper traces to one job in one language, not an ambiguous "the pipeline." Table columns in E1–E7 result CSVs should record which engine produced each row (a `producer` column: `pyspark-embed-job` / `scala-shard-partitioner` / `python-eval`) for exactly this reason.

---

## 2. Methodology Pipeline (novel components + key references per step)

### Stage 0 — Streaming ingestion & sharding layer
Documents arrive as a stream; a partitioner assigns them to shards (by time window, topic cluster, or hash). Each shard records the embedding-model version used at its last (re)indexing → the **version-heterogeneous index** that defines your problem.

- Infrastructure: Kafka (or Redpanda) for the stream; Spark Structured Streaming (Scala for shard assignment, PySpark for chunk/embed) micro-batches; FAISS HNSW/IVF-PQ per shard (Milvus if you want a production-grade distributed store).
- Key refs: Johnson et al., *Billion-scale similarity search with GPUs* (FAISS, IEEE TBD 2021); Malkov & Yashunin, HNSW (TPAMI 2020); Wang et al., *Milvus* (SIGMOD 2021); Zaharia et al. (Spark); Carbone et al. (Flink); Kreps et al., *Kafka: a Distributed Messaging System for Log Processing* (NetDB 2011); Jiang et al., *RAGCache* (project file — knowledge caching for RAG at scale, use for the serving-efficiency discussion).

### Stage 1 — Drift State Ledger (novel bookkeeping primitive)
A distributed, append-only ledger storing, per shard s and per model transition vⱼ→vⱼ₊₁: (i) the estimated drift vector(s) Δ (global or per-cluster), (ii) anchor-set residual statistics, (iii) data-drift statistics, (iv) last re-embed timestamp and version. This is the state that every other component reads/writes. Implemented as a replicated key-value store (Redis, with etcd as a documented alternative for stronger consistency) + Parquet history for offline analytics.

- Key refs: Yu et al., *Semantic Drift Compensation* (CVPR 2020 — origin of drift-vector estimation); **QDC** (CoLLAs 2025, project file — cumulative drift vectors Σ Δⱼ→ⱼ₊₁ and re-indexing-free retrieval); **Drift-Adapter** (EMNLP 2025, project file — learned Procrustes / low-rank affine / residual-MLP mappings between spaces). Your novelty: QDC keeps *one global* drift vector per transition; you keep **per-shard, per-cluster** drift state in a distributed ledger.

### Stage 2 — Dual Drift Detector (novel: separates two orthogonal drift axes)
1. **Model drift (cross-space):** an *anchor-residual monitor*. Maintain a small fixed anchor set A (e.g., 1–5k representative chunks per shard, selected by k-means coverage). On every model upgrade, re-embed only A and measure residual r_s = mean‖f_t(a) − (f_{t'}(a) + Δ̂)‖ after compensation. This is the cross-space error signal that intra-space monitors (e.g., DARLA's MMD) cannot see — your Literature-Review-4 differentiator.
2. **Data drift (intra-space):** streaming distribution-change detection over incoming chunk embeddings per shard — MMD two-sample tests on sliding windows and/or ADWIN on summary statistics (centroid displacement, density).

- Key refs: Gretton et al., *A Kernel Two-Sample Test* (JMLR 2012 — MMD); Bifet & Gavaldà, ADWIN (SDM 2007); Gama et al., *A Survey on Concept Drift Adaptation* (ACM CSUR 2014); **DARLA** (project file — MMD-based drift detection for log streams; also your strongest applied baseline); *Drift-Aware RAG* log-anomaly design patterns therein.

### Stage 3 — Query-time Compensation Path (extends QDC to distributed, cluster-local corrections)
- Query router (RAGRoute-style shallow NN or centroid heuristics) selects candidate shards using shard centroids, density, and *version metadata* from the ledger — a **version-aware router** is itself a small novelty.
- For each selected shard indexed under f_{t'}: embed query with current f_t, subtract the **cumulative per-cluster drift** Σ Δ (t'→t) for the query's nearest cluster, then run ANN search in the old space. Fuse results across shards with score calibration (residual-weighted reciprocal rank fusion — another small novelty: down-weight shards whose anchor residual is high).
- Key refs: **QDC** (project file — Eq. 3–7, cumulative drift + cosine retrieval); **Drift-Adapter** (project file — the learned-mapping alternative you ablate against); **RAGRoute** (project file — lightweight federated query routing; EuroMLSys 2025); **DRAGON / Distributed RAG** (project files — device–cloud and peer-to-peer distributed retrieval context); Guerraoui et al. federated-search cost analysis.

### Stage 4 — Budgeted Re-embedding Scheduler (the paper's decision-theoretic core)
Per maintenance epoch, given compute budget B (GPU-hours), choose per shard among {**compensate** (cost ≈ 0), **partial re-embed** (hot clusters only), **full re-embed**} to maximize predicted retrieval quality. Formulated as a multiple-choice-knapsack problem, solved exactly by dynamic programming over a discretized budget axis (`schedule/scheduler.py::schedule_shards`); each shard's *predicted quality drop* comes from a lightweight regressor over two cheap signals: anchor residual + data-drift score (this operationalizes RQ3).

- Novelty: Drift-Adapter "defers" re-embedding but has no stopping rule; DARLA refreshes content but ignores model-version drift; QDC never re-embeds at all. Your scheduler is the missing decision layer, and the quality-drop predictor is a publishable sub-contribution on its own.
- Key refs: Drift-Adapter §5.5–5.6 (cost projections for 1M→1B corpora — reuse their cost-model style); RAGCache (cost/latency modeling discipline); classic budgeted-maintenance framing from database view-maintenance literature (cite one survey, e.g., Chirkova & Yang, *Materialized Views*, FnT-DB 2012).

### Stage 5 — Downstream streaming-analytics tasks (proves "Value")
Two applied tracks so the paper isn't retrieval-only:
1. **Log-analytics / AIOps:** retrieval-augmented anomaly detection over LogHub streams (compare with DARLA-style content refresh).
2. **News/QA analytics:** temporal QA over CC-News / StreamingQA where answers depend on freshly ingested documents.
- Key refs: Liška et al., *StreamingQA* (ICML 2022); He et al., *LogHub* datasets (ISSRE 2023 version); Lewis et al., RAG (NeurIPS 2020); **CoRAG** (project file — if you add a multi-hop generation experiment); **Language Drift in Multilingual RAG** (project file — motivates a multilingual drift ablation, optional E8).

---

## 3. Experimental Design

### E0 — Infrastructure & drift-simulation protocol (prerequisite, ~weeks 1–4)
1. Build the model-version sequence v₁→v₅: continually fine-tune nomic-embed-text-v1.5 (and BGE-base as second backbone) on 5 sequential tasks/domains — reuse the QDC task order (MS MARCO → NQ → HotpotQA → FEVER → FiQA) for direct comparability with their published nDCG@10 numbers, then add a domain-incremental variant (news → logs → finance) for the streaming story. `src/driftdrag/ingest/embeddings.py::HashingDriftBackend` is a dependency-free stand-in with a *known* drift ground truth, used by the local smoke test and unit tests until real checkpoints are trained.
2. Build the sharded corpus: start at 10M chunks (pilot), scale to 100M (headline), extrapolate to 1B via the Drift-Adapter-style cost table. 32–256 shards; each shard tagged with its index-time model version.
3. Streaming harness: Kafka topics for CC-News (2016–2019 slices, chronological replay) and LogHub (HDFS/BGL/Thunderbird); the Scala shard-partitioner + PySpark embed jobs for chunk→embed→upsert; Poisson query arrivals (follow RAGCache's workload-generation methodology).
4. Reproducibility scaffolding: Docker Compose (`docker-compose.yml` + `docker-compose.cluster.yml`) / Ansible (`infra/ansible/`) for VM provisioning, fixed seeds, config-driven runs (Hydra, `configs/`), MLflow tracking.

### E1 — Degradation quantification (RQ1)
**Question:** How much does retrieval quality degrade on a sharded, version-heterogeneous index as the model is updated without re-indexing?
- Sweep: #model transitions (1–4), fraction of stale shards (0–100%), shard count, corpus size (10M/50M/100M). See `configs/experiment/e1_degradation.yaml`.
- Metrics: Recall@{1,10,100}, nDCG@10, MRR per task-age; **forgetting** = quality(t) − quality at index time (`eval/metrics.py::forgetting`); report per-shard heatmaps (staleness × transition-distance).
- Baselines here: *Misaligned* (no compensation) vs *full re-index oracle* — these bound the achievable range for everything that follows.
- Big-data hook: degradation vs corpus size and vs stream position (velocity).

### E2 — Compensation accuracy (RQ2)
**Question:** Can a small per-cluster correction translate new-model queries into old shard spaces accurately?
- Compare compensation operators on identical shards (`compensate/drift_vectors.py`): (a) global drift vector (QDC-global), (b) **per-shard drift vector**, (c) **per-cluster drift vectors** (k ∈ {8, 32, 128} clusters/shard), (d) Orthogonal Procrustes, (e) Low-rank affine, (f) Residual MLP (the three Drift-Adapter parameterizations), (g) oracle re-embed.
- Metrics: Adaptation Recall Ratio (ARR, `compensate/drift_vectors.py::adaptation_recall_ratio`, from Drift-Adapter) so numbers are cross-paper comparable; added query latency (µs); memory per shard; #anchor pairs needed (sample-efficiency curve: 100 → 20k pairs).
- Expected headline: per-cluster drift vectors ≈ learned adapters' ARR at near-zero training cost and O(k·d) memory, and compose additively across transitions (adapters must be chained or retrained — test error accumulation over 4 chained transitions; this is a genuinely open comparison neither paper ran).

### E3 — Quality-drop prediction (RQ3)
**Question:** Do two cheap per-shard signals (anchor residual, data-drift score) predict actual retrieval-quality drop?
- Collect (signal, measured ΔRecall) pairs across E1/E2 grid; fit ridge / gradient-boosted regressor (`schedule/scheduler.py::QualityDropPredictor`); report R², Spearman ρ, and calibration; ablate each signal alone.
- This regressor is the scheduler's oracle — its accuracy directly justifies Stage 4.

### E4 — Data-drift × model-drift interaction (the "jointly" claim)
- 2×2 factorial: {data drift on/off (chronological CC-News vs shuffled)} × {model drift on/off}. Show the interaction term is non-additive — degradation under both is worse than the sum — which is exactly the gap no prior system addresses.
- Include a domain-shift stress test (news→logs) where compensation should partially fail (honest negative result; reviewers reward it), matching Drift-Adapter's finding that drastic drift breaks linear maps.

### E5 — End-to-end streaming system evaluation
- Full pipeline live: Kafka ingest at λ ∈ {1k, 10k, 100k} docs/min, concurrent Poisson queries at {10, 100, 1000} QPS, model upgrade injected mid-run.
- Metrics: p50/p99 query latency, ingest lag, throughput, recall-over-time curves around the upgrade event (the "no-downtime" money plot: full re-index shows a quality/latency cliff; Drift-DRAG stays flat).
- Compare serving strategies: full re-index, dual-index serving, Drift-Adapter (central adapter), Drift-DRAG.

### E6 — Budgeted scheduler & cost–quality Pareto frontier (headline experiment)
- Sweep budget B from 0% to 100% of full-re-embed cost; policies: **never-update**, **QDC-global (compensate-only)**, **Drift-Adapter (adapt-only)**, **DARLA-style content refresh (data-drift-only trigger)**, **round-robin re-embed**, **greedy-by-staleness**, **Drift-DRAG scheduler (predictor-driven knapsack, `schedule/scheduler.py::schedule_shards`)**, **full re-index oracle**.
- Metrics: Recall/nDCG vs GPU-hours and vs #vectors re-embedded; area-under-Pareto; regret vs oracle.
- Expected headline number for the abstract: "X% of oracle quality at Y% of oracle cost" (Drift-Adapter's 95–99% ARR at >100× cost reduction sets the bar — you must beat or match it *while also handling data drift and multiple chained transitions*).

### E7 — Downstream analytics value
- Log anomaly detection F1/AUROC on LogHub under drift (vs DARLA reproduction/approximation); temporal QA EM/F1 on StreamingQA; optionally CoRAG-style multi-hop with your retriever underneath. Show retrieval gains propagate to task metrics.

### E8 (optional, strengthens Variety) — Multilingual drift
- Replay multilingual news; measure whether compensation quality is language-dependent (motivated by the Language-Drift-in-Multilingual-RAG project file). Cheap to run, adds a distinctive ablation.

### Ablations checklist (fold into E2/E6)
Anchor-set size & selection strategy (random vs k-means coverage vs herding, `detect/anchor_residual.py::select_anchor_set`); #clusters per shard; ledger update frequency; router with/without version-awareness; residual-weighted vs plain fusion; detector thresholds; backbone (nomic vs BGE); ANN index type (HNSW vs IVF-PQ vs Milvus).

### Statistical rigor (Q1 requirement)
- ≥3 seeds per configuration (5 for headline results); mean ± 95% CI (`eval/metrics.py::aggregate_recall`); paired bootstrap (`eval/metrics.py::paired_bootstrap_pvalue`) or Wilcoxon signed-rank for baseline comparisons; Holm–Bonferroni across the baseline family; report effect sizes, not just p-values. Release code + configs + drift-simulation scripts (anonymized repo at submission).

---

## 4. Baselines (exact list, one line of provenance each)
1. **Full re-index oracle** — upper bound (standard practice).
2. **Never-update / Misaligned** — lower bound (Drift-Adapter's "no adaptation").
3. **QDC-global** — single cumulative drift vector, no re-embedding (CoLLAs 2025, project file).
4. **Drift-Adapter** (OP / LA / MLP variants) — learned central mapping (EMNLP 2025, project file).
5. **DARLA-content-refresh** — data-drift-triggered knowledge-base refresh, blind to model version (project file).
6. **Dual-index serving** — operational baseline from Drift-Adapter's comparison set.
7. **Round-robin / staleness-greedy re-embed** — naive scheduling controls (yours, needed to isolate the predictor's contribution).

---

## 5. Datasets, Models, Metrics (consolidated)

- **Datasets:** MS MARCO, NQ, HotpotQA, FEVER, FiQA (BEIR splits — comparability with QDC); StreamingQA (temporal QA); CC-News (chronological stream); LogHub HDFS/BGL/Thunderbird (log streams). Cite BEIR (Thakur et al., NeurIPS D&B 2021) for the retrieval-benchmark protocol.
- **Embedding backbones:** nomic-embed-text-v1.5 (137M), BGE-base-en-v1.5 (110M); continual fine-tuning with the QDC recipe (FT + KD) to generate realistic version sequences. `SentenceTransformerBackend` in `ingest/embeddings.py` is the drop-in wrapper for these.
- **Generators (E7 only):** Llama-3.x-8B-Instruct or comparable open model, held fixed across conditions.
- **Metrics:** Recall@k, nDCG@10, MRR, ARR; forgetting (Kemker/Chaudhry-style CL measure); GPU-hours; #vectors re-embedded; p50/p99 latency; QPS; ingest lag; downstream EM/F1/AUROC; predictor R²/ρ.

---

## 6. Project / Repository Structure

```
drift-drag/
├── PLAN.md, README.md, DEPLOYMENT.md, LICENSE, CITATION.cff
├── pyproject.toml, requirements.txt        # driftdrag Python package
├── docker-compose.yml                      # single-VM stack: Kafka, Spark master/worker, Redis, MinIO, MLflow
├── docker-compose.cluster.yml              # multi-VM overlay: profiles + host networking (see DEPLOYMENT.md)
├── docker-compose.milvus.yml               # optional distributed-ANN overlay for E5/E6 scale-out (future work slot)
├── .env.example                            # every hostname/port that changes when moving to N VMs
├── configs/                                # Hydra configs
│   ├── config.yaml
│   ├── infra/{local,docker,cluster}.yaml   # the ONE thing that changes per environment
│   └── experiment/e1_degradation.yaml ...  # e2..e8 follow the same pattern
├── src/driftdrag/
│   ├── ingest/                             # sharding.py (CRC32, shared w/ Scala), chunking.py, embeddings.py,
│   │                                       # stream_source.py (kafka|file), spark_embed_job.py (PySpark)
│   ├── ledger/                             # drift_ledger.py (Redis/fakeredis + Parquet history)
│   ├── detect/                             # anchor_residual.py, data_drift.py (MMD + AdwinLite)
│   ├── compensate/                         # drift_vectors.py (global/per-shard/per-cluster/Procrustes/low-rank)
│   ├── schedule/                           # scheduler.py (QualityDropPredictor + knapsack DP)
│   ├── index/                              # faiss_shard.py (per-shard FAISS build/query)
│   └── eval/                               # metrics.py (Recall/nDCG/MRR/ARR/forgetting/bootstrap)
├── scala/shard-partitioner/                # sbt project: Stage 0 Scala/Spark half (Kafka -> shard assign -> ledger)
│   ├── build.sbt, project/
│   └── src/{main,test}/scala/com/driftdrag/ingest/
├── scripts/
│   ├── env.ps1                             # local-only JDK17/winutils shim for Windows dev (see README)
│   ├── produce_sample_docs.py              # file-source stream producer for the smoke test
│   └── run_smoke_test.py                   # end-to-end local smoke test (Stage 0 -> 1 -> 3)
├── infra/ansible/                          # inventory.ini.example, site.yml -- multi-VM provisioning
├── data/                                   # local data lake root (gitignored; MinIO plays this role on VMs)
├── notebooks/                              # analysis + paper figures (one notebook per figure)
├── results/                                # MLflow/W&B exports, CSVs, seeds
└── tests/                                  # pytest: drift math, ledger ops, scheduler knapsack, sharding parity
```

---

## 7. Key Reference Map (step → citations)

**From your project library (verified):**
- QDC — *Query Drift Compensation…* CoLLAs 2025 → Stages 1, 3; baselines; E2.
- Drift-Adapter — Vejendla, EMNLP 2025 → Stage 3–4; baselines; ARR metric; cost-table methodology.
- DARLA — *Drift-Aware Retrieval-Augmented Log Anomaly Detection* → Stage 2 (MMD), baseline, E7 log track.
- RAGRoute — *Efficient Federated Search for RAG*, EuroMLSys 2025 → Stage 3 router.
- DRAGON / *Distributed RAG* / *Efficient Distributed RAG* → distributed-retrieval related work, latency framing.
- RAGCache — ACM TOCS 2025 → serving-cost methodology, Poisson workload generation.
- FedE4RAG — *Privacy-Preserving Federated Embedding Learning* → related work (federated axis), optional privacy discussion.
- CoRAG — NeurIPS 2025 → optional E7 multi-hop generation layer.
- *Language Drift in Multilingual RAG* → E8 motivation.

**Infrastructure/systems references added for the Big-Data Framework Mapping (Section 1.1):**
- Kreps, Narkhede & Rao, *Kafka: a Distributed Messaging System for Log Processing* (NetDB 2011) → Stage 0 streaming bus.
- Zaharia et al., *Spark: Cluster Computing with Working Sets* (HotCloud 2010) / *Structured Streaming* (SIGMOD 2018) → Stage 0/5 batch+stream processing, both language bindings.
- Carbone et al., *Apache Flink* (IEEE Data Eng. Bull. 2015) → documented alternative low-latency engine for E5.
- Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm* (Raft, USENIX ATC 2014) → rationale for etcd as the stronger-consistency alternative to Redis for the Drift State Ledger at large VM counts.

---

## 8. Multi-VM Deployment & Portability Design

This section is the answer to "how does this move off one machine": the architectural rule is that **no job, script, or config may hardcode a hostname, port, or filesystem path that differs between environments** — every such value is read from `.env` / Hydra's `infra` group, so `docker-compose.yml` + `configs/infra/*.yaml` + `infra/ansible/` are the only three places that change when the deployment topology changes. See [DEPLOYMENT.md](DEPLOYMENT.md) for the operational walkthrough; this section explains why it's structured that way.

### 8.1 Three deployment profiles, one codebase
| Profile | `infra=` | Where jobs run | Stream source | Ledger | Data lake |
|---|---|---|---|---|---|
| Local smoke test | `local` | `local[*]` in-process | JSON files under `data/stream_in/` | `fakeredis` (in-memory) | local disk |
| Single-VM dev | `docker` | `docker-compose.yml` (Kafka, Spark master+worker, Redis, MinIO, MLflow) | Kafka topic | Redis container | MinIO (`s3a://`) |
| Multi-VM (E5/E6 scale) | `cluster` | `docker-compose.yml` + `docker-compose.cluster.yml`, one container group per VM | Kafka topic (multi-broker) | Redis (or etcd) on a dedicated VM | MinIO/S3 on a dedicated VM |

Moving between rows changes exactly one thing per job invocation: which `configs/infra/*.yaml` is selected (`+infra=cluster` on the Hydra command line) and the values in `.env` — never the job code in `src/driftdrag/` or `scala/shard-partitioner/`.

### 8.2 VM roles for the multi-VM profile
`docker-compose.cluster.yml` tags every service with a Compose `profile` matching one of these roles, so a given VM starts only its own containers with `docker compose -f docker-compose.yml -f docker-compose.cluster.yml --profile <role> up`:

| Role | Services | Typical count |
|---|---|---|
| `broker` | Kafka (KRaft) | 1 (3+ for a real multi-broker cluster, out of scope for this scaffold) |
| `spark_master` | Spark master | 1 |
| `spark_worker` | Spark worker | N (scale with `docker compose up --scale spark-worker=N`, or one per worker VM) |
| `storage` | Redis, MinIO, MLflow | 1 (split further if Redis/MinIO need independent scaling) |
| `client` | nothing containerized — this is wherever `spark-submit` / the PySpark driver / the Scala jar is launched from | 1+ |

(Role names match `infra/ansible/inventory.ini.example` group names exactly, since `site.yml` passes each host's group names straight through as `--profile` values — Ansible sanitizes hyphens out of group names, so both sides use underscores.)

`network_mode: host` is used on every service in the cluster overlay because cross-VM traffic cannot ride Docker's internal bridge network — this is why the overlay is Linux-only (matches the target: VMs, not a Docker-Desktop laptop).

### 8.3 What has to be externalized for this to work at all
Three specific things break a "just copy the folder to another VM" story if they are allowed to default to local state; the codebase keeps all three externalized from day one so scaling out is additive, not a rewrite:
1. **The Drift State Ledger** (`ledger/drift_ledger.py`) — never a local dict/file; always injected as a Redis client, so a dedicated ledger VM is a config change, not a code change (`connect(host=..., port=..., use_fake=False)`).
2. **The data lake root** (`lake.root` in `configs/infra/*.yaml`) — `data/lake` locally, `s3a://driftdrag-lake` on MinIO/S3 once there's more than one VM writing to it, since two Spark executors on two different machines cannot share a local disk.
3. **Shard assignment** (`ingest/sharding.py` / `ShardPartitioning.scala`) — a pure function of `doc_id`, not of which machine/partition happened to receive the message, so the same document lands on the same shard no matter which VM's Kafka partition or Spark executor processed it (see the CRC32 cross-language note in both files).

### 8.4 Rollout order (pilot → 100M-scale multi-VM)
1. Local smoke test (this scaffold, `scripts/run_smoke_test.py`) — proves the logic before spending a single GPU-hour or provisioning a single VM.
2. Single-VM `docker-compose.yml` — proves the same logic against real Kafka/Redis/MinIO/Spark-standalone, still with one worker.
3. `infra/ansible/` provisions N VMs with Docker installed and the repo cloned; `docker-compose.cluster.yml --profile <role>` starts each VM's slice (see DEPLOYMENT.md for the exact command sequence and inventory format).
4. Scale `spark-worker` count and shard count together as corpus size grows from the 10M pilot to the 100M headline run (E1/E6); `NUM_SPARK_EXECUTORS` and `num_shards` are both plain config values, swept the same way Hydra sweeps any other experiment axis.
5. Swap FAISS-per-shard for `docker-compose.milvus.yml` (Milvus standalone) only once a single VM's FAISS indexes stop fitting in memory — this is a Stage-3 config flag (`index.backend: faiss|milvus`), not a pipeline rewrite.
