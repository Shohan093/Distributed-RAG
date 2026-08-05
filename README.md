# Drift-RAG: Drift-Compensated Distributed RAG for Streaming Analytics

A research project for the Big Data Analytics course investigating whether **lightweight query-side compensation can recover retrieval quality when a RAG system's embedding model is upgraded**, without re-embedding the historical corpus.

This document summarizes what was built, why, and what we found — meant as an onboarding reference, not a final report.

---

## 1. The Problem We're Solving

RAG systems ingest documents continuously. Embedding models improve over time (BGE v1 → v1.5 → m3, for example). When you upgrade the embedding model:

- Old documents are indexed in the *old* embedding space.
- New queries are encoded in the *new* embedding space.
- Query and document vectors no longer align → retrieval quality collapses.

The naive fix is to re-embed the entire historical corpus on every model upgrade. At scale, this is expensive and often impractical.

**Our idea:** instead of re-embedding millions of documents, transform the *query* at search time so it lands back in each historical index's original space. Each shard (one per historical "generation") owns its own small transformation ("adapter") and applies it locally before searching its own index.

---

## 2. Architecture

### 2.1 Two phases

**Offline pipeline (run once):** download → clean → split into 3 chronological "generations" → embed each generation with its own model → build a small shared calibration set → train adapters → build FAISS indexes.

**Live deployment (5 VMs):**

```
                    Router VM (worker1)
                 Encodes query with V3, fuses results
                              │
              ┌───────────────┼───────────────┐
              ▼                ▼                ▼
        Shard VM (worker2)  Shard VM (worker3)  Shard VM (worker4)
        Epoch 1, V1 adapter Epoch 2, V2 adapter Epoch 3, native V3
              │                │                │
        (metrics/logs, not query path)
              └────────────────┼────────────────┘
                              ▼
                  Orchestrator VM (worker5)
                Training, calibration, evaluation
```

Each shard receives the **raw V3 query embedding** from the router, applies its **own local adapter** (or none, if it's the native V3 generation), and searches its **own FAISS index**. The router never knows about adapters — it only encodes and fuses. This is a deliberate design decision: adaptation is a shard-owned responsibility, making each shard closer to an autonomous retrieval service.

### 2.2 Why this stack

- **Scala / Apache Spark** — corpus ingestion, cleaning, deduplication, chronological splitting. Chosen because the course explicitly wanted a Big Data / distributed-systems component, not just an ML pipeline.
- **Python** — all ML: embedding, adapter training, FAISS indexing, retrieval services.
- **FastAPI + uvicorn** — turns the router and shard logic into real, independently deployable HTTP services.
- **FAISS** — vector indexing (both flat/exact and IVF/approximate, built and compared).

---

## 3. Data

**Corpus:** [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) (708,241 real news articles, Jan 2017–2019).

**Chronological split, not arbitrary:** we sorted all articles by real publish date and cut the first 300,000 (chronologically) into three equal 100,000-article windows — deliberately avoiding a sparse, low-density tail we discovered in the raw data (article density collapses after ~mid-2018). Final epoch boundaries (after cleaning) all fall within 2017.

**Cleaning pipeline (Spark):**
- Strip AP/wire transmission artifacts (`^By AUTHOR=` style markers).
- Drop degenerate stub articles (<200 characters).
- Deduplicate exact `(title, text)` matches (~7–8% of each epoch).
- Truncate to ≤400 words **and** ≤3000 characters — the character cap was added after we discovered a document containing 15,000+ characters of leaked JSON metadata (no whitespace, so word-count truncation didn't catch it) that was crashing the embedding job on OOM.

**Final corpus sizes:** Epoch 1 = 88,209 docs · Epoch 2 = 86,577 docs · Epoch 3 = 85,254 docs.

---

## 4. Embedding Models (the three "generations")

| Role | Model | Dim | Released | Query instruction |
|---|---|---|---|---|
| V1 | `BAAI/bge-large-en` | 1024 | Aug 2023 | Required |
| V2 | `BAAI/bge-large-en-v1.5` | 1024 | Sep 2023 | Optional (works fine either way) |
| V3 | `BAAI/bge-m3` | 1024 | Jan 2024 | Not needed (dense output only used) |

Real BAAI releases, same architecture lineage, all 1024-dimensional (no projection needed between them) — chosen deliberately because they represent a genuine "same-family successor" evolution, matching how a real organization would actually upgrade its embedding model.

Each epoch is embedded **only by its own designated model**: Epoch 1 → V1, Epoch 2 → V2, Epoch 3 → V3 (the current, "live" model).

---

## 5. Calibration Set & Adapters

### 5.1 Calibration set

A **shared calibration set** — not separate per model pair — so every adapter is trained on exactly the same reference documents, enabling clean cross-comparisons.

Per epoch: **2,333 stratified anchor documents** (1,333 k-means centroid representatives + 500 random + 500 farthest-from-centroid outliers), embedded by **all three models**. Total: 6,999 documents × 3 models.

This means only ~1/37th of the full corpus ever needs re-embedding per model transition — the actual cost basis of the whole approach (see §7.4).

### 5.2 Three adapter designs, tested as an ablation

- **L0 — global translation.** One vector, `mean(target) − mean(V3)`, added to every query. Cheapest possible approach.
- **L1 — global rotation.** Orthogonal Procrustes fit on the calibration set: `argmin ||V3_calib @ R − target_calib||`.
- **L2 — rotation + local correction.** L1's rotation, plus a small per-cluster residual bias (20 clusters in V3 space) on top.

---

## 6. Evaluation Methodology

Two separate evaluation setups were used, deliberately:

**NFCorpus (BEIR benchmark)** — used early, *before* building any real infrastructure, to cheaply validate the core hypothesis with real human relevance judgments. Small (3,128 docs, 323 queries), fast to iterate on.

**Real CC-News eval set** — since CC-News has no existing relevance judgments, we built our own: **title-as-query, body-as-document self-retrieval**. For each held-out article (excluded from the calibration set), the title becomes a synthetic query and the body (title stripped out, to avoid trivial lexical matching) becomes the document to retrieve. Ground truth is unambiguous — a query's own document is the one correct answer. 5,000 index docs + 500 queries per epoch.

For every comparison, we measure three conditions:
- **Naive** — raw V3 query against the historical index, no compensation (the "do nothing" baseline).
- **Compensated** — V3 query + adapter, against the historical index.
- **Oracle** — native-generation query against native-generation index (the upper-bound ceiling; what full re-embedding would buy you).

---

## 7. Key Results

### 7.1 Core hypothesis: does compensation work?

| Dataset | Naive NDCG@10 | L1 NDCG@10 | Oracle NDCG@10 | Recovery |
|---|---|---|---|---|
| NFCorpus (V3→V1) | ~0.011 | ~0.276 | ~0.359 | ~77% |
| NFCorpus (V3→V2) | ~0.010 | ~0.280 | ~0.400 | ~70% |
| CC-News (V3→V1) | 0.000 | 0.772 | 0.791 | **97.6%** |
| CC-News (V3→V2) | 0.001 | 0.810 | 0.866 | **93.5%** |

**Naive collapses to near-zero in every single test** — doing nothing when the embedding model changes doesn't just hurt retrieval, it destroys it almost completely. This is the strongest "why this project matters" evidence.

**Compensation recovers the large majority of oracle-level quality**, on two independent datasets. The CC-News numbers are higher than NFCorpus's — expected, since the CC-News eval task (title→body self-retrieval) is structurally easier than NFCorpus's graded, human-judged relevance task. Both point the same direction.

**A consistent, replicated pattern across both datasets:** the older transition (V3→V1) recovers a *higher* percentage than the more recent one (V3→V2) — seen independently on NFCorpus and CC-News. Not yet fully explained; flagged as a genuine open finding.

### 7.2 L0 vs. L1 vs. L2 ablation

| Method | Epoch 1 NDCG@10 | Epoch 2 NDCG@10 |
|---|---|---|
| L0 (translation only) | 0.0007 | 0.0000 |
| **L1 (rotation)** | **0.7722** | **0.8095** |
| L2 (rotation + local bias) | 0.7705 | 0.7988 |
| Oracle | 0.7910 | 0.8662 |

**L0 fails almost completely** — a bare translation is nearly as large in magnitude as the distance between two *unrelated* document vectors, so it provides no real signal. This tells us the drift between BGE generations is fundamentally **rotational, not positional**.

**L2 does not beat L1** — if anything, marginally worse, likely due to overfitting on the modest ~117 documents per cluster. **The simple global rotation (L1) already captures essentially all the recoverable signal.** A clean, honest negative result: added complexity didn't help.

### 7.3 Flat vs. IVF FAISS index comparison

| | NDCG@10 (avg) | Query latency (avg) |
|---|---|---|
| Flat (exact) | ~0.82 | ~1.3ms |
| IVF (approximate, nprobe=16) | ~0.74 | ~0.17ms |

IVF is **5–13x faster** but costs **~8–12% relative retrieval quality**. At our current corpus scale (~85k docs/shard), flat search is *already* fast enough in absolute terms (sub-2ms) that the practical case for IVF is weaker than the raw speedup number alone suggests — the tradeoff only becomes compelling at much larger scale.

### 7.4 Cost baseline: why not just re-embed everything?

| | Time per model transition |
|---|---|
| Full corpus re-embed | 6.8 – 9.3 hours |
| Calibration-only re-embed (our approach) | 11 – 15 minutes |

**~37x fewer documents need re-embedding** with the compensation approach — turning a multi-hour job into a coffee break, for 93–98% of the retrieval quality.

### 7.5 Distributional drift measurement (MMD)

Beyond retrieval-quality recovery, we independently measured **how much the embedding distributions themselves drift**, using Maximum Mean Discrepancy (MMD) with a Gaussian RBF kernel — a technique borrowed explicitly from the DARLA paper's drift-detection methodology (see §9), applied here to a different problem (representation-space drift, not data-distribution drift).

| Transition | MMD² before compensation | MMD² after compensation | Reduction |
|---|---|---|---|
| V3 → V1 | 0.3964 | 0.0689 | **82.6%** |
| V3 → V2 | 0.2625 | 0.0029 | **98.9%** |

This gives a second, independent line of evidence for the core claim: compensation substantially reduces the statistical distance between the query-generation's embedding distribution and the historical index's distribution.

**A genuinely interesting discrepancy, worth stating honestly rather than smoothing over:** MMD reduction and NDCG-based retrieval recovery **do not rank the two transitions the same way**. V2 shows a much larger MMD reduction (98.9% vs. 82.6%) than V1, but V1 recovers *more* retrieval quality than V2 in every measurement we took (NFCorpus and CC-News both). Our interpretation: MMD captures **global distributional alignment** (do the two embedding clouds look statistically similar overall), while NDCG@10 captures **local, rank-sensitive alignment** (do individual query-document pairs land close enough, relative to other candidates, to rank correctly). These are related but distinct notions of "how well compensation worked" — a rotation can align two distributions' overall shape very well while still leaving small individual-pair misalignments that matter for fine-grained ranking, or vice versa. Reporting both metrics, including where they disagree, is more honest than reporting only the one that tells a cleaner story.

### 7.6 End-to-end system latency (real, distributed, network-measured)

A genuine bug was found and fixed here: the router originally called its three shards **sequentially**, not in parallel. Fixing this (via a thread pool) dropped mean end-to-end query latency from **331ms → ~171–242ms**.

Full breakdown of a real distributed query: **query encoding (~65–90ms, CPU-bound `bge-m3` inference) and parallel shard search (~80ms, network + FAISS + adapter combined) are roughly comparable in cost** — fusion itself is free (~0.1ms). A genuinely non-obvious finding: the single-machine encoding step costs about as much as coordinating three networked shards.

---

## 8. What's Actually Implemented (Verified Working)

- Full Spark ingestion pipeline (download → clean → dedupe → truncate → chronological split), verified against real data.
- Config-driven Python embedding pipeline (YAML-driven, checkpointed/resumable — survives OOM kills without losing progress).
- Real embeddings for all 3 epochs × their native model, verified shapes/IDs.
- Shared calibration set, embedded by all 3 models, ID-alignment cross-verified.
- L0, L1, L2 adapters, all trained and evaluated.
- Flat + IVF FAISS indexes, all 3 epochs.
- A **genuine, working distributed system**: FastAPI router (worker1) + 3 FastAPI shard services (worker2–4), communicating over real inter-VM HTTP, with RRF fusion — tested and confirmed with real network queries.
- Full evaluation suite: naive/oracle/L0/L1/L2 ablation, flat-vs-IVF comparison, cost baseline, end-to-end latency benchmark — all with saved JSON results and PNG figures under `results/`.

## 9. What's Explicitly Out of Scope / Not Built

- **Generation** — the system returns ranked, retrieved documents; there is no LLM call synthesizing an answer. This is a retrieval system with an intentionally unimplemented "optional" generation stage, per the original design.
- **Full re-implementation of DARLA** — deliberately not attempted. DARLA (Cai et al.) is a supervised log-anomaly-detection system built around a **fixed, unchanging encoder**; it detects and adapts to *data* distribution drift (new log templates, workload shifts), not *representation-space* drift from an encoder upgrade. Different task (binary classification vs. our unsupervised retrieval), different domain (structured system logs vs. news text), different metric (F1 vs. NDCG@10) — there is no natural task to run a head-to-head comparison on without inventing a different system that no longer resembles the paper. Cited as related work instead, with its actual mechanism precisely characterized (see below) rather than left vague.

## 10. No Off-the-Shelf RAG Framework

Everything (router, shard, adapters, fusion, FAISS wiring, HTTP services) was hand-built — no LangChain, LangGraph, or LlamaIndex. This was a deliberate choice: the project's contribution is a novel architecture (per-shard adaptation across historical embedding spaces) that doesn't map cleanly onto what those frameworks assume (one current index, no cross-generation compensation), and hand-rolling it keeps the actual mechanism under study fully visible and measurable.

---

## 11. Positioning Against Prior Work (QDC, DARLA)

**QDC (Query Drift Compensation, Goswami et al., arXiv 2506.00037)** — QDC's core mechanism is a **single global drift vector** per model transition: `δ = q_new − q_old`, applied as a translation to the query. This is mathematically the same category of transformation as our **L0 baseline** (a global mean-shift), just estimated differently — QDC approximates the vector from current-task query drift (since their continual-learning setting can't access old-task data directly), while we compute it exactly from paired calibration embeddings (a genuine advantage of our setup). **Our L0 ablation result (NDCG@10 ≈ 0, near-total failure) is therefore a fair, arguably favorable-conditions proxy for QDC's core mechanism applied to BGE-generation drift** — and it fails, while our rotation-based L1 recovers 93–98%. This is presented in the report as: *QDC's translation-only mechanism does not work for this drift pattern; the drift is rotational, not positional (see §7.2).*

**DARLA (Cai et al.)** — not a retrieval-compensation method at all; it's a drift-*detection*-triggered knowledge-base refresh system for log anomaly detection, under a fixed encoder. Not comparable as a baseline (see §9). However, its **drift-detection mechanism (MMD over embeddings)** is directly reusable as a diagnostic technique, independent of its original application — and we did reuse it (§7.5) to independently quantify how much representation drift our adapters remove. This is an honest, precise use of prior work: borrowing a *measurement technique*, not claiming a *baseline comparison*.

## 12. Repository Layout

```
drift-rag/
├── spark/                          # Scala/Spark ingestion pipeline
├── python/
│   ├── embeddings/                 # corpus + calibration embedding scripts
│   ├── adapters/                   # L0/L1/L2 training
│   ├── retrieval/                  # router, shard, FAISS index building, HTTP services
│   ├── evaluation/                 # adapter eval, index comparison, cost baseline, latency               
│   └── utils/                      # shared config/embedding helpers
├── configs/embedding.yaml          # single source of truth: models, paths, epochs
├── scripts/                        # deploy_code.sh, deploy_data.sh, download_ccnews.sh
├── data/                           # raw + cleaned corpus (per epoch)
├── embeddings/                     # corpus + calibration embeddings (per epoch/model)
├── indexes/                        # flat + ivf FAISS indexes (per epoch)
└──checkpoints/calibration_ids/    # trained L0/L1/L2 adapter weights
 
```

## 13. VM Roles Quick Reference

| VM | Role | What it holds |
|---|---|---|
| worker1 | Router | V3 encoder, fusion logic (FastAPI service) |
| worker2 | Shard, Epoch 1 | V1 embeddings, V1 adapter, flat+ivf index (FastAPI service) |
| worker3 | Shard, Epoch 2 | V2 embeddings, V2 adapter, flat+ivf index (FastAPI service) |
| worker4 | Shard, Epoch 3 | V3 embeddings, no adapter (native), flat+ivf index (FastAPI service) |
| worker5 | Orchestrator | All data, all embeddings, all adapters, all evaluation results |

All VMs share a single `.master` Python virtual environment (`~/.master`) to save disk space across projects. Dependencies are recorded in `python/requirements.txt`.

## 14. How to Reproduce

```bash
# 1. Sync code to all VMs
./scripts/deploy_code.sh

# 2. Ingest & clean data (Scala/Spark, run locally, syncs output to VMs)
cd spark && sbt package && cd ..
spark-submit --class IngestionApp --master "local[*]" spark/target/out/jvm/scala-2.12.18/drift-rag-spark/drift-rag-spark_2.12-0.1.0.jar

# 3. Embed each epoch on its designated shard VM (parallel, checkpointed)
ssh worker2 "cd ~/drift-rag/python && ../scripts/resilient_run.sh 1"
ssh worker3 "cd ~/drift-rag/python && ../scripts/resilient_run.sh 2"
ssh worker4 "cd ~/drift-rag/python && ../scripts/resilient_run.sh 3"

# 4. Build calibration set, train adapters, build indexes -- on worker5 / shards respectively
# (see scripts/run_calibration_pipeline.sh)

# 5. Launch the live system
ssh worker2/3/4: uvicorn retrieval.shard_service:app --host 0.0.0.0 --port 8000
ssh worker1:     uvicorn retrieval.router_service:app --host 0.0.0.0 --port 8000

# 6. Query it
curl -X POST http://<worker1-ip>:8000/query -d '{"query": "your search text", "k": 10}'
```

---

## 15. Honest Notes for Whoever Reads This Next

- All numbers in `results/tables/*.json` are reproducible from the pipeline above, but the git-commit provenance tagging currently only works when run from a machine with the full `.git` history (worker5's copy doesn't have one — this is a known, minor gap).
- The NFCorpus and CC-News recovery ratios are **not directly comparable numbers** — different evaluation tasks, similar direction. Don't present them as the same metric without the caveat.
- The V1-recovers-better-than-V2 pattern is real and replicated twice on retrieval quality (NFCorpus and CC-News), but the MMD drift-reduction numbers rank the two transitions the *opposite* way (§7.5). Not yet fully explained — present as an open, honest observation, not a resolved finding.
- The MMD drift measurement used a fixed 1,000-document random subsample (seed=42) for computational tractability. Worth a re-run with a different seed before treating the exact percentages as final, to confirm they aren't a sampling artifact.
- Everything under `results/` was generated on real infrastructure, with real network calls — none of it is simulated or extrapolated except the cost baseline's full-corpus timing (which is a measured-throughput linear extrapolation, stated as such).