# Cost budget: fitting this on $300 trial credits

Prices below are on-demand estimates for `asia-south1` (Mumbai) at the time of writing — check the [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for current numbers before a long run. The point of this document is the **shape** of the spend (what's persistent vs. ephemeral), not exact cents.

## Per-component cost and how long it should run

| Component | Machine | ~$/hour | Should run... | Why |
|---|---|---|---|---|
| Kafka VM | e2-medium | ~$0.04 | Persistent, but **stop it (not delete) between work sessions** | Cheap enough to leave up during active development; stopping (not deleting) preserves its disk so Kafka's log data survives |
| Redis VM | e2-medium | ~$0.04 | Same as Kafka | Drift Ledger state should survive between sessions |
| Airflow VM | e2-medium | ~$0.04 | Same as Kafka | DAG definitions/history persist across stop/start |
| Ray head (+ worker if added) | e2-standard-4 | ~$0.15 | **Start only when running a Ray job, stop after** | No state worth keeping between runs |
| Dataproc cluster | 1 master + 2 workers, n2-standard-4 | ~$0.65 combined | **Ephemeral: create -> submit job(s) -> delete, every time** (see `05_dataproc/`) | This is the single biggest lever — a Dataproc cluster left running 24/7 would burn ~$470/month on its own |
| GPU VM | n1-standard-4 + 1x T4 | ~$0.60 | **Ephemeral: create -> run job -> delete, every time** (see `08_gpu_embedding/`) | GPUs are the most expensive resource per hour available to this project; never leave one idle |
| GCS storage | Standard class | ~$0.02/GB/month | Persistent (it's just storage, not compute) | Cheapest place to leave data sitting between sessions |

## Two spending patterns

**Disciplined (recommended):** Kafka/Redis/Airflow VMs stopped whenever you're not actively working (a `gcloud compute instances stop` costs nothing while stopped, only the ~$0.50/month for the boot disk); Dataproc and the GPU VM only exist for the duration of a single job. Under this pattern, a full week of active daily work (~4 hrs/day) costs roughly:
- Kafka+Redis+Airflow: 3 x $0.04 x 28 hrs ≈ $3.40
- Ray: $0.15 x 10 hrs ≈ $1.50
- Dataproc: $0.65 x 5 runs x 1.5 hrs/run ≈ $4.90
- GPU: $0.60 x 5 runs x 1 hr/run ≈ $3.00
- **≈ $13/week** — the $300 credit lasts well over a month of active experimentation.

**Undisciplined (avoid):** everything left running 24/7, including the GPU VM. That alone is ~$0.60 x 730 ≈ **$438/month** — more than the entire trial credit in three weeks, on the GPU line item alone.

## The one rule that matters

> **After every work session, run `scripts/status.sh` to see what's running, and `scripts/teardown_all.sh` (or at minimum stop the GPU VM and delete the Dataproc cluster) before you walk away.**

`scripts/teardown_all.sh` (built in this directory) deletes the Dataproc cluster if present, deletes the GPU VM if present, and stops (not deletes) the Kafka/Redis/Airflow/Ray VMs. Re-running the setup scripts for Dataproc/GPU is cheap and fast (a few minutes); leaving a GPU VM up overnight by accident is the failure mode this whole document exists to prevent.

## Monitoring actual spend

```bash
# Check current month-to-date cost (requires Cloud Billing Budget API / BigQuery export
# to be set up once -- see https://cloud.google.com/billing/docs/how-to/export-data-bigquery)
gcloud billing accounts list
gcloud alpha billing budgets list --billing-account=<YOUR_BILLING_ACCOUNT_ID>
```

Simplest option for a trial account: set a **budget alert** once, in the console, at `Billing -> Budgets & alerts -> Create budget`, scoped to the `drift-rag` project, at e.g. $50/$100/$200/$250 thresholds — this emails you before you're anywhere near the $300 ceiling, independent of anything in this repo.
