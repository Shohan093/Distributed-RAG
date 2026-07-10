"""Stage 4 (budgeted re-embedding scheduler), reading the per-shard signals
that parallel_drift_detection.py wrote to the ledger.

The scheduler's knapsack DP itself (driftdrag.schedule.scheduler.schedule_shards)
is a single small optimization over all shards -- not worth parallelizing --
so this script does NOT use ray.remote; it just runs on the Ray head node
after the two parallel jobs above have populated the ledger, closing the
loop from Stage 2 -> Stage 4 the same way PLAN.md Section 2 describes.

Cost/recall numbers here (cost_partial/cost_full, predicted_drop_partial/full)
are placeholders proportional to shard size -- replace with your fitted
QualityDropPredictor once you have (signal, measured recall-drop) pairs
from E1/E2 (see PLAN.md E3).
"""

from __future__ import annotations

import os
from pathlib import Path

DRIFTDRAG_PKG = str(Path.home() / "driftdrag_pkg" / "driftdrag")
REDIS_HOST = os.environ["REDIS_HOST"]
BUDGET_GPU_HOURS = float(os.environ.get("BUDGET_GPU_HOURS", "2.0"))

import sys  # noqa: E402

sys.path.insert(0, DRIFTDRAG_PKG.rsplit("/", 1)[0])

from driftdrag.ledger.drift_ledger import connect  # noqa: E402
from driftdrag.schedule.scheduler import ShardCandidate, schedule_shards, total_cost, total_quality_recovered  # noqa: E402


def main() -> None:
    client = connect(host=REDIS_HOST, port=6379)
    keys = client.keys("ledger:shard-*")
    candidates = []
    for key in keys:
        blob = client.get(key)
        if blob is None:
            continue
        import json

        record = json.loads(blob)
        residual = record.get("anchor_residual")
        drift_score = record.get("data_drift_score")
        if residual is None or drift_score is None:
            continue
        shard_id = record["shard_id"]
        # Placeholder cost/predicted-drop model -- see docstring above.
        candidates.append(
            ShardCandidate(
                shard_id=shard_id,
                predicted_drop_compensate=residual + drift_score,
                predicted_drop_partial=(residual + drift_score) * 0.4,
                predicted_drop_full=0.0,
                cost_partial=0.5,
                cost_full=1.0,
            )
        )

    if not candidates:
        print("No shards with drift scores in the ledger yet -- run parallel_drift_detection.py first.")
        return

    decisions = schedule_shards(candidates, budget=BUDGET_GPU_HOURS)
    for d in decisions:
        print(f"shard {d.shard_id}: action={d.action} cost={d.cost:.2f} recovered={d.quality_recovered:.4f}")
    print(f"\nBudget: {BUDGET_GPU_HOURS} GPU-hours | spent: {total_cost(decisions):.2f} | "
          f"quality recovered: {total_quality_recovered(decisions):.4f}")


if __name__ == "__main__":
    main()
