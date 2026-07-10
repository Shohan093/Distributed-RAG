"""Budgeted re-embedding scheduler (Stage 4) -- the decision-theoretic core.

Per maintenance epoch, given a compute budget B (GPU-hours), choose per
shard among {compensate (cost ~= 0), partial re-embed (hot clusters only),
full re-embed} to maximize predicted retrieval quality. Two pieces:

1. QualityDropPredictor: a lightweight regressor mapping the two cheap
   per-shard signals (anchor residual, data-drift score) to a predicted
   recall drop (this is what E3 evaluates directly -- its R²/Spearman rho
   *is* the justification for trusting the scheduler below).
2. schedule_shards: a multiple-choice-knapsack scheduler that spends the
   budget on the actions with the best marginal (quality recovered / cost)
   ratio first. This is a greedy heuristic, not the exact Lagrangian-
   relaxed optimum described in the plan; it is provably optimal for the
   LP relaxation and is the baseline the exact ILP/Lagrangian solver
   (future work) must beat to justify its extra complexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QualityDropPredictor:
    """Predicts expected recall drop from (anchor_residual, data_drift_score)."""

    def __init__(self, model=None):
        if model is None:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
        self.model = model
        self._fitted = False

    def fit(self, anchor_residuals: np.ndarray, data_drift_scores: np.ndarray, recall_drops: np.ndarray) -> "QualityDropPredictor":
        x = np.column_stack([anchor_residuals, data_drift_scores])
        self.model.fit(x, recall_drops)
        self._fitted = True
        return self

    def predict(self, anchor_residuals: np.ndarray, data_drift_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("QualityDropPredictor.fit() must be called before predict()")
        x = np.column_stack([anchor_residuals, data_drift_scores])
        return self.model.predict(x)


@dataclass
class ShardCandidate:
    shard_id: str
    predicted_drop_compensate: float
    predicted_drop_partial: float
    predicted_drop_full: float
    cost_partial: float
    cost_full: float


@dataclass
class ScheduleDecision:
    shard_id: str
    action: str
    cost: float
    quality_recovered: float


def schedule_shards(candidates: list[ShardCandidate], budget: float, grid_points: int = 500) -> list[ScheduleDecision]:
    """Multiple-choice-knapsack scheduler, solved exactly by dynamic
    programming over a discretized budget axis.

    Each shard offers exactly one of three *mutually exclusive* actions per
    epoch: compensate (cost 0), partial re-embed (cost_partial), or full
    re-embed (cost_full); quality recovered is the predicted-drop reduction
    relative to compensate-only. This is the standard 0/1 multiple-choice
    knapsack (one item picked per group); ``grid_points`` discretizes the
    continuous GPU-hour budget so the DP stays O(n_shards * grid_points).
    Exact on the grid, so it strictly dominates a greedy ratio heuristic and
    is itself the baseline an ILP/Lagrangian-relaxed solver (future work)
    would need to beat to justify its extra complexity.
    """
    if budget <= 0 or not candidates:
        return [ScheduleDecision(c.shard_id, "compensate", 0.0, 0.0) for c in candidates]

    scale = grid_points / budget
    options_by_shard = []
    for c in candidates:
        opts = [("compensate", 0.0, 0.0)]
        recovered_partial = c.predicted_drop_compensate - c.predicted_drop_partial
        if c.cost_partial >= 0 and recovered_partial > 0:
            opts.append(("partial", c.cost_partial, recovered_partial))
        recovered_full = c.predicted_drop_compensate - c.predicted_drop_full
        if c.cost_full >= 0 and recovered_full > 0:
            opts.append(("full", c.cost_full, recovered_full))
        options_by_shard.append((c.shard_id, opts))

    dp = np.zeros(grid_points + 1)
    choice = [[0] * (grid_points + 1) for _ in options_by_shard]

    for i, (_, opts) in enumerate(options_by_shard):
        new_dp = dp.copy()
        for b in range(grid_points + 1):
            best_val, best_opt = dp[b], 0
            for opt_idx, (_, cost, recovered) in enumerate(opts):
                cost_grid = int(round(cost * scale))
                if cost_grid <= b:
                    val = dp[b - cost_grid] + recovered
                    if val > best_val:
                        best_val, best_opt = val, opt_idx
            new_dp[b] = best_val
            choice[i][b] = best_opt
        dp = new_dp

    decisions = []
    b = grid_points
    for i in range(len(options_by_shard) - 1, -1, -1):
        shard_id, opts = options_by_shard[i]
        opt_idx = choice[i][b]
        action, cost, recovered = opts[opt_idx]
        decisions.append(ScheduleDecision(shard_id, action, cost, recovered))
        b -= int(round(cost * scale))
        b = max(b, 0)
    decisions.reverse()
    return decisions


def total_cost(decisions: list[ScheduleDecision]) -> float:
    return sum(d.cost for d in decisions)


def total_quality_recovered(decisions: list[ScheduleDecision]) -> float:
    return sum(d.quality_recovered for d in decisions)
