"""Retrieval-quality and continual-learning metrics shared across E1-E7.

Kept dependency-free (numpy only) so the same functions run inside a PySpark
UDF at 100M-vector scale (E6) or a two-line unit test.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant)


def mrr(retrieved: list, relevant: set) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list, relevance: dict, k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevance.get(doc_id, 0)
        if rel:
            score += (2 ** rel - 1) / np.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list, relevance: dict, k: int) -> float:
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_order, start=1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevance, k) / idcg


def forgetting(quality_at_index_time: float, quality_now: float) -> float:
    """Continual-learning-style forgetting measure (Chaudhry et al.):
    positive => quality degraded since the shard was indexed.
    """
    return quality_at_index_time - quality_now


def aggregate_recall(per_query_recalls: list[float]) -> tuple[float, float]:
    """Returns (mean, 95%-CI half-width) via normal approximation, used for
    the "mean +/- 95% CI" reporting the plan's statistical-rigor section
    requires across >=3 seeds.
    """
    arr = np.asarray(per_query_recalls, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    sem = arr.std(ddof=1) / np.sqrt(len(arr))
    return mean, float(1.96 * sem)


def paired_bootstrap_pvalue(
    scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 2000, random_state: int = 0
) -> float:
    """Paired bootstrap test for "method A beats method B" (E-series baseline
    comparisons), returning a two-sided p-value.
    """
    rng = np.random.default_rng(random_state)
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    observed = diffs.mean()
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    centered = boot_means - boot_means.mean()
    p = float(np.mean(np.abs(centered) >= np.abs(observed)))
    return p
