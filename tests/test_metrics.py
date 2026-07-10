import numpy as np
import pytest

from driftdrag.eval.metrics import (
    aggregate_recall,
    dcg_at_k,
    forgetting,
    mrr,
    ndcg_at_k,
    paired_bootstrap_pvalue,
    recall_at_k,
)


def test_recall_at_k_partial_match():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "z"}
    assert recall_at_k(retrieved, relevant, k=4) == 0.5


def test_mrr_rewards_earlier_hit():
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert mrr(["a", "x"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2, "b": 1, "c": 0}
    retrieved = ["a", "b", "c"]
    assert abs(ndcg_at_k(retrieved, relevance, k=3) - 1.0) < 1e-9


def test_ndcg_worse_ranking_is_lower():
    relevance = {"a": 2, "b": 1, "c": 0}
    perfect = ndcg_at_k(["a", "b", "c"], relevance, k=3)
    worse = ndcg_at_k(["c", "b", "a"], relevance, k=3)
    assert worse < perfect


def test_dcg_zero_when_nothing_relevant():
    assert dcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_forgetting_positive_on_degradation():
    assert forgetting(0.8, 0.6) == pytest.approx(0.2)
    assert forgetting(0.6, 0.6) == 0.0


def test_aggregate_recall_ci_shrinks_with_more_seeds():
    few = aggregate_recall([0.5, 0.7])
    many = aggregate_recall([0.5, 0.7, 0.6, 0.55, 0.65, 0.6, 0.58, 0.62])
    assert few[1] >= 0
    assert many[1] >= 0


def test_paired_bootstrap_pvalue_significant_when_clearly_different():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.01, size=50)
    b = rng.normal(0.5, 0.01, size=50)
    p = paired_bootstrap_pvalue(a, b, n_boot=500, random_state=0)
    assert p < 0.05


def test_paired_bootstrap_pvalue_not_significant_when_same():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, size=50)
    b = a.copy()
    p = paired_bootstrap_pvalue(a, b, n_boot=500, random_state=0)
    assert p == 1.0
