import numpy as np

from driftdrag.detect.anchor_residual import anchor_residual, per_cluster_anchor_residual, select_anchor_set


def test_anchor_residual_zero_when_compensation_perfect():
    rng = np.random.default_rng(0)
    old = rng.normal(size=(50, 6))
    assert anchor_residual(old.copy(), old) == 0.0


def test_anchor_residual_positive_when_uncompensated():
    rng = np.random.default_rng(0)
    old = rng.normal(size=(50, 6))
    new_uncompensated = old + 2.0
    assert anchor_residual(new_uncompensated, old) > 1.0


def test_per_cluster_anchor_residual_isolates_bad_cluster():
    rng = np.random.default_rng(0)
    old = rng.normal(size=(100, 4))
    labels = np.array([0] * 50 + [1] * 50)
    compensated = old.copy()
    compensated[50:] += 3.0  # cluster 1 compensation fails
    residuals = per_cluster_anchor_residual(compensated, old, labels)
    assert residuals[0] < 0.1
    assert residuals[1] > 2.0


def test_select_anchor_set_returns_unique_in_range_indices():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(300, 8))
    idx = select_anchor_set(embeddings, k=10)
    assert len(idx) <= 10
    assert len(set(idx.tolist())) == len(idx)
    assert idx.min() >= 0 and idx.max() < 300
