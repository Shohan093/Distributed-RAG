import numpy as np
import pytest

from driftdrag.compensate.drift_vectors import (
    adaptation_recall_ratio,
    apply_cluster_drift,
    global_drift_vector,
    low_rank_affine,
    orthogonal_procrustes,
    per_cluster_drift_vectors,
)


def test_global_drift_vector_recovers_known_shift():
    rng = np.random.default_rng(0)
    old = rng.normal(size=(200, 8))
    shift = np.array([1.0, -2.0, 0.5, 0, 0, 0, 0, 0])
    new = old + shift
    estimated = global_drift_vector(old, new)
    assert np.allclose(estimated, shift, atol=1e-8)


def test_per_cluster_drift_vectors_recover_piecewise_shift():
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(size=(100, 4)) + np.array([10, 0, 0, 0])
    cluster_b = rng.normal(size=(100, 4)) + np.array([-10, 0, 0, 0])
    old = np.vstack([cluster_a, cluster_b])
    shift_a, shift_b = np.array([1.0, 0, 0, 0]), np.array([-1.0, 0, 0, 0])
    new = np.vstack([cluster_a + shift_a, cluster_b + shift_b])

    km, vectors = per_cluster_drift_vectors(old, new, n_clusters=2)
    query_a = cluster_a[0]
    corrected = apply_cluster_drift(query_a, km, vectors)
    assert np.allclose(corrected, query_a + shift_a, atol=1e-8)


def test_orthogonal_procrustes_recovers_rotation():
    rng = np.random.default_rng(2)
    old = rng.normal(size=(50, 3))
    theta = np.pi / 6
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    new = old @ rotation
    r = orthogonal_procrustes(old, new)
    assert np.allclose(new @ r, old, atol=1e-6)


def test_low_rank_affine_fits_linear_map():
    rng = np.random.default_rng(3)
    new = rng.normal(size=(100, 5))
    w_true = rng.normal(size=(5, 5))
    b_true = rng.normal(size=5)
    old = new @ w_true + b_true

    w, b = low_rank_affine(old, new, rank=5)
    reconstructed = new @ w + b
    assert np.allclose(reconstructed, old, atol=1e-6)


def test_adaptation_recall_ratio_bounds():
    assert adaptation_recall_ratio(0.8, 0.9, 0.5) == pytest.approx(0.75)
    assert adaptation_recall_ratio(0.9, 0.9, 0.5) == pytest.approx(1.0)
    assert adaptation_recall_ratio(0.5, 0.5, 0.5) == 1.0
