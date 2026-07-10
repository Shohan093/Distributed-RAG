"""Query-time compensation operators (Stage 3 / E2).

Given anchor embeddings under the old model f_t' and the new model f_t,
estimate a correction that lets a query embedded with f_t be compared
against a shard indexed under f_t' without re-embedding the shard.

Implements the operator family compared head-to-head in E2:
  - global drift vector (QDC-global baseline)
  - per-shard drift vector
  - per-cluster drift vectors (this project's default)
  - orthogonal Procrustes map
  - low-rank affine map

All operators are pure numpy so they run identically inside a PySpark UDF,
a driver-side batch job, or a unit test -- no Spark session required.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def global_drift_vector(anchors_old: np.ndarray, anchors_new: np.ndarray) -> np.ndarray:
    """QDC-style single cumulative drift vector: mean(new - old)."""
    return (anchors_new - anchors_old).mean(axis=0)


def per_cluster_drift_vectors(
    anchors_old: np.ndarray,
    anchors_new: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 0,
) -> tuple[KMeans, dict[int, np.ndarray]]:
    """Cluster anchors in the *old* space, then compute a local mean drift
    vector per cluster. Query-time lookup assigns the query's nearest
    cluster centroid (in old space) and applies that cluster's correction.
    This is the finer-grained alternative to a single global drift vector
    that Section 2 Stage 1/3 identifies as this project's main departure
    from QDC.
    """
    n_clusters = min(n_clusters, len(anchors_old))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(anchors_old)
    labels = km.labels_
    vectors: dict[int, np.ndarray] = {}
    diffs = anchors_new - anchors_old
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            vectors[c] = diffs[mask].mean(axis=0)
        else:
            vectors[c] = diffs.mean(axis=0)
    return km, vectors


def apply_cluster_drift(query_old_space: np.ndarray, km: KMeans, vectors: dict[int, np.ndarray]) -> np.ndarray:
    cluster = int(km.predict(query_old_space.reshape(1, -1))[0])
    return query_old_space + vectors[cluster]


def orthogonal_procrustes(anchors_old: np.ndarray, anchors_new: np.ndarray) -> np.ndarray:
    """Best orthogonal map R minimizing ||anchors_new @ R - anchors_old||_F."""
    m = anchors_new.T @ anchors_old
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def low_rank_affine(anchors_old: np.ndarray, anchors_new: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank-constrained affine map new -> old via truncated-SVD least squares.

    Returns (W, b) such that anchors_new @ W + b ~= anchors_old.
    """
    x = np.hstack([anchors_new, np.ones((anchors_new.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(x, anchors_old, rcond=None)
    w_full, b = solution[:-1], solution[-1]
    if rank is not None and rank < min(w_full.shape):
        u, s, vt = np.linalg.svd(w_full, full_matrices=False)
        w_full = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return w_full, b


def adaptation_recall_ratio(recall_compensated: float, recall_oracle: float, recall_misaligned: float) -> float:
    """ARR metric from Drift-Adapter: fraction of the achievable recall gap
    (misaligned -> oracle) recovered by a compensation method. Used
    throughout E2/E6 so numbers are directly comparable across papers.
    """
    denom = recall_oracle - recall_misaligned
    if denom <= 0:
        return 1.0
    return (recall_compensated - recall_misaligned) / denom
