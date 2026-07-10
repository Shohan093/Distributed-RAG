"""Anchor-residual monitor (Stage 2.1): the cross-space, model-drift signal.

On every model upgrade v_from -> v_to, re-embed only the small fixed anchor
set A and measure the residual error left *after* compensation. Unlike
intra-space monitors (MMD/ADWIN in data_drift.py) this is sensitive to drift
that a compensation operator failed to correct -- exactly the signal
Section 2 Stage 2 and E3's quality-drop predictor consume.
"""

from __future__ import annotations

import numpy as np


def select_anchor_set(embeddings: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """k-means-coverage anchor selection: pick the embedding closest to each
    of k cluster centroids, so anchors represent the shard's density rather
    than a uniform random sample (ablated against random/herding in E2).
    """
    from sklearn.cluster import KMeans

    k = min(k, len(embeddings))
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(embeddings)
    idx = []
    for c in range(k):
        cluster_points = np.where(km.labels_ == c)[0]
        if len(cluster_points) == 0:
            continue
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(embeddings[cluster_points] - centroid, axis=1)
        idx.append(cluster_points[int(np.argmin(dists))])
    return np.array(sorted(set(idx)))


def anchor_residual(
    anchors_new_compensated: np.ndarray,
    anchors_old: np.ndarray,
) -> float:
    """Mean residual distance after compensation:
    mean || f_t(a) compensated - f_t'(a) ||.
    High residual => the compensation operator is failing for this
    shard/cluster/transition => a candidate for re-embedding (feeds Stage 4).
    """
    diffs = anchors_new_compensated - anchors_old
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def per_cluster_anchor_residual(
    anchors_new_compensated: np.ndarray,
    anchors_old: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        out[int(c)] = anchor_residual(anchors_new_compensated[mask], anchors_old[mask])
    return out
