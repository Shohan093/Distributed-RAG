"""Data-drift monitor (Stage 2.2): the intra-space, streaming-content signal.

Two complementary detectors over a per-shard sliding window of chunk
embeddings, as specified in Section 2 Stage 2:
  - MMD two-sample test (Gretton et al., JMLR 2012) comparing a reference
    window to the current window.
  - ADWIN-lite: an adaptive-windowing change detector over a scalar summary
    statistic (e.g. centroid displacement), used when a full MMD pass is too
    expensive to run on every micro-batch.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_sq = np.sum(x ** 2, axis=1)[:, None]
    y_sq = np.sum(y ** 2, axis=1)[None, :]
    sq_dists = x_sq + y_sq - 2 * x @ y.T
    return np.exp(-gamma * np.clip(sq_dists, 0, None))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack([x, y])
    n = min(len(combined), 500)
    sample = combined[np.random.choice(len(combined), n, replace=False)] if len(combined) > n else combined
    dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    median = np.median(dists[dists > 0])
    return 1.0 / (2 * median ** 2 + 1e-12)


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD^2 estimate between reference window x and current window y."""
    if gamma is None:
        gamma = median_heuristic_gamma(x, y)
    kxx = _rbf_kernel(x, x, gamma)
    kyy = _rbf_kernel(y, y, gamma)
    kxy = _rbf_kernel(x, y, gamma)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    term_x = kxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_y = kyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = kxy.sum() / (n * m)
    return float(term_x + term_y - 2 * term_xy)


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 200, gamma: float | None = None) -> tuple[float, float]:
    """Returns (mmd2_statistic, p_value) via label-permutation null."""
    observed = mmd2_unbiased(x, y, gamma)
    combined = np.vstack([x, y])
    n = len(x)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(combined))
        xp, yp = combined[perm[:n]], combined[perm[n:]]
        if mmd2_unbiased(xp, yp, gamma) >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


class AdwinLite:
    """A simplified adaptive-windowing change detector (Bifet & Gavalda,
    SDM 2007) over a scalar stream (e.g. per-batch centroid displacement or
    MMD statistic). Cheaper than a full MMD pass on every micro-batch, so it
    is what the streaming job calls per-record while MMD runs per-epoch.
    """

    def __init__(self, delta: float = 0.05, max_window: int = 2000):
        self.delta = delta
        self.max_window = max_window
        self.window: deque[float] = deque(maxlen=max_window)
        self.drift_points: list[int] = []
        self._n_seen = 0

    def update(self, value: float) -> bool:
        self.window.append(value)
        self._n_seen += 1
        if len(self.window) < 10:
            return False
        arr = np.array(self.window)
        n = len(arr)
        for cut in range(5, n - 5):
            w0, w1 = arr[:cut], arr[cut:]
            n0, n1 = len(w0), len(w1)
            mean_diff = abs(w0.mean() - w1.mean())
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1 / (2 * m)) * np.log(4 / self.delta))
            if mean_diff > epsilon:
                for _ in range(cut):
                    self.window.popleft()
                self.drift_points.append(self._n_seen)
                return True
        return False


def centroid_displacement(reference: np.ndarray, current: np.ndarray) -> float:
    return float(np.linalg.norm(current.mean(axis=0) - reference.mean(axis=0)))
