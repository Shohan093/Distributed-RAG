import numpy as np

from driftdrag.detect.data_drift import AdwinLite, centroid_displacement, mmd2_unbiased


def test_mmd_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 4))
    stat = mmd2_unbiased(x, y)
    assert abs(stat) < 0.05


def test_mmd_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 4)) + 5.0
    stat = mmd2_unbiased(x, y)
    assert stat > 0.5


def test_centroid_displacement_matches_known_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 3))
    shifted = ref + np.array([3.0, 0.0, 0.0])
    assert abs(centroid_displacement(ref, shifted) - 3.0) < 1e-6


def test_adwin_lite_detects_a_mean_shift():
    rng = np.random.default_rng(0)
    detector = AdwinLite(delta=0.05)
    detected_at = None
    for i in range(200):
        value = rng.normal(loc=0.0, scale=0.1) if i < 100 else rng.normal(loc=5.0, scale=0.1)
        if detector.update(value):
            detected_at = i
            break
    assert detected_at is not None
    assert detected_at >= 100
