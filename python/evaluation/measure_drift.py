import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np


def pairwise_sq_dists(A, B):
    """Squared Euclidean distances between every row of A and every row of B,
    without ever materializing an (n, m, d) tensor."""
    return np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T


def rbf_mmd2(X, Y, gamma=None):
    if gamma is None:
        combined = np.vstack([X, Y])
        sq_dists = pairwise_sq_dists(combined, combined)
        gamma = 1.0 / (2 * np.median(sq_dists[sq_dists > 0]))

    def rbf(A, B):
        return np.exp(-gamma * pairwise_sq_dists(A, B))

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    n, m = len(X), len(Y)
    return Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2 * Kxy.sum() / (n * m)

def load_calibration(model_key):
    emb = np.load(f"embeddings/calibration/{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/calibration/{model_key}/doc_ids.json"))
    return emb, ids


def measure_drift_for_transition(target_model_key):
    v3_emb, v3_ids = load_calibration("v3")
    target_emb, target_ids = load_calibration(target_model_key)
    assert v3_ids == target_ids

    R = np.load(f"checkpoints/adapters/R_v3_to_{target_model_key}.npy")

    # A modest, fixed sample -- MMD is O(n^2), and 6999^2 pairwise comparisons
    # would be needlessly expensive for a diagnostic statistic
    rng = np.random.default_rng(42)
    idx = rng.choice(len(v3_emb), size=min(1000, len(v3_emb)), replace=False)

    drift_before = rbf_mmd2(v3_emb[idx], target_emb[idx])
    compensated = v3_emb[idx] @ R
    drift_after = rbf_mmd2(compensated, target_emb[idx])

    reduction_pct = (1 - drift_after / drift_before) * 100 if drift_before > 0 else 0

    print(f"V3 -> {target_model_key}: MMD^2 before={drift_before:.4f}  after={drift_after:.4f}  "
          f"reduction={reduction_pct:.1f}%")

    return {"target": target_model_key, "mmd2_before": float(drift_before),
            "mmd2_after": float(drift_after), "reduction_pct": float(reduction_pct)}


def main():
    results = [measure_drift_for_transition("v1"), measure_drift_for_transition("v2")]
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/drift_measurement.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("Saved results/tables/drift_measurement.json")


if __name__ == "__main__":
    main()