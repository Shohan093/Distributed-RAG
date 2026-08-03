import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import orthogonal_procrustes

N_CLUSTERS = 20  # small, since we're clustering the calibration set (6999 docs), not the full corpus


def load_calibration(model_key):
    emb = np.load(f"embeddings/calibration/{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/calibration/{model_key}/doc_ids.json"))
    return emb, ids


def train_l2(target_model_key):
    v3_emb, v3_ids = load_calibration("v3")
    target_emb, target_ids = load_calibration(target_model_key)
    assert v3_ids == target_ids, "Calibration id mismatch"

    # Reuse the already-trained global L1 rotation as the base
    R = np.load(f"checkpoints/adapters/R_v3_to_{target_model_key}.npy")

    # Cluster calibration docs in v3 space -- this defines which cluster
    # a NEW query will be assigned to at inference time
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(v3_emb)

    # After applying the global rotation, compute the RESIDUAL error per cluster
    # -- this residual is exactly what L2's small correction is meant to capture
    rotated = v3_emb @ R
    residuals = target_emb - rotated

    cluster_biases = np.zeros((N_CLUSTERS, v3_emb.shape[1]))
    for c in range(N_CLUSTERS):
        mask = cluster_assignments == c
        if mask.sum() > 0:
            cluster_biases[c] = residuals[mask].mean(axis=0)

    os.makedirs("checkpoints/adapters", exist_ok=True)
    np.save(f"checkpoints/adapters/L2_centroids_v3_to_{target_model_key}.npy", kmeans.cluster_centers_)
    np.save(f"checkpoints/adapters/L2_biases_v3_to_{target_model_key}.npy", cluster_biases)

    avg_bias_norm = np.linalg.norm(cluster_biases, axis=1).mean()
    print(f"L2 (v3 -> {target_model_key}): {N_CLUSTERS} clusters, avg bias norm={avg_bias_norm:.4f}")


if __name__ == "__main__":
    train_l2("v1")
    train_l2("v2")