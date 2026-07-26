import os
import json
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
ORIGINAL_CWD = os.getcwd()
os.chdir(REPO_ROOT)

import numpy as np
from sklearn.cluster import KMeans

def resolve_path(p):
    return p if os.path.isabs(p) else os.path.join(ORIGINAL_CWD, p)

def select_anchors(embeddings, doc_ids, n_clusters, n_centroid, n_random, n_outlier, seed=42):
    rng = np.random.default_rng(seed)
    n_docs = embeddings.shape[0]

    # Cluster this epoch's document in its own embedding space
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Centroid representatives: nearest real document to each centroid
    centroid_doc_indices = []
    for c in range(n_clusters):
        cluster_number_indices = np.where(cluster_assignments == c)[0]
        if len(cluster_number_indices) == 0:
            continue
        cluster_embeddings = embeddings[cluster_number_indices]
        dists = np.linalg.norm(cluster_embeddings - centroids[c], axis=1)
        nearest_local_idx = cluster_number_indices[np.argmin(dists)]
        centroid_doc_indices.append(nearest_local_idx)
    centroid_doc_indices = centroid_doc_indices[:n_centroid]

    # Outliers: farthest document from its own assigned centroid
    all_dists_to_own_centroid = np.linalg.norm(
        embeddings - centroids[cluster_assignments], axis=1
    )
    outlier_order = np.argsort(-all_dists_to_own_centroid)
    outlier_doc_indices = [i for i in outlier_order if i not in centroid_doc_indices][:n_outlier]

    # --- Random: anything not already selected ---
    already_selected = set(centroid_doc_indices) | set(outlier_doc_indices)
    remaining = [i for i in range(n_docs) if i not in already_selected]
    random_doc_indices = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False).tolist()

    selected_indices = list(set(centroid_doc_indices) | set(outlier_doc_indices) | set(random_doc_indices))
    selected_doc_ids = [doc_ids[i] for i in selected_indices]

    return selected_doc_ids, len(centroid_doc_indices), len(outlier_doc_indices), len(random_doc_indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-embeddings", required=True)
    parser.add_argument("--doc-ids", required=True)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--n-centroid", type=int, default=1333)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--n-outlier", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    embeddings = np.load(resolve_path(args.doc_embeddings))
    with open(resolve_path(args.doc_ids)) as f:
        doc_ids = json.load(f)

    selected_ids, n_c, n_o, n_r = select_anchors(
        embeddings, doc_ids, args.n_clusters, args.n_centroid, args.n_random, args.n_outlier
    )

    with open(resolve_path(args.output), "w") as f:
        json.dump(selected_ids, f)

    print(f"Selected {len(selected_ids)} anchors: {n_c} centroid, {n_o} outlier, {n_r} random")

if __name__ == "__main__":
    main()