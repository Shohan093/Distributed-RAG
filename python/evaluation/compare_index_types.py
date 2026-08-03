import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
import faiss
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ndcg_and_recall_at_10(query_ids, ranked_doc_ids_per_query):
    ndcgs, recalls = [], []
    for qid, ranked in zip(query_ids, ranked_doc_ids_per_query):
        top10 = ranked[:10]
        if qid in top10:
            rank = top10.index(qid)
            ndcgs.append(1.0 / np.log2(rank + 2))
            recalls.append(1.0)
        else:
            ndcgs.append(0.0)
            recalls.append(0.0)
    return float(np.mean(ndcgs)), float(np.mean(recalls))


def evaluate_index(epoch, target_model_key, index_type, query_emb, query_ids, adapter=None):
    index = faiss.read_index(f"indexes/generation{epoch}/{index_type}.index")

    q = query_emb.copy()
    if adapter is not None:
        q = q @ adapter
    q = normalize(q).astype("float32")

    start = time.perf_counter()
    distances, doc_ids = index.search(q, 10)
    elapsed = time.perf_counter() - start
    per_query_ms = (elapsed / len(query_ids)) * 1000

    ranked_doc_ids = [[int(d) for d in row if d != -1] for row in doc_ids]
    ndcg, recall = ndcg_and_recall_at_10(query_ids, ranked_doc_ids)

    return {
        "epoch": epoch, "index_type": index_type,
        "ndcg_at_10": ndcg, "recall_at_10": recall,
        "avg_query_latency_ms": per_query_ms, "n_queries": len(query_ids),
    }


def main():
    all_results = []
    epoch_to_model = {1: "v1", 2: "v2", 3: "v3"}

    for epoch, target in epoch_to_model.items():
        v3_q_emb = np.load(f"embeddings/eval/generation{epoch}/query_v3/doc_embeddings.npy")
        v3_q_ids = json.load(open(f"embeddings/eval/generation{epoch}/query_v3/doc_ids.json"))

        adapter = None
        if target != "v3":
            adapter = np.load(f"checkpoints/adapters/R_v3_to_{target}.npy")

        for index_type in ["flat", "ivf"]:
            result = evaluate_index(epoch, target, index_type, v3_q_emb, v3_q_ids, adapter)
            print(f"Epoch {epoch} [{index_type}]: NDCG@10={result['ndcg_at_10']:.4f}  "
                  f"Recall@10={result['recall_at_10']:.4f}  "
                  f"avg_latency={result['avg_query_latency_ms']:.3f}ms")
            all_results.append(result)

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/index_comparison.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print("\nSaved results/tables/index_comparison.json")


if __name__ == "__main__":
    main()