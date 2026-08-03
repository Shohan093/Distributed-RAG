import datetime
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
from scipy.linalg import orthogonal_procrustes


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def ndcg_and_recall_at_10(query_ids, ranked_doc_ids_per_query):
    """Single-relevant-document case: ground truth doc id == query id."""
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


def load_calibration(model_key):
    emb = np.load(f"embeddings/calibration/{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/calibration/{model_key}/doc_ids.json"))
    return emb, ids


def load_eval_index(epoch, model_key):
    emb = np.load(f"embeddings/eval/generation{epoch}/index_{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/eval/generation{epoch}/index_{model_key}/doc_ids.json"))
    return emb, ids


def load_eval_queries(epoch, model_key):
    emb = np.load(f"embeddings/eval/generation{epoch}/query_{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/eval/generation{epoch}/query_{model_key}/doc_ids.json"))
    return emb, ids


def retrieve_and_score(query_emb, query_ids, index_emb, index_ids, label):
    index_norm = normalize(index_emb)
    query_norm = normalize(query_emb)
    sims = query_norm @ index_norm.T
    ranked = np.argsort(-sims, axis=1)
    ranked_doc_ids = [[index_ids[j] for j in row] for row in ranked]
    ndcg, recall = ndcg_and_recall_at_10(query_ids, ranked_doc_ids)
    print(f"[{label}] NDCG@10: {ndcg:.4f}  Recall@10: {recall:.4f}  (n={len(query_ids)})")
    return ndcg, recall


def evaluate_transition(epoch, target_model_key, all_results):
    print(f"\n=== Epoch {epoch}: target model = {target_model_key} ===")

    index_emb, index_ids = load_eval_index(epoch, target_model_key)

    # --- Oracle: native query vs native index ---
    oracle_q_emb, oracle_q_ids = load_eval_queries(epoch, target_model_key)
    ndcg, recall = retrieve_and_score(oracle_q_emb, oracle_q_ids, index_emb, index_ids, f"Oracle (epoch {epoch})")
    all_results.append({"epoch": epoch, "condition": "oracle", "ndcg_at_10": ndcg, "recall_at_10": recall, "n_queries": len(oracle_q_ids)})

    if target_model_key == "v3":
        print("(No compensation story for v3 -- oracle sanity check only)")
        return

    # --- Train adapter on calibration set: v3 -> target ---
    v3_cal_emb, v3_cal_ids = load_calibration("v3")
    target_cal_emb, target_cal_ids = load_calibration(target_model_key)
    assert v3_cal_ids == target_cal_ids, "Calibration id mismatch -- would silently corrupt the adapter"

    R, _ = orthogonal_procrustes(v3_cal_emb, target_cal_emb)

    os.makedirs("checkpoints/adapters", exist_ok=True)
    np.save(f"checkpoints/adapters/R_v3_to_{target_model_key}.npy", R)

    # --- Naive: raw v3 query vs target index ---
    v3_q_emb, v3_q_ids = load_eval_queries(epoch, "v3")
    ndcg, recall = retrieve_and_score(v3_q_emb, v3_q_ids, index_emb, index_ids, f"Naive (epoch {epoch})")
    all_results.append({"epoch": epoch, "condition": "naive", "ndcg_at_10": ndcg, "recall_at_10": recall, "n_queries": len(v3_q_ids)})

    # --- Compensated: v3 query x R vs target index ---
    compensated_emb = v3_q_emb @ R
    ndcg, recall = retrieve_and_score(compensated_emb, v3_q_ids, index_emb, index_ids, f"Compensated (epoch {epoch})")
    all_results.append({"epoch": epoch, "condition": "compensated", "ndcg_at_10": ndcg, "recall_at_10": recall, "n_queries": len(v3_q_ids)})


def main():
    all_results = []

    evaluate_transition(1, "v1", all_results)
    evaluate_transition(2, "v2", all_results)
    evaluate_transition(3, "v3", all_results)

    os.makedirs("results/tables", exist_ok=True)
    output = {
        "generated_at": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "results": all_results,
    }
    output_path = "results/tables/adapter_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path} (commit {output['git_commit'][:8]})")


if __name__ == "__main__":
    main()