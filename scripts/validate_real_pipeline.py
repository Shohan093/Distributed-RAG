"""E0 step 1, first sub-step: validate the pipeline against ONE real BEIR
dataset and ONE real embedding model before touching continual fine-tuning
or the full 5-task drift sequence (see PLAN.md section 3, E0).

Downloads FiQA (smallest of the five BEIR datasets used across E1-E6),
embeds corpus + queries with nomic-embed-text-v1.5 ("v0" in the version
sequence, no fine-tuning yet), retrieves with FAISS, and scores with the
same Recall@k/nDCG@10/MRR functions used throughout the E-series -- so a
plausible number here means embed -> index -> retrieve -> score is correct
end to end on real data, not just on the smoke test's synthetic embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from driftdrag.eval.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from driftdrag.index.faiss_shard import FaissShard  # noqa: E402
from driftdrag.ingest.embeddings import SentenceTransformerBackend  # noqa: E402

DATASET = "fiqa"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_VERSION = "v0"
BEIR_DATA_DIR = REPO_ROOT / "data" / "beir"


def download_dataset() -> Path:
    from beir import util

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    out_dir = util.download_and_unzip(url, str(BEIR_DATA_DIR))
    return Path(out_dir)


def load_dataset(data_path: Path):
    from beir.datasets.data_loader import GenericDataLoader

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    return corpus, queries, qrels


def main() -> None:
    print(f"== 1/5 downloading BEIR/{DATASET} ==")
    data_path = download_dataset()

    print("== 2/5 loading corpus/queries/qrels ==")
    corpus, queries, qrels = load_dataset(data_path)
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]
    print(f"corpus: {len(doc_ids)} docs, queries: {len(query_ids)}")

    print(f"== 3/5 embedding with {MODEL_NAME} (GPU if available) ==")
    backend = SentenceTransformerBackend(
        checkpoints={MODEL_VERSION: MODEL_NAME},
        dim=768,
        trust_remote_code=True,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    doc_embeddings = backend.embed_documents(doc_texts, MODEL_VERSION)
    query_embeddings = backend.embed_queries(query_texts, MODEL_VERSION)
    print(f"doc_embeddings: {doc_embeddings.shape}, query_embeddings: {query_embeddings.shape}")

    print("== 4/5 building FAISS index + retrieving ==")
    shard = FaissShard(dim=doc_embeddings.shape[1], index_type="flat")
    shard.add(doc_embeddings, doc_ids, MODEL_VERSION)

    k = 10
    recalls, ndcgs, mrrs = [], [], []
    for qid, qvec in zip(query_ids, query_embeddings):
        if qid not in qrels or not qrels[qid]:
            continue
        relevant = {d for d, rel in qrels[qid].items() if rel > 0}
        if not relevant:
            continue
        relevance = {d: rel for d, rel in qrels[qid].items()}
        results = shard.search(qvec, k=k)
        retrieved = [doc_id for doc_id, _score in results]
        recalls.append(recall_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevance, k))
        mrrs.append(mrr(retrieved, relevant))

    print("== 5/5 results ==")
    print(f"queries scored: {len(recalls)}")
    print(f"Recall@{k}: {np.mean(recalls):.4f}")
    print(f"nDCG@{k}:  {np.mean(ndcgs):.4f}")
    print(f"MRR:      {np.mean(mrrs):.4f}")
    print("\nCompare nDCG@10 against nomic-embed-text-v1.5's published BEIR/FiQA number "
          "(model card / BEIR leaderboard) -- in the right ballpark means embed->index->"
          "retrieve->score is validated on real data.")


if __name__ == "__main__":
    main()
