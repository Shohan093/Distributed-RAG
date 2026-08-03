import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np
import faiss
from retrieval.faiss_index import build_flat_index, build_ivf_index

EPOCH_TO_MODEL = {1: "v1", 2: "v2", 3: "v3"}


def build_for_epoch(epoch):
    model_key = EPOCH_TO_MODEL[epoch]
    print(f"--- Epoch {epoch} ({model_key}): building indexes ---")

    emb = np.load(f"embeddings/generation{epoch}/{model_key}/doc_embeddings.npy")
    doc_ids = json.load(open(f"embeddings/generation{epoch}/{model_key}/doc_ids.json"))

    out_dir = f"indexes/generation{epoch}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Building flat index over {len(doc_ids)} documents...")
    flat = build_flat_index(emb, doc_ids)
    faiss.write_index(flat, f"{out_dir}/flat.index")

    print(f"Building IVF index over {len(doc_ids)} documents...")
    ivf = build_ivf_index(emb, doc_ids)
    faiss.write_index(ivf, f"{out_dir}/ivf.index")

    print(f"Saved both indexes to {out_dir}/")


if __name__ == "__main__":
    epoch = int(sys.argv[1])
    build_for_epoch(epoch)