import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)

sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import load_config, cleaned_data_path, embeddings_output_path

# Each epoch is embedded ONLY by its own designated generation's model —
# not by all three. This matches the actual project architecture:
# Epoch 1 -> V1, Epoch 2 -> V2, Epoch 3 -> V3.
EPOCH_TO_MODEL_KEY = {1: "v1", 2: "v2", 3: "v3"}

def embed_epoch(config, epoch):
    model_key = EPOCH_TO_MODEL_KEY[epoch]
    model_info = config["models"][model_key]
    model_name = model_info["name"]

    print(f"--- Epoch {epoch}: embedding with {model_name} ({model_key}) ---")
    model = SentenceTransformer(model_name)

    input_path = cleaned_data_path(config, epoch)
    df = pd.read_parquet(input_path)

    doc_ids = df["row_num"].tolist()
    texts = [
        f"{title.strip()}. {text.strip()}" if title and title.strip() else text.strip()
        for title, text in zip(df["title"], df["text"])
    ]

    print(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    output_dir = embeddings_output_path(config, epoch, model_key)
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "doc_embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        import json
        json.dump(doc_ids, f)

    print(f"Saved {embeddings.shape} to {output_dir}")

def main():
    config = load_config("configs/embedding.yaml")
    for epoch in config["epochs"]:
        embed_epoch(config, epoch)

if __name__ == "__main__":
    main()