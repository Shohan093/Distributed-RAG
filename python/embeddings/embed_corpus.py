import argparse
import json
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

EPOCH_TO_MODEL_KEY = {1: "v1", 2: "v2", 3: "v3"}
BATCH_SIZE = 2000

def embed_epoch(config, epoch):
    model_key = EPOCH_TO_MODEL_KEY[epoch]
    model_info = config["models"][model_key]
    model_name = model_info["name"]

    output_dir = embeddings_output_path(config, epoch, model_key)
    final_embeddings_path = os.path.join(output_dir, "doc_embeddings.npy")
    final_ids_path = os.path.join(output_dir, "doc_ids.json")

    # --- Epoch-level skip: already fully done, nothing to do ---
    if os.path.exists(final_embeddings_path) and os.path.exists(final_ids_path):
        print(f"Epoch {epoch}: already complete, skipping entirely.")
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")

    print(f"--- Epoch {epoch}: embedding with {model_name} ({model_key}) ---")

    input_path = cleaned_data_path(config, epoch)
    df = pd.read_parquet(input_path)

    doc_ids = df["row_num"].tolist()
    texts = [
        f"{title.strip()}. {text.strip()}" if title and title.strip() else text.strip()
        for title, text in zip(df["title"], df["text"])
    ]

    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    # --- Load manifest of already-completed batches, if resuming ---
    completed_batches = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            completed_batches = set(json.load(f))
        print(f"Resuming: {len(completed_batches)}/{n_batches} batches already done.")

    model = SentenceTransformer(model_name)

    for batch_idx in range(n_batches):
        if batch_idx in completed_batches:
            continue  # already done, skip re-encoding

        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]

        print(f"Batch {batch_idx + 1}/{n_batches} ({start}:{end})...")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

        batch_path = os.path.join(checkpoint_dir, f"batch_{batch_idx:05d}.npy")
        np.save(batch_path, batch_embeddings)

        completed_batches.add(batch_idx)
        with open(manifest_path, "w") as f:
            json.dump(sorted(completed_batches), f)

    # --- All batches done: stitch them together into the final output ---
    print("All batches complete. Assembling final embeddings file...")
    all_embeddings = []
    for batch_idx in range(n_batches):
        batch_path = os.path.join(checkpoint_dir, f"batch_{batch_idx:05d}.npy")
        all_embeddings.append(np.load(batch_path))

    final_embeddings = np.vstack(all_embeddings)
    np.save(final_embeddings_path, final_embeddings)
    with open(final_ids_path, "w") as f:
        json.dump(doc_ids, f)

    print(f"Saved {final_embeddings.shape} to {final_embeddings_path}")

    # --- Clean up checkpoint files now that the final output is safely written ---
    for batch_idx in range(n_batches):
        os.remove(os.path.join(checkpoint_dir, f"batch_{batch_idx:05d}.npy"))
    os.remove(manifest_path)
    os.rmdir(checkpoint_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, choices=[1, 2, 3], default=None,
                        help="Run only this epoch. If omitted, runs all epochs sequentially.")
    args = parser.parse_args()

    config = load_config("configs/embedding.yaml")
    epochs_to_run = [args.epoch] if args.epoch else config["epochs"]

    for epoch in epochs_to_run:
        embed_epoch(config, epoch)

if __name__ == "__main__":
    main()