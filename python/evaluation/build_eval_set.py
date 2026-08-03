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
from utils.config import load_config, cleaned_data_path

INDEX_POOL_SIZE = 5000
QUERY_SAMPLE_SIZE = 500
SEED = 42

def build_eval_set_for_epoch(config, epoch):
    cleaned_path = cleaned_data_path(config, epoch)
    df = pd.read_parquet(cleaned_path)

    anchor_ids_path = f"checkpoints/calibration_ids/epoch{epoch}.json"
    with open(anchor_ids_path) as f:
        calibration_ids = set(json.load(f))

    held_out = df[~df["row_num"].isin(calibration_ids)].copy()
    print(f"Epoch {epoch}: {len(df)} total, {len(held_out)} held out after excluding {len(calibration_ids)} calibration docs")

    # Only keep documents with a real, non-empty title AND a reasonably substantial body,
    # since a query needs real title text and the body needs to be searchable on its own.
    held_out = held_out[(held_out["title"].str.strip().str.len() > 0) & (held_out["text"].str.len() > 200)]
    print(f"Epoch {epoch}: {len(held_out)} after requiring non-empty title and substantial body")

    rng = np.random.default_rng(SEED)
    pool_size = min(INDEX_POOL_SIZE, len(held_out))
    index_pool = held_out.sample(n=pool_size, random_state=SEED)

    query_size = min(QUERY_SAMPLE_SIZE, len(index_pool))
    queries = index_pool.sample(n=query_size, random_state=SEED)

    output_dir = f"data/eval/generation{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    index_records = [{"id": row["row_num"], "text": row["text"]} for _, row in index_pool.iterrows()]
    query_records = [{"id": row["row_num"], "title": row["title"]} for _, row in queries.iterrows()]

    with open(os.path.join(output_dir, "index_docs.json"), "w") as f:
        json.dump(index_records, f)
    with open(os.path.join(output_dir, "queries.json"), "w") as f:
        json.dump(query_records, f)

    print(f"Epoch {epoch}: saved {len(index_records)} index docs, {len(query_records)} queries to {output_dir}")

def main():
    config = load_config("configs/embedding.yaml")
    for epoch in config["epochs"]:
        build_eval_set_for_epoch(config, epoch)

if __name__ == "__main__":
    main()