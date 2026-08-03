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
from sentence_transformers import SentenceTransformer
from utils.config import load_config

BENCHMARK_SAMPLE_SIZE = 1000  # controlled, fixed-size timing sample per model

EPOCH_TO_MODEL = {1: "v1", 2: "v2", 3: "v3"}
CORPUS_SIZES = {1: 88209, 2: 86577, 3: 85254}
CALIBRATION_SIZE_PER_EPOCH = 2333


def measure_throughput(model_name, sample_texts):
    model = SentenceTransformer(model_name)
    # warm-up run, excluded from timing -- first call includes model
    # initialization overhead that wouldn't repeat in a real batch job
    _ = model.encode(sample_texts[:10], batch_size=8, normalize_embeddings=True)

    start = time.perf_counter()
    _ = model.encode(sample_texts, batch_size=8, normalize_embeddings=True)
    elapsed = time.perf_counter() - start

    docs_per_sec = len(sample_texts) / elapsed
    return docs_per_sec, elapsed


def main():
    config = load_config("configs/embedding.yaml")
    all_results = []

    for epoch, model_key in EPOCH_TO_MODEL.items():
        model_name = config["models"][model_key]["name"]
        print(f"--- Benchmarking {model_name} ({model_key}) ---")

        cleaned_df_path = f"data/generation{epoch}/cleaned"
        import pandas as pd
        df = pd.read_parquet(cleaned_df_path)
        sample = df.sample(n=min(BENCHMARK_SAMPLE_SIZE, len(df)), random_state=42)
        texts = [f"{t.strip()}. {x.strip()}" if t and t.strip() else x.strip()
                 for t, x in zip(sample["title"], sample["text"])]

        docs_per_sec, sample_elapsed = measure_throughput(model_name, texts)
        print(f"Throughput: {docs_per_sec:.2f} docs/sec (measured on {len(texts)} docs, {sample_elapsed:.1f}s)")

        corpus_size = CORPUS_SIZES[epoch]
        full_reembed_seconds = corpus_size / docs_per_sec
        calibration_seconds = CALIBRATION_SIZE_PER_EPOCH / docs_per_sec
        savings_ratio = corpus_size / CALIBRATION_SIZE_PER_EPOCH

        result = {
            "epoch": epoch,
            "model_key": model_key,
            "docs_per_sec": docs_per_sec,
            "corpus_size": corpus_size,
            "calibration_size": CALIBRATION_SIZE_PER_EPOCH,
            "full_reembed_seconds": full_reembed_seconds,
            "full_reembed_hours": full_reembed_seconds / 3600,
            "calibration_only_seconds": calibration_seconds,
            "calibration_only_minutes": calibration_seconds / 60,
            "savings_ratio": savings_ratio,
        }
        print(f"Full re-embed (extrapolated): {result['full_reembed_hours']:.2f} hours")
        print(f"Calibration-only re-embed: {result['calibration_only_minutes']:.2f} minutes")
        print(f"Savings ratio: {savings_ratio:.1f}x fewer documents\n")

        all_results.append(result)

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/cost_baseline.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print("Saved results/tables/cost_baseline.json")


if __name__ == "__main__":
    main()