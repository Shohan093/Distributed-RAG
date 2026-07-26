import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.config import load_config
from utils.embedding_helpers import checkpointed_encode

def embed_calibration_with_model(config, model_key):
    model_info = config["models"][model_key]
    print(f"--- Calibration set: embedding with {model_info['name']} ({model_key}) ---")

    df = pd.read_parquet(config["calibration"]["texts_output"])
    doc_ids = df["row_num"].tolist()
    texts = [
        f"{title.strip()}. {text.strip()}" if title and title.strip() else text.strip()
        for title, text in zip(df["title"], df["text"])
    ]

    model = SentenceTransformer(model_info["name"])
    output_dir = config["calibration"]["embeddings_output_template"].format(model_key=model_key)
    checkpointed_encode(model, texts, doc_ids, output_dir)

def main():
    config = load_config("configs/embedding.yaml")
    for model_key in ["v1", "v2", "v3"]:
        embed_calibration_with_model(config, model_key)

if __name__ == "__main__":
    main()