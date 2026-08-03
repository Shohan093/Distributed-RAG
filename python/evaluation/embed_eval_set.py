import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

from sentence_transformers import SentenceTransformer
from utils.config import load_config
from utils.embedding_helpers import checkpointed_encode

def embed_index_docs(config, epoch, model_key):
    """Embed epoch N's index documents with their OWN native model only."""
    model_info = config["models"][model_key]
    print(f"--- Epoch {epoch} index docs: embedding with {model_info['name']} ({model_key}) ---")

    with open(f"data/eval/generation{epoch}/index_docs.json") as f:
        records = json.load(f)

    doc_ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]

    model = SentenceTransformer(model_info["name"])
    output_dir = f"embeddings/eval/generation{epoch}/index_{model_key}"
    checkpointed_encode(model, texts, doc_ids, output_dir, batch_size=500)

def embed_queries(config, epoch, model_key):
    """Embed epoch N's queries with EVERY model, since we need cross-generation comparisons."""
    model_info = config["models"][model_key]
    print(f"--- Epoch {epoch} queries: embedding with {model_info['name']} ({model_key}) ---")

    with open(f"data/eval/generation{epoch}/queries.json") as f:
        records = json.load(f)

    query_ids = [r["id"] for r in records]
    titles = [r["title"] for r in records]

    model = SentenceTransformer(model_info["name"])
    output_dir = f"embeddings/eval/generation{epoch}/query_{model_key}"
    checkpointed_encode(model, titles, query_ids, output_dir, batch_size=500)

def main():
    config = load_config("configs/embedding.yaml")
    epoch_to_model = {1: "v1", 2: "v2", 3: "v3"}

    for epoch in config["epochs"]:
        native_model = epoch_to_model[epoch]
        embed_index_docs(config, epoch, native_model)

        # queries embedded with ALL THREE models, for cross-generation comparisons
        for model_key in ["v1", "v2", "v3"]:
            embed_queries(config, epoch, model_key)

if __name__ == "__main__":
    main()