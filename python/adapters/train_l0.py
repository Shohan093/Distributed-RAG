import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import numpy as np


def load_calibration(model_key):
    emb = np.load(f"embeddings/calibration/{model_key}/doc_embeddings.npy")
    ids = json.load(open(f"embeddings/calibration/{model_key}/doc_ids.json"))
    return emb, ids


def train_l0(target_model_key):
    v3_emb, v3_ids = load_calibration("v3")
    target_emb, target_ids = load_calibration(target_model_key)
    assert v3_ids == target_ids, "Calibration id mismatch"

    delta = target_emb.mean(axis=0) - v3_emb.mean(axis=0)

    os.makedirs("checkpoints/adapters", exist_ok=True)
    np.save(f"checkpoints/adapters/L0_v3_to_{target_model_key}.npy", delta)
    print(f"L0 delta (v3 -> {target_model_key}): shape={delta.shape}, norm={np.linalg.norm(delta):.4f}")


if __name__ == "__main__":
    train_l0("v1")
    train_l0("v2")