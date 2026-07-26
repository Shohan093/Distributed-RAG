import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import pandas as pd
from utils.config import load_config, cleaned_data_path

def build_calibration_set(config):
    all_rows = []

    for epoch in config["epochs"]:
        anchor_ids_path = config["calibration"]["anchor_ids_template"].format(epoch=epoch)
        with open(anchor_ids_path) as f:
            anchor_ids = set(json.load(f))

        cleaned_path = cleaned_data_path(config, epoch)
        df = pd.read_parquet(cleaned_path)

        matched = df[df["row_num"].isin(anchor_ids)].copy()
        matched["epoch"] = epoch
        print(f"Epoch {epoch}: matched {len(matched)}/{len(anchor_ids)} requested anchor ids")

        all_rows.append(matched[["row_num", "title", "text", "epoch"]])

    combined = pd.concat(all_rows, ignore_index=True)

    output_path = config["calibration"]["texts_output"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(f"Combined calibration set: {len(combined)} total documents saved to {output_path}")

if __name__ == "__main__":
    config = load_config("configs/embedding.yaml")
    build_calibration_set(config)