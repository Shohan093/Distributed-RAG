import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    with open("results/tables/index_comparison.json") as f:
        return json.load(f)["results"]


def plot_quality_vs_speed(results):
    epochs = sorted(set(r["epoch"] for r in results))
    index_types = ["flat", "ivf"]
    colors = {"flat": "#378ADD", "ivf": "#D85A30"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bar_width = 0.35
    x = np.arange(len(epochs))

    for i, idx_type in enumerate(index_types):
        ndcgs = []
        for epoch in epochs:
            match = [r for r in results if r["epoch"] == epoch and r["index_type"] == idx_type]
            ndcgs.append(match[0]["ndcg_at_10"] if match else 0)
        ax1.bar(x + i * bar_width, ndcgs, bar_width, label=idx_type, color=colors[idx_type])

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NDCG@10")
    ax1.set_title("Retrieval quality: flat vs. IVF")
    ax1.set_xticks(x + bar_width / 2)
    ax1.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax1.set_ylim(0, 1.0)
    ax1.legend()

    for i, idx_type in enumerate(index_types):
        latencies = []
        for epoch in epochs:
            match = [r for r in results if r["epoch"] == epoch and r["index_type"] == idx_type]
            latencies.append(match[0]["avg_query_latency_ms"] if match else 0)
        ax2.bar(x + i * bar_width, latencies, bar_width, label=idx_type, color=colors[idx_type])

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg. query latency (ms)")
    ax2.set_title("Search speed: flat vs. IVF")
    ax2.set_xticks(x + bar_width / 2)
    ax2.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax2.legend()

    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/index_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved results/figures/index_comparison.png")


def main():
    results = load_results()
    plot_quality_vs_speed(results)


if __name__ == "__main__":
    main()