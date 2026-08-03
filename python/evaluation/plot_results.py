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
    with open("results/tables/adapter_evaluation.json") as f:
        data = json.load(f)
    return data["results"]


def plot_full_ablation(results, metric, ylabel, filename):
    epochs = sorted(e for e in set(r["epoch"] for r in results) if e != 3)  # epoch 3 has no compensation story
    conditions = ["naive", "l0_compensated", "l2_compensated", "l1_compensated", "oracle"]
    labels = {"naive": "Naive", "l0_compensated": "L0", "l2_compensated": "L2", "l1_compensated": "L1", "oracle": "Oracle"}
    colors = {
        "naive": "#E24B4A", "l0_compensated": "#F2A623", "l2_compensated": "#7F77DD",
        "l1_compensated": "#378ADD", "oracle": "#639922",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_width = 0.16
    x = np.arange(len(epochs))

    for i, cond in enumerate(conditions):
        values = []
        for epoch in epochs:
            match = [r for r in results if r["epoch"] == epoch and r["condition"] == cond]
            values.append(match[0][metric] if match else 0)
        ax.bar(x + i * bar_width, values, bar_width, label=labels[cond], color=colors[cond])

    ax.set_xlabel("Epoch (embedding generation)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across compensation methods (naive, L0, L1, L2, oracle)")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved results/figures/{filename}")


def plot_recovery_ratio(results, filename):
    epochs = sorted(e for e in set(r["epoch"] for r in results) if e != 3)
    conditions = ["l0_compensated", "l2_compensated", "l1_compensated"]
    labels = {"l0_compensated": "L0", "l2_compensated": "L2", "l1_compensated": "L1"}
    colors = {"l0_compensated": "#F2A623", "l2_compensated": "#7F77DD", "l1_compensated": "#378ADD"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.25
    x = np.arange(len(epochs))

    for i, cond in enumerate(conditions):
        ratios = []
        for epoch in epochs:
            comp = [r for r in results if r["epoch"] == epoch and r["condition"] == cond][0]
            oracle = [r for r in results if r["epoch"] == epoch and r["condition"] == "oracle"][0]
            ratio = comp["ndcg_at_10"] / oracle["ndcg_at_10"] if oracle["ndcg_at_10"] > 0 else 0
            ratios.append(ratio)
        bars = ax.bar(x + i * bar_width, ratios, bar_width, label=labels[cond], color=colors[cond])
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f"{r:.1%}", ha="center", fontsize=8)

    ax.axhline(1.0, color="#888780", linestyle="--", linewidth=1, label="Oracle ceiling")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NDCG@10 recovery ratio vs. oracle")
    ax.set_title("Compensation method recovery ratio: L0 vs. L1 vs. L2")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved results/figures/{filename}")


def main():
    results = load_results()
    plot_full_ablation(results, "ndcg_at_10", "NDCG@10", "ndcg_full_ablation.png")
    plot_full_ablation(results, "recall_at_10", "Recall@10", "recall_full_ablation.png")
    plot_recovery_ratio(results, "recovery_ratio_l0_l1_l2.png")


if __name__ == "__main__":
    main()