import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)
os.chdir(REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    with open("results/tables/adapter_evaluation.json") as f:
        data = json.load(f)
    return data["results"], data["generated_at"], data["git_commit"]

def plot_grouped_bars(results, metric, ylabel, filename):
    epochs = sorted(set(r["epoch"] for r in results))
    conditions = ["naive", "compensated", "oracle"]
    colors = {"naive": "#E24B4A", "compensated": "#378ADD", "oracle": "#639922"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.25
    x = np.arange(len(epochs))

    for i, cond in enumerate(conditions):
        values = []
        for epoch in epochs:
            match = [r for r in results if r["epoch"] == epoch and r["condition"] == cond]
            values.append(match[0][metric] if match else 0)
        ax.bar(x + i * bar_width, values, bar_width, label=cond, color=colors[cond])

    ax.set_xlabel("Epoch (embedding generation)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by condition and epoch")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved results/figures/{filename}")

def plot_recovery_ratio(results, filename):
    epochs = sorted(e for e in set(r["epoch"] for r in results) if e != 3)  # epoch 3 has no compensation story

    ratios = []
    for epoch in epochs:
        naive = [r for r in results if r["epoch"] == epoch and r["condition"] == "naive"][0]
        comp = [r for r in results if r["epoch"] == epoch and r["condition"] == "compensated"][0]
        oracle = [r for r in results if r["epoch"] == epoch and r["condition"] == "oracle"][0]
        ratio = comp["ndcg_at_10"] / oracle["ndcg_at_10"] if oracle["ndcg_at_10"] > 0 else 0
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Epoch {e}" for e in epochs], ratios, color="#378ADD")
    ax.axhline(1.0, color="#888780", linestyle="--", linewidth=1, label="Oracle ceiling")
    ax.set_ylabel("NDCG@10 recovery ratio (compensated / oracle)")
    ax.set_title("Compensation recovery relative to oracle")
    ax.set_ylim(0, 1.1)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f"{r:.1%}", ha="center")
    ax.legend()
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved results/figures/{filename}")

def main():
    results, generated_at, commit = load_results()
    print(f"Plotting results generated at {generated_at}, commit {commit[:8]}")

    plot_grouped_bars(results, "ndcg_at_10", "NDCG@10", "ndcg_by_condition.png")
    plot_grouped_bars(results, "recall_at_10", "Recall@10", "recall_by_condition.png")
    plot_recovery_ratio(results, "recovery_ratio.png")

if __name__ == "__main__":
    main()