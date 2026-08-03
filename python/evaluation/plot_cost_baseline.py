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
    with open("results/tables/cost_baseline.json") as f:
        return json.load(f)["results"]


def plot_cost_comparison(results):
    epochs = [r["epoch"] for r in results]
    full_hours = [r["full_reembed_hours"] for r in results]
    calib_minutes_as_hours = [r["calibration_only_minutes"] / 60 for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(epochs))
    bar_width = 0.35

    ax.bar(x - bar_width / 2, full_hours, bar_width, label="Full re-embed", color="#E24B4A")
    ax.bar(x + bar_width / 2, calib_minutes_as_hours, bar_width, label="Calibration-only (compensation)", color="#639922")

    ax.set_yscale("log")  # the gap is ~37x -- log scale makes both bars readable at once
    ax.set_xlabel("Epoch (model transition)")
    ax.set_ylabel("Compute time (hours, log scale)")
    ax.set_title("Re-indexing cost: full corpus re-embed vs. compensation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.legend()

    for i, r in enumerate(results):
        ax.text(i - bar_width / 2, full_hours[i] * 1.15, f"{full_hours[i]:.1f}h", ha="center", fontsize=9)
        ax.text(i + bar_width / 2, calib_minutes_as_hours[i] * 1.15, f"{r['calibration_only_minutes']:.0f}min", ha="center", fontsize=9)

    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/cost_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved results/figures/cost_comparison.png")


if __name__ == "__main__":
    plot_cost_comparison(results=load_results())