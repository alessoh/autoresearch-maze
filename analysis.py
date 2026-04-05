"""
Reads results.tsv and generates a progress.png chart showing solve_rate over experiments.

Usage:
    uv run analysis.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    tsv_path = "results.tsv"
    if not os.path.exists(tsv_path):
        print(f"No {tsv_path} found. Run some experiments first.")
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t")

    if len(df) == 0:
        print("results.tsv is empty (header only). Run some experiments first.")
        sys.exit(1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    experiments = range(1, len(df) + 1)
    colors = []
    for _, row in df.iterrows():
        if row["status"] == "crash":
            colors.append("#d32f2f")
        elif row["status"] == "discard":
            colors.append("#ff9800")
        else:
            colors.append("#2e7d32")

    # Top subplot: solve_rate over experiments
    ax1 = axes[0]
    ax1.bar(experiments, df["solve_rate"], color=colors, alpha=0.7, width=0.8)

    # Running best line
    running_best = df["solve_rate"].cummax()
    ax1.plot(experiments, running_best, color="#1565c0", linewidth=2,
             marker="", label="Running best")
    ax1.set_ylabel("solve_rate (higher is better)")
    ax1.set_title("Autoresearch-Maze: Solve Rate Progress")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.3)

    # Bottom subplot: memory usage
    ax2 = axes[1]
    ax2.bar(experiments, df["memory_gb"], color=colors, alpha=0.5, width=0.8)
    ax2.set_ylabel("Peak VRAM (GB)")
    ax2.set_xlabel("Experiment #")
    ax2.grid(axis="y", alpha=0.3)

    # Legend for status colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", alpha=0.7, label="keep"),
        Patch(facecolor="#ff9800", alpha=0.7, label="discard"),
        Patch(facecolor="#d32f2f", alpha=0.7, label="crash"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150)
    print(f"Saved progress.png ({len(df)} experiments)")


if __name__ == "__main__":
    main()
