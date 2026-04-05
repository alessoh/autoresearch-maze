"""
Reads results.tsv and the logs/ directory to generate a comprehensive report.md.

Usage:
    uv run report.py
"""

import os
import re
import sys
from datetime import datetime

import pandas as pd


def parse_log_file(log_path):
    """Extract key metrics from a single experiment log."""
    metrics = {}
    tier_breakdown = {}

    if not os.path.exists(log_path):
        return metrics, tier_breakdown

    with open(log_path, "r") as f:
        content = f.read()

    patterns = {
        "solve_rate": r"^solve_rate:\s+([\d.]+)",
        "training_seconds": r"^training_seconds:\s+([\d.]+)",
        "total_seconds": r"^total_seconds:\s+([\d.]+)",
        "eval_seconds": r"^eval_seconds:\s+([\d.]+)",
        "peak_vram_mb": r"^peak_vram_mb:\s+([\d.]+)",
        "total_mazes_generated": r"^total_mazes_generated:\s+(\d+)",
        "total_valid_mazes": r"^total_valid_mazes:\s+(\d+)",
        "maze_validity_rate": r"^maze_validity_rate:\s+([\d.]+)",
        "total_solver_steps": r"^total_solver_steps:\s+(\d+)",
        "num_params_solver": r"^num_params_solver:\s+(\d+)",
        "num_params_generator": r"^num_params_generator:\s+(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            metrics[key] = match.group(1)

    tier_names = ["trivial", "easy", "medium", "hard", "vhard"]
    for tier in tier_names:
        match = re.search(rf"^\s+{tier}:\s+([\d.]+)", content, re.MULTILINE)
        if match:
            tier_breakdown[tier] = float(match.group(1))

    return metrics, tier_breakdown


def main():
    tsv_path = "results.tsv"
    if not os.path.exists(tsv_path):
        print(f"No {tsv_path} found. Run some experiments first.")
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t")
    if len(df) == 0:
        print("results.tsv is empty.")
        sys.exit(1)

    lines = []
    lines.append("# Autoresearch-Maze Experiment Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary statistics
    total = len(df)
    kept = len(df[df["status"] == "keep"])
    discarded = len(df[df["status"] == "discard"])
    crashed = len(df[df["status"] == "crash"])
    non_crash = df[df["status"] != "crash"]
    best_rate = non_crash["solve_rate"].max() if len(non_crash) > 0 else 0.0
    best_row = non_crash.loc[non_crash["solve_rate"].idxmax()] if len(non_crash) > 0 else None

    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total experiments: {total}")
    lines.append(f"Kept: {kept}, Discarded: {discarded}, Crashed: {crashed}")
    lines.append(f"Best solve_rate: {best_rate:.6f}")
    if best_row is not None:
        lines.append(f"Best experiment: {best_row['commit']} ({best_row['description']})")
    lines.append("")

    # Baseline vs best improvement
    if len(non_crash) >= 2:
        baseline_rate = non_crash.iloc[0]["solve_rate"]
        improvement = best_rate - baseline_rate
        lines.append(f"Baseline solve_rate: {baseline_rate:.6f}")
        lines.append(f"Improvement over baseline: {improvement:+.6f} ({improvement/max(baseline_rate, 1e-8)*100:+.1f}%)")
        lines.append("")

    # Per-tier breakdown for the best experiment
    if best_row is not None:
        log_path = os.path.join("logs", f"{best_row['commit']}.log")
        metrics, tier_breakdown = parse_log_file(log_path)
        if tier_breakdown:
            lines.append("## Best Experiment: Per-Tier Breakdown")
            lines.append("")
            lines.append("| Tier | Solve Rate |")
            lines.append("|------|-----------|")
            for tier in ["trivial", "easy", "medium", "hard", "vhard"]:
                rate = tier_breakdown.get(tier, 0.0)
                lines.append(f"| {tier} | {rate:.4f} |")
            lines.append("")

    # Full experiment log
    lines.append("## Experiment Log")
    lines.append("")
    lines.append("| # | Commit | Solve Rate | VRAM (GB) | Status | Description |")
    lines.append("|---|--------|-----------|-----------|--------|-------------|")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        status_icon = {"keep": "✓", "discard": "✗", "crash": "💥"}.get(row["status"], "?")
        lines.append(
            f"| {i} | {row['commit']} | {row['solve_rate']:.6f} | "
            f"{row['memory_gb']:.1f} | {status_icon} {row['status']} | {row['description']} |"
        )
    lines.append("")

    # Advancement history (only kept experiments)
    kept_df = df[df["status"] == "keep"]
    if len(kept_df) > 0:
        lines.append("## Advancement History")
        lines.append("")
        lines.append("These experiments improved solve_rate and were kept:")
        lines.append("")
        for i, (_, row) in enumerate(kept_df.iterrows(), 1):
            lines.append(f"{i}. **{row['commit']}** solve_rate={row['solve_rate']:.6f} — {row['description']}")
        lines.append("")

    # Detailed logs for each experiment
    lines.append("## Detailed Experiment Logs")
    lines.append("")
    for _, row in df.iterrows():
        commit = row["commit"]
        log_path = os.path.join("logs", f"{commit}.log")
        lines.append(f"### Experiment {commit}: {row['description']}")
        lines.append("")
        lines.append(f"Status: {row['status']}, solve_rate: {row['solve_rate']:.6f}, VRAM: {row['memory_gb']:.1f} GB")
        lines.append("")

        metrics, tier_breakdown = parse_log_file(log_path)
        if metrics:
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        if tier_breakdown:
            lines.append("Per-tier breakdown:")
            lines.append("")
            for tier, rate in tier_breakdown.items():
                lines.append(f"- {tier}: {rate:.4f}")
            lines.append("")

        if not metrics and not tier_breakdown:
            if os.path.exists(log_path):
                lines.append("(Log file exists but could not parse metrics)")
            else:
                lines.append("(No log file found)")
            lines.append("")

    report = "\n".join(lines)
    with open("report.md", "w") as f:
        f.write(report)
    print(f"Generated report.md ({len(df)} experiments)")


if __name__ == "__main__":
    main()
