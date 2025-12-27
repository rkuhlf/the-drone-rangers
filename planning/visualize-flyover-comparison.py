"""
Flyover Comparison Visualization

This script visualizes the performance comparison between flyover (conditional repulsion)
and non-flyover herding strategies, plotting completion steps against flock size.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CSV_PATH = "./planning/results/2025-10-08--21-05-00--evaluation_trials.csv"
COLORS = {"flyover": "red", "non_flyover": "blue"}


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: Results file not found at {CSV_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Group by spawn type (e.g., uniform, circle)
    for spawn_type, group_data in df.groupby("spawn_type"):
        # Separate data based on repulsion strategy
        fly_overs = group_data[group_data["conditionally_apply_repulsion"]]
        nonfly_overs = group_data[~group_data["conditionally_apply_repulsion"]]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Scatter Plots
        if not fly_overs.empty:
            fly_overs.plot.scatter(
                x="N",
                y="Completion Steps",
                color=COLORS["flyover"],
                label="Flyover",
                ax=ax,
                alpha=0.6,
            )

        if not nonfly_overs.empty:
            nonfly_overs.plot.scatter(
                x="N",
                y="Completion Steps",
                color=COLORS["non_flyover"],
                label="Non-Flyover",
                ax=ax,
                alpha=0.6,
            )

        # 2. Mean Trend Lines
        if not fly_overs.empty:
            fly_means = fly_overs.groupby("N")["Completion Steps"].mean()
            ax.plot(
                fly_means.index,
                fly_means.values,
                color=COLORS["flyover"],
                linestyle="--",
                label="Flyover Mean",
            )

        if not nonfly_overs.empty:
            nonfly_means = nonfly_overs.groupby("N")["Completion Steps"].mean()
            ax.plot(
                nonfly_means.index,
                nonfly_means.values,
                color=COLORS["non_flyover"],
                linestyle="--",
                label="Non-Flyover Mean",
            )

        # Formatting
        ax.set_title(f"Completion Steps vs Flock Size (Spawn: {spawn_type})")
        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel("Completion Steps")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.show()
