"""
Drone Count Visualization

This script visualizes the relationship between the number of drones and the
completion steps/success rate for different spawn types, based on evaluation results.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CSV_PATH = "./planning/results/2025-10-08--21-05-00--evaluation_trials.csv"
COLORS = ("red", "green", "blue", "orange", "purple")


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

    # Filter for runs where flyover (conditional repulsion) was enabled
    if "conditionally_apply_repulsion" in df.columns:
        df = df[df["conditionally_apply_repulsion"]]

    # Group by spawn type (e.g., uniform, circle)
    for spawn_type, group_data in df.groupby("spawn_type"):
        # 1. Scatter Plot: Completion Steps vs N
        fig, ax = plt.subplots(figsize=(10, 6))

        # Iterate over drone counts
        drone_groups = group_data.groupby("Drone Count")
        for i, (drone_count, data) in enumerate(drone_groups):
            color = COLORS[i % len(COLORS)]

            # Scatter plot of individual trials
            data.plot.scatter(
                x="N",
                y="Completion Steps",
                label=f"{drone_count} Drones",
                color=color,
                ax=ax,
                alpha=0.6,
            )

            # Line plot of mean completion steps
            means = data.groupby("N")["Completion Steps"].mean()
            ax.plot(
                means.index,
                means.values,
                color=color,
                linestyle="--",
                label=f"{drone_count} Drones (Mean)",
            )

        ax.set_title(f"Completion Steps vs Flock Size (Spawn: {spawn_type})")
        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel("Completion Steps")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        # 2. Bar Chart: Success Rate vs Drone Count
        success_rate = group_data.groupby("Drone Count")["Success"].mean().reset_index()

        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        success_rate.set_index("Drone Count").plot.bar(
            y="Success", ax=ax_bar, legend=False, color="skyblue", edgecolor="black"
        )

        ax_bar.set_title(f"Success Rate by Number of Drones (Spawn: {spawn_type})")
        ax_bar.set_ylabel("Success Rate")
        ax_bar.set_xlabel("Number of Drones")
        ax_bar.set_ylim(0, 1.05)
        plt.xticks(rotation=0)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

    plt.show()
