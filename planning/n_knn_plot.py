"""
N vs. k-NN Plotting Script

Analyzes simulation results to visualize the relationship between the number of agents (N)
and the number of nearest neighbors (k-NN) on the success rate of shepherding.
Generates a heatmap and overlays theoretical guide curves.
"""


import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# List of CSV files to process.
# Can be populated manually or via glob patterns.
CSV_PATHS = ["./planning/results/3_drones_2000_steps.csv"]

# Plot settings
FIG_SIZE = (10, 5.5)
CMAP = "Greens"
X_LIMIT = (0, 150)
Y_LIMIT = (0, 150)


# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------


def load_and_process_data(paths: List[str]) -> pd.DataFrame:
    """Load CSVs, normalize data, and aggregate into a single DataFrame."""
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: File not found: {p}")
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

        # Normalize Success to 0/1
        if df["Success"].dtype != np.number:
            df["Success"] = (
                df["Success"].astype(str).str.lower().map({"true": 1, "false": 0})
            )

        # Ensure N and k_nn are ints for grouping stability
        if "N" in df.columns:
            df["N"] = df["N"].astype(int)
        if "k_nn" in df.columns:
            df["k_nn"] = df["k_nn"].astype(int)

        # Synthesize seed if missing
        if "seed" not in df.columns:
            df["seed"] = np.arange(len(df))

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate: keep first occurrence of (N, k_nn, seed)
    if {"N", "k_nn", "seed"}.issubset(all_df.columns):
        all_df = all_df.drop_duplicates(subset=["N", "k_nn", "seed"])

    return all_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_results(df: pd.DataFrame):
    """Generate and display the success rate plot."""
    if df.empty:
        print("No data to plot.")
        return

    # Aggregate success rate per (N, k_nn)
    agg = df.groupby(["N", "k_nn"], as_index=False).agg(
        success_rate=("Success", "mean"), trials=("Success", "size")
    )

    # Keep only valid combos
    agg = agg[agg["k_nn"] < agg["N"]]

    if agg.empty:
        print("No valid (N, k_nn) pairs after filtering.")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Scatter "heatmap": one square per (N, k_nn)
    sc = ax.scatter(
        agg["N"],
        agg["k_nn"],
        c=agg["success_rate"],
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        s=55,  # marker size; tweak to taste
        marker="s",  # square markers look like pixels
        edgecolors="none",
    )

    # Guide curves
    N_line = np.linspace(1, 150, 500)
    ax.plot(N_line, 0.53 * N_line, "k-", lw=2, zorder=3, label="n = 0.53N")
    ax.plot(N_line, 3.0 * np.log(N_line), "k--", lw=2, zorder=3, label="n = 3log(N)")

    # Axes & limits
    ax.set_xlabel("No. Agents (N)")
    ax.set_ylabel("No. Neighbors (n)")
    ax.set_xlim(X_LIMIT)
    ax.set_ylim(Y_LIMIT)

    # (optional) make axes aspect closer to equal so the triangle looks “true”
    # ax.set_aspect("equal", adjustable="box")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion of Successes")

    ax.legend(loc="upper left")
    plt.title("Proportion of Herding Tasks Completed")

    plt.tight_layout()
    plt.show()

    if "Wall Time (s)" in df.columns:
        mean_time = df.loc[df["Success"] == 1, "Wall Time (s)"].mean()
        print(f"Average time (successful runs only): {mean_time:.3f} s")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    data = load_and_process_data(CSV_PATHS)

    if data.empty:
        print("No valid data found in the specified CSV files.")
        sys.exit(1)

    print(f"Loaded {len(data)} records.")
    plot_results(data)
