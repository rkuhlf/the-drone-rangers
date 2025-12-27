"""
Target Distance Visualization

This script visualizes the relationship between target distance from [125, 125]
and mean completion steps, based on evaluation results.
"""

import os
import sys
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CSV_PATH = "./planning/results/2025-12-08--15-36-29--evaluation_trials.csv"
REFERENCE_POINT = np.array([125.0, 125.0])


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def parse_target_xy(target_str: str) -> np.ndarray:
    """
    Parse target_xy string like "[240. 240.]" into numpy array.
    """
    # Extract numbers from the string
    numbers = re.findall(r"[\d.]+", target_str)
    if len(numbers) >= 2:
        return np.array([float(numbers[0]), float(numbers[1])])
    return np.array([0.0, 0.0])


def calculate_distance_to_reference(
    target_xy: np.ndarray, reference: np.ndarray
) -> float:
    """Calculate Euclidean distance from target to reference point."""
    return float(np.linalg.norm(target_xy - reference))


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

    # Calculate distances from Target_X and Target_Y (or parse target_xy if needed)
    if "Target_X" in df.columns and "Target_Y" in df.columns:
        # Use the already-parsed columns
        df["target_xy_parsed"] = df.apply(
            lambda row: np.array([row["Target_X"], row["Target_Y"]]), axis=1
        )
    else:
        # Fallback: parse target_xy column
        df["target_xy_parsed"] = df["target_xy"].apply(parse_target_xy)

    df["distance_to_ref"] = df["target_xy_parsed"].apply(
        lambda xy: calculate_distance_to_reference(xy, REFERENCE_POINT)
    )

    # Group by distance and calculate mean completion steps
    # Round distances to nearest integer for grouping (or use bins)
    df["distance_rounded"] = df["distance_to_ref"].round(0).astype(int)

    # Calculate mean completion steps for each distance
    mean_steps = df.groupby("distance_rounded")["Completion Steps"].mean().reset_index()
    mean_steps.columns = ["Distance", "Mean Completion Steps"]

    # Sort by distance
    mean_steps = mean_steps.sort_values("Distance")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line with markers
    ax.plot(
        mean_steps["Distance"] / 1000,
        mean_steps["Mean Completion Steps"] / 60,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="Mean Completion Steps",
    )

    # Also show individual data points as scatter for context
    ax.scatter(
        df["distance_rounded"] / 1000,
        df["Completion Steps"] / 60,
        alpha=0.6,
        color="lightblue",
        s=30,
        label="Individual Trials",
    )

    ax.set_title("Time Required to Herd a Given Distance")
    ax.set_xlabel("Distance Traveled (km)")
    ax.set_ylabel("Time (mins)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
