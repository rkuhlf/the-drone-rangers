import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CSV_PATH = "./planning/results/3_drones_500_sheep_10_200.csv"
N_TARGET = 500

FIG_SIZE = (7.5, 4.5)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Ensure Success is numeric
if df["Success"].dtype != np.number:
    df["Success"] = df["Success"].astype(str).str.lower().map({"true": 1, "false": 0})

# Filter to N = 500
df = df[df["N"] == N_TARGET]

if df.empty:
    raise RuntimeError(f"No data found for N = {N_TARGET}")

# ------------------------------------------------------------------
# AGGREGATE ACROSS SEEDS
# ------------------------------------------------------------------
agg = (
    df.groupby("k_nn", as_index=False)
    .agg(
        mean_success=("Success", "mean"),
        std_success=("Success", "std"),
        trials=("Success", "count"),
    )
    .sort_values("k_nn")
)

# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIG_SIZE)

# Compute 95% CI for a binomial proportion
p = agg["mean_success"].values  # in [0,1]
n = agg["trials"].values.astype(float)

se = np.sqrt(p * (1 - p) / n)  # standard error
z = 1.96  # ~95% CI
lower = np.clip(p - z * se, 0.0, 1.0)
upper = np.clip(p + z * se, 0.0, 1.0)

# Main line
ax.plot(
    agg["k_nn"],
    100 * p,
    marker="o",
    lw=2,
    label="Completion rate",
)

# Confidence band
ax.fill_between(
    agg["k_nn"],
    100 * lower,
    100 * upper,
    alpha=0.2,
    color=ax.lines[0].get_color(),
    linewidth=0,
)

# Axes & labels
ax.set_xlabel("Number of nearest neighbors (k)")
ax.set_ylabel("Completion rate (%)")
ax.set_ylim(0, 105)
ax.set_xticks(agg["k_nn"])  # integer ticks at your k values
ax.set_title("Completion Rate vs Local Neighborhood Size\n(N = 500, 3 drones)")

ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()
