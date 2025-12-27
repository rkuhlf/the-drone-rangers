"""
Policy Evaluation Script

Runs the given policy on a variety of scenarios, aggregating:
- Whether the scenario could be completed.
- Completion time for successfully completed scenarios.
- Performance over n different random seeds for a scenario.
"""

import os
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from planning.herding import policy
from planning.herding.policy import is_goal_satisfied
from planning.run_demo import Renderer
from planning.state import Job, Circle
from simulation import world
from simulation.scenarios import (
    spawn_circle,
    spawn_clusters,
    spawn_corners,
    spawn_line,
    spawn_uniform,
)

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

BASE_CONFIG = {
    "max_steps": 2000,
    "boundary": "none",
    "clusters": 3,
    "dt": 1.0,
}

SPAWN_BOUNDS = (0.0, 250.0, 0.0, 250.0)
TARGET_POS = np.array([240, 240])
RESULTS_DIR = "./planning/results"


# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------


def run_one_trial(
    config: Dict[str, Any],
    spawn_type: str,
    seed: int,
    current_trial: int,
    total_trials: int,
    visualize: bool = False,
) -> Tuple[bool, int]:
    """
    Initializes and runs a single simulation trial for one scenario and seed.
    Returns a tuple: (was_successful, steps_taken).
    """
    # Spawn sheep based on the specified scenario
    if spawn_type == "circle":
        sheep_xy = spawn_circle(config["N"], center=(0, 0), radius=5.0, seed=seed)
    elif spawn_type == "uniform":
        sheep_xy = spawn_uniform(config["N"], SPAWN_BOUNDS, seed=seed)
    elif spawn_type == "clusters":
        sheep_xy = spawn_clusters(
            config["N"], config["clusters"], SPAWN_BOUNDS, spread=4.0, seed=seed
        )
    elif spawn_type == "corners":
        sheep_xy = spawn_corners(config["N"], SPAWN_BOUNDS, jitter=2.0, seed=seed)
    else:  # line
        sheep_xy = spawn_line(config["N"], SPAWN_BOUNDS, seed=seed)

    drone_xy = config["drone_xy"]
    target_xy = TARGET_POS

    # Build world with simulation parameters
    # Note: Disabling obstacles for evaluation by default (w_obs=0, etc.)
    world_kwargs = {
        **config,
        "bounds": SPAWN_BOUNDS,
        "seed": seed,
    }
    W = world.World(
        sheep_xy,
        drone_xy,
        target_xy.tolist(),
        w_obs=0,
        w_tan=0,
        keep_out=0,
        world_keep_out=0,
        wall_follow_boost=0,
        stuck_speed_ratio=0,
        near_wall_ratio=0,
        **world_kwargs,
    )

    # Initialize the herding policy
    total_area = 0.5 * W.N * (W.ra**2)
    collected_herd_radius = np.sqrt(total_area)

    shepherd_policy = policy.ShepherdPolicy(
        fN=collected_herd_radius,
        umax=W.umax,
        too_close=1.5 * W.ra,
        collect_standoff=1.0 * W.ra,
        conditionally_apply_repulsion=config["conditionally_apply_repulsion"],
    )

    # Create a Job with a Circle target
    success_radius = config["success_radius"]
    target = Circle(center=target_xy.copy(), radius=success_radius)
    current_time = time.time()
    num_drones = drone_xy.shape[0]
    jobs = [
        Job(
            target=target,
            remaining_time=None,
            is_active=True,
            drone_count=num_drones,
            status="running",
            start_at=None,
            completed_at=None,
            scenario_id=None,
            maintain_until="target_is_reached",
            created_at=current_time,
            updated_at=current_time,
        )
    ]

    renderer = None
    if visualize:
        renderer = Renderer(W, bounds=SPAWN_BOUNDS)

    # Main simulation loop for this trial
    for t in range(config["max_steps"]):
        state = W.get_state()
        plan = shepherd_policy.plan(state, jobs, W.dt)
        W.step(plan)

        # Check for success condition using is_goal_satisfied
        state = W.get_state()
        if is_goal_satisfied(state, target):
            return True, t  # Success!

        if visualize:
            if t % 2 == 0 and renderer:
                renderer.render_world(W, plan, t, target, debug=False)
            plt.pause(0.01)

        # Print the progress on a single line
        # Calculate farthest distance for display
        farthest = np.max(np.linalg.norm(state.flock - target.center, axis=1))
        progress_str = (
            f"  Trial {current_trial + 1:>2}/{total_trials} | "
            f"Step: {t + 1:<5}/{config['max_steps']}, "
            f"Flock Distance: {farthest:.0f}/{success_radius:.0f}      "
        )
        print(progress_str, end="\r", flush=True)

    if visualize:
        plt.ioff()
        plt.show()

    return False, config["max_steps"]  # Failure due to timeout


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    # Define evaluation parameters
    Ns = list(range(2, 150, 5))
    n_values = list(range(1, 149, 2))
    spawn_types = ["uniform"]
    seeds = range(3)
    flyovers = [True]
    drone_xy_configs = [
        np.array([[0, 0], [0, 100]]),
    ]

    # Generate scenarios
    scenarios_to_run = [
        {
            **BASE_CONFIG,
            "drone_xy": d_xy,
            "k_nn": n_nb,
            "conditionally_apply_repulsion": flyover,
            "N": N,
            "spawn_type": pattern,
            "seed": seed,
            "success_radius": N ** (1 / 2) * 4,
        }
        for seed, N, pattern, flyover, n_nb, d_xy in product(
            seeds, Ns, spawn_types, flyovers, n_values, drone_xy_configs
        )
        if n_nb < N and n_nb < 0.53 * N
    ]

    trial_results_list = []

    print(f"\nRunning evaluation: {len(scenarios_to_run)} scenarios.")
    print("-" * 65)

    for s_idx, config in enumerate(scenarios_to_run):
        # Time the execution of a single trial
        start_time = time.perf_counter()
        success, completion_steps = run_one_trial(
            config,
            config["spawn_type"],
            config["seed"],
            s_idx,
            len(scenarios_to_run),
            visualize=False,
        )
        end_time = time.perf_counter()
        trial_duration = end_time - start_time

        trial_results_list.append(
            {
                "Success": success,
                "Completion Steps": completion_steps if success else np.nan,
                "Wall Time (s)": trial_duration,
                "Drone Count": config["drone_xy"].shape[0],
                **config,
            }
        )

    # Create and save the detailed, trial-by-trial DataFrame
    trials_df = pd.DataFrame(trial_results_list)
    output_csv_file = f"{RESULTS_DIR}/{date_str}--evaluation_trials.csv"
    trials_df.to_csv(output_csv_file, index=False)
    print(f"\nTrial-by-trial results saved to '{output_csv_file}'")

    # Generate and print the aggregate summary
    if not trials_df.empty:
        summary_df = (
            trials_df.groupby("spawn_type")
            .agg(
                Trials=("seed", "count"),
                Successes=("Success", "sum"),
                Avg_Steps=("Completion Steps", "mean"),
                Avg_Wall_Time=("Wall Time (s)", "mean"),
            )
            .reset_index()
        )

        # Format for printing
        summary_df["Success Rate"] = summary_df["Successes"] / summary_df["Trials"]
        summary_df["Successes"] = summary_df.apply(
            lambda row: f"{row['Successes']}/{row['Trials']}", axis=1
        )
        summary_df = summary_df[
            ["spawn_type", "Successes", "Success Rate", "Avg_Steps", "Avg_Wall_Time"]
        ]
        summary_df.rename(
            columns={"Avg_Steps": "Avg Steps", "Avg_Wall_Time": "Avg Time (s)"},
            inplace=True,
        )

        print("\n" + "=" * 85)
        print(" " * 32 + "EVALUATION SUMMARY")
        print("=" * 85)
        print(summary_df.to_string(index=False, float_format="%.2f"))
        print("=" * 85)
    else:
        print("\nNo results to summarize.")
