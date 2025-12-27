"""
Simulation Demo Runner

This script runs a visual demonstration of the herding simulation with the configured
policy and scenario. It uses Matplotlib for real-time rendering of the agents,
obstacles, and targets.
"""

import argparse
import time
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from planning import plan_type
from planning import state as state_module
from planning.herding import policy
from planning.state import Job
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

DEFAULT_BOUNDS = (0.0, 500.0, 0.0, 500.0)
SPAWN_BOUNDS = (0.0, 250.0, 0.0, 250.0)
TARGET_XY = np.array([240.0, 240.0])
DEFAULT_DRONE_POS = np.array([[-20, -36]])


# -----------------------------------------------------------------------------
# Renderer Class
# -----------------------------------------------------------------------------


class Renderer:
    """Handles the visualization of the simulation state using Matplotlib."""

    def __init__(
        self,
        world_instance: world.World,
        bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
        obstacles_polygons: Optional[List[np.ndarray]] = None,
    ):
        """Initialize figure, axes, and scatter plots."""
        xmin, xmax, ymin, ymax = bounds

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        # Draw world bounds (fence)
        self.ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            linestyle="--",
        )

        # Draw polygon obstacles
        self.polygon_patches = []
        if obstacles_polygons:
            for poly in obstacles_polygons:
                # Close the polygon by adding the first vertex at the end
                closed_poly = np.vstack([poly, poly[0]])
                patch = MplPolygon(
                    closed_poly[:-1], facecolor="red", alpha=0.3, edgecolor="red"
                )
                self.ax.add_patch(patch)
                self.polygon_patches.append(patch)

        # Initial state
        state = world_instance.get_state()
        self.sheep_sc = self.ax.scatter(state.flock[:, 0], state.flock[:, 1], s=20)
        self.drone_sc = self.ax.scatter(
            [state.drones[:, 0]], [state.drones[:, 1]], marker="x"
        )
        self.target: Optional[Any] = None
        self.prev_target: Optional[state_module.Target] = None

        # Debug circle for GCM/Cohesion
        self.circle = plt.Circle((0, 0), 0, color="b", fill=False)
        self.ax.add_patch(self.circle)

    def render_world(
        self,
        world_instance: world.World,
        plan: plan_type.Plan,
        step_number: int,
        target: Optional[state_module.Target],
        debug: bool = False,
    ):
        """Update the scatter plots for the current state of the world."""
        state = world_instance.get_state()

        # Update sheep positions
        self.sheep_sc.set_offsets(state.flock)

        if debug:
            if isinstance(plan, plan_type.DronePositions):
                # Highlight target sheep if specified
                colors = [(0.0, 0.0, 1.0, 1.0)] * len(state.flock)  # all blue
                for i in plan.target_sheep_indices:
                    if 0 <= i < len(colors):
                        colors[i] = (1.0, 0.0, 0.0, 1.0)  # target sheep red
                self.sheep_sc.set_facecolor(colors)

                self.circle.center = plan.gcm
                self.circle.radius = plan.radius
            elif isinstance(plan, plan_type.DoNothing):
                pass
            else:
                # Should not happen with current types
                pass

        # Update drone and target markers
        self.drone_sc.set_offsets(state.drones)

        # Render the target, but only if it's different from the last time.
        if self.prev_target != target:
            if self.target is not None:
                self.target.remove()
                self.target = None

            if isinstance(target, state_module.Circle):
                self.target = plt.Circle((0, 0), 0, color="g", fill=False)
                self.target.center = target.center
                self.target.radius = target.radius
                self.ax.add_patch(self.target)
            elif isinstance(target, state_module.Polygon):
                self.target = MplPolygon(
                    target.points, facecolor="g", alpha=0.3, edgecolor="g"
                )
                self.ax.add_patch(self.target)
            self.prev_target = target

        # Title
        self.ax.set_title(f"Step {step_number}")

        # Redraw
        self.fig.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run herding simulation demo.")
    p.add_argument("--N", type=int, default=100, help="Number of sheep")
    p.add_argument(
        "--spawn",
        choices=["circle", "uniform", "clusters", "corners", "line"],
        default="uniform",
        help="Initial sheep distribution",
    )
    p.add_argument(
        "--clusters", type=int, default=3, help="# clusters for spawn=clusters"
    )
    p.add_argument("--seed", type=int, default=3, help="Random seed")
    p.add_argument("--steps", type=int, default=10000, help="Max simulation steps")
    p.add_argument(
        "--obstacles",
        action="store_true",
        default=True,
        help="Enable obstacles (default: True)",
    )
    p.add_argument(
        "--no-obstacles",
        dest="obstacles",
        action="store_false",
        help="Disable obstacles",
    )
    args = p.parse_args()

    # Bounds (match World defaults so plotting aligns)
    xmin, xmax, ymin, ymax = SPAWN_BOUNDS

    # --- Choose spawn pattern ---
    if args.spawn == "circle":
        sheep_xy = spawn_circle(args.N, center=(100, 100), radius=5.0, seed=args.seed)
    elif args.spawn == "uniform":
        sheep_xy = spawn_uniform(args.N, SPAWN_BOUNDS, seed=args.seed)
    elif args.spawn == "clusters":
        sheep_xy = spawn_clusters(
            args.N, args.clusters, SPAWN_BOUNDS, spread=4.0, seed=args.seed
        )
    elif args.spawn == "corners":
        sheep_xy = spawn_corners(args.N, SPAWN_BOUNDS, jitter=2.0, seed=args.seed)
    else:  # line
        sheep_xy = spawn_line(args.N, SPAWN_BOUNDS, seed=args.seed)

    drone_xy = DEFAULT_DRONE_POS
    target_xy = TARGET_XY

    # Create example polygon obstacles
    obstacles_polygons = None
    if args.obstacles:
        # Rectangle
        rect = np.array(
            [
                [50.0, 100.0],
                [50.0, 50.0],
                [100.0, 50.0],
                [100.0, 100.0],
            ]
        )

        # Triangle (unused in current list but available)
        triangle = np.array([[150.0, 50.0], [180.0, 50.0], [165.0, 80.0]])

        # L-shape (unused in current list but available)
        l_shape = np.array(
            [
                [50.0, 150.0],
                [90.0, 150.0],
                [90.0, 170.0],
                [70.0, 170.0],
                [70.0, 200.0],
                [50.0, 200.0],
            ]
        )

        obstacles_polygons = [rect]

    # Initialize World
    W = world.World(
        sheep_xy,
        drone_xy,
        target_xy.tolist(),
        seed=args.seed,
        obstacles_polygons=obstacles_polygons,
        obstacle_influence=30.0,
        w_obs=5.0,
        w_tan=12.0,
        keep_out=5.0,
        world_keep_out=5.0,
        wall_follow_boost=6.0,
        stuck_speed_ratio=0.08,
        near_wall_ratio=0.8,
        k_nn=8,
        dt=1,
    )

    # Initialize Policy
    total_area = 0.5 * W.N * (W.ra**2)
    collected_herd_radius = np.sqrt(total_area)

    shepherd_policy = policy.ShepherdPolicy(
        fN=collected_herd_radius,  # cohesion radius
        umax=W.umax,  # keep in sync with world
        too_close=1.5 * W.ra,  # safety stop
        collect_standoff=1.0 * W.ra,  # paper: r_a behind the stray
        conditionally_apply_repulsion=True,
    )

    s0 = W.get_state()
    drone_count = s0.drones.shape[0]

    current_time = time.time()

    # Setup Job and Target
    target = state_module.Circle(center=target_xy.copy(), radius=10.0)

    jobs = [
        Job(
            target=target,
            remaining_time=None,
            is_active=True,
            drone_count=drone_count,
            status="running",
            start_at=None,
            completed_at=None,
            scenario_id=None,
            maintain_until="target_is_reached",
            created_at=current_time,
            updated_at=current_time,
        )
    ]

    renderer = Renderer(W, bounds=DEFAULT_BOUNDS, obstacles_polygons=obstacles_polygons)

    # Main Loop
    for t in range(args.steps):
        plan = shepherd_policy.plan(W.get_state(), jobs, W.dt)
        W.step(plan)

        if t % 2 == 0:
            state = W.get_state()
            renderer.render_world(W, plan, t, jobs[0].target, debug=True)

        plt.pause(0.01)

    plt.ioff()
    plt.show()
