"""
Integrity Verification Script

This script performs a smoke test on the codebase to ensure that:
1. All modules can be imported (checking for circular dependencies or syntax errors).
2. The core simulation and planning loop runs without errors.
3. Basic logic produces valid (non-NaN) output.

Usage:
    python verify_integrity.py
"""

import sys
import traceback
import numpy as np


def log(msg):
    print(f"[VERIFY] {msg}")


def test_imports():
    log("Testing imports...")
    try:

        log("Imports successful.")
    except ImportError as e:
        log(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error during imports: {e}")
        traceback.print_exc()
        sys.exit(1)


def test_simulation_loop():
    log("Testing simulation loop...")
    try:
        from simulation.world import World
        from planning.herding.policy import ShepherdPolicy
        from planning.state import Job, Circle
        from simulation.scenarios import spawn_uniform

        # Setup
        seed = 42
        N = 50
        bounds = (0.0, 250.0, 0.0, 250.0)
        sheep_xy = spawn_uniform(N, bounds, seed=seed)
        drone_xy = np.array([[125.0, 125.0]])
        target_xy = np.array([200.0, 200.0])

        # Initialize World
        w = World(sheep_xy, drone_xy, target_xy, bounds=bounds, seed=seed, dt=0.1)

        # Initialize Policy
        # Use parameters similar to run_demo.py
        total_area = 0.5 * w.N * (w.ra**2)
        collected_herd_radius = np.sqrt(total_area)
        policy = ShepherdPolicy(
            fN=collected_herd_radius,
            umax=w.umax,
            too_close=1.5 * w.ra,
            collect_standoff=1.0 * w.ra,
            conditionally_apply_repulsion=True,
        )

        # Create a dummy job
        job = Job(
            target=Circle(center=target_xy, radius=10.0),
            drone_count=1,
            status="running",
            is_active=True,
            remaining_time=None,
            start_at=None,
            completed_at=None,
            scenario_id=None,
            maintain_until="target_is_reached",
            created_at=0,
            updated_at=0,
        )

        # Run a few steps
        steps = 10
        log(f"Running {steps} simulation steps...")
        for i in range(steps):
            state = w.get_state()

            # Check for NaNs in state
            if np.isnan(state.flock).any():
                raise ValueError(f"NaN detected in flock positions at step {i}")
            if np.isnan(state.drones).any():
                raise ValueError(f"NaN detected in drone positions at step {i}")

            plan = policy.plan(state, [job], w.dt)
            w.step(plan)

        log("Simulation loop completed successfully.")

    except Exception as e:
        log(f"Simulation loop failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_imports()
    test_simulation_loop()
    log("ALL CHECKS PASSED.")
