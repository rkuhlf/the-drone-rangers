import numpy as np

from simulation import world, scenarios
from planning import state
from planning.herding import policy


def test_full_simulation_flow():
    """
    End-to-End test of the simulation flow:
    1. Initialize World with a flock and a drone.
    2. Create a Job to move the flock to a target.
    3. Run the simulation loop.
    4. Verify the flock moves towards the target.
    """
    # 1. Setup
    seed = 123
    N = 30
    bounds = (0.0, 200.0, 0.0, 200.0)

    # Spawn sheep in bottom-left
    sheep_xy = scenarios.spawn_circle(N, center=(50, 50), radius=10.0, seed=seed)

    # Spawn drone near sheep
    drone_xy = np.array([[40.0, 40.0]])

    # Target in top-right
    target_pos = np.array([150.0, 150.0])
    target = state.Circle(center=target_pos, radius=15.0)

    # Initialize World
    w = world.World(
        sheep_xy, drone_xy, target_xy=target_pos, bounds=bounds, seed=seed, k_nn=10
    )

    # Initialize Policy
    total_area = 0.5 * w.N * (w.ra**2)
    collected_herd_radius = np.sqrt(total_area)
    pol = policy.ShepherdPolicy(
        fN=collected_herd_radius,
        umax=w.umax,
        too_close=1.5 * w.ra,
        collect_standoff=1.0 * w.ra,
    )

    # Create Job
    job = state.Job(
        target=target,
        drone_count=1,
        status="running",
        is_active=True,
        remaining_time=None,
        start_at=None,
        completed_at=None,
        scenario_id="e2e-test",
        maintain_until="target_is_reached",
        created_at=0,
        updated_at=0,
    )

    # 2. Run Simulation Loop
    steps = 50
    initial_dist = np.mean(np.linalg.norm(w.P - target_pos, axis=1))

    print(f"Initial mean distance to target: {initial_dist:.2f}")

    for i in range(steps):
        # Get current state
        s = w.get_state()
        s.jobs = [job]

        # Plan
        plan = pol.plan(s, [job], w.dt)

        # Step
        w.step(plan)

    # 3. Verify
    final_dist = np.mean(np.linalg.norm(w.P - target_pos, axis=1))
    print(f"Final mean distance to target: {final_dist:.2f}")

    assert final_dist < initial_dist, "Flock did not move towards target"
    assert np.isfinite(w.P).all()
    assert w.P.shape[0] == N


def test_obstacle_avoidance_flow():
    """
    E2E test with obstacle:
    Flock moves from left to right, obstacle in middle.
    Verify they don't pass through it.
    """
    seed = 42
    N = 20
    bounds = (0.0, 200.0, 0.0, 200.0)

    # Sheep at (50, 100)
    sheep_xy = scenarios.spawn_circle(N, center=(50, 100), radius=5.0, seed=seed)
    drone_xy = np.array([[40.0, 100.0]])

    # Target at (150, 100)
    target_pos = np.array([150.0, 100.0])
    target = state.Circle(center=target_pos, radius=15.0)

    # Obstacle blocking path: (90, 80) to (110, 120)
    obs = np.array([[90, 80], [110, 80], [110, 120], [90, 120]])

    w = world.World(
        sheep_xy,
        drone_xy,
        target_xy=target_pos,
        bounds=bounds,
        obstacles_polygons=[obs],
        seed=seed,
        k_nn=5,
    )

    pol = policy.ShepherdPolicy(
        fN=20.0, umax=w.umax, too_close=1.5 * w.ra, collect_standoff=1.0 * w.ra
    )

    job = state.Job(
        target=target,
        drone_count=1,
        status="running",
        is_active=True,
        remaining_time=None,
        start_at=None,
        completed_at=None,
        scenario_id="obs-test",
        maintain_until="target_is_reached",
        created_at=0,
        updated_at=0,
    )

    # Run loop
    steps = 100
    for i in range(steps):
        s = w.get_state()
        s.jobs = [job]
        plan = pol.plan(s, [job], w.dt)
        w.step(plan)

        # Check obstacle penetration at every step
        # Simple bounding box check: x in [90, 110], y in [80, 120]
        # Allow small margin for soft repulsion
        in_x = (w.P[:, 0] > 91) & (w.P[:, 0] < 109)
        in_y = (w.P[:, 1] > 81) & (w.P[:, 1] < 119)
        in_obs = in_x & in_y

        # Assert no sheep are deeply inside obstacle
        assert not np.any(in_obs), f"Sheep penetrated obstacle at step {i}"

    # Verify progress (should have moved right, maybe around obstacle)
    final_mean_x = np.mean(w.P[:, 0])
    assert final_mean_x > 50.0, "Flock failed to move right"
