"""
Module-Level Integrity Tests

This test suite verifies the correctness and functionality of the beautified modules.
It covers:
1. Planning Utils (Vector math)
2. State & Plan Types (Dataclasses)
3. Simulation Scenarios (Spawning)
4. Simulation World (Physics & Logic)
5. Herding Policy (Planning logic)
"""


import sys

import numpy as np
import pytest


# Import modules to test
from planning.herding import utils
from planning import state, plan_type
from planning.herding import policy
from simulation import world, scenarios

# -----------------------------------------------------------------------------
# 1. Planning Utils Tests
# -----------------------------------------------------------------------------


def test_utils_norm():
    """Test vector norm calculation."""
    # Test single vector
    v = np.array([3.0, 4.0])
    n = utils.norm(v)
    expected = np.array([0.6, 0.8])
    np.testing.assert_allclose(n, expected, atol=1e-9)


def test_utils_smooth_push():
    """Test smooth push force calculation."""
    # Case 1: Distance > radius (no force)
    f1 = utils.smooth_push(dist=10.0, rs=5.0)
    assert f1 == 0.0

    # Case 2: Distance = 0 (peak force)
    f2 = utils.smooth_push(dist=0.0, rs=5.0)
    assert f2 == 1.0

    # Case 3: Distance = radius (approx no force)
    f3 = utils.smooth_push(dist=5.0, rs=5.0)
    assert f3 < 1e-6

    # Case 4: Mid-range
    f4 = utils.smooth_push(dist=2.5, rs=5.0)
    assert 0.0 < f4 < 1.0


# -----------------------------------------------------------------------------
# 2. State & Plan Types Tests
# -----------------------------------------------------------------------------


def test_state_dataclasses():
    """Verify State dataclasses structure and serialization."""
    # Circle
    c = state.Circle(center=np.array([10, 10]), radius=5.0)
    assert np.array_equal(c.center, np.array([10, 10]))
    assert c.radius == 5.0
    assert c.to_dict() == {"type": "circle", "center": [10, 10], "radius": 5.0}

    # Job
    job = state.Job(
        target=c,
        drone_count=2,
        status="pending",
        is_active=True,
        remaining_time=100.0,
        start_at=1000.0,
        completed_at=None,
        scenario_id="test_scen",
        maintain_until="target_is_reached",
        created_at=1000.0,
        updated_at=1000.0,
    )
    d = job.to_dict()
    assert d["status"] == "pending"
    assert d["target"]["type"] == "circle"


def test_plan_types():
    """Verify Plan dataclasses."""
    # DoNothing
    dn = plan_type.DoNothing()
    assert isinstance(dn, plan_type.DoNothing)

    # DronePositions
    pos = np.array([[1, 2], [3, 4]])
    dp = plan_type.DronePositions(
        positions=pos,
        apply_repulsion=True,
        target_sheep_indices=[0, 1],
        gcm=np.array([2, 3]),
        radius=10.0,
    )
    assert np.array_equal(dp.positions, pos)
    assert dp.apply_repulsion is True


# -----------------------------------------------------------------------------
# 3. Simulation Scenarios Tests
# -----------------------------------------------------------------------------


def test_spawn_uniform():
    """Test uniform spawning."""
    N = 100
    bounds = (0, 100, 0, 100)
    pts = scenarios.spawn_uniform(N, bounds, seed=42)

    assert pts.shape == (N, 2)
    assert np.all(pts[:, 0] >= bounds[0])
    assert np.all(pts[:, 0] <= bounds[1])
    assert np.all(pts[:, 1] >= bounds[2])
    assert np.all(pts[:, 1] <= bounds[3])


def test_spawn_circle():
    """Test circle spawning."""
    N = 50
    center = (50, 50)
    radius = 10.0
    pts = scenarios.spawn_circle(N, center, radius, seed=42)

    assert pts.shape == (N, 2)
    # Check all points are within radius (approx)
    dists = np.linalg.norm(pts - np.array(center), axis=1)
    assert np.all(dists <= radius + 1e-9)


# -----------------------------------------------------------------------------
# 4. Simulation World Tests
# -----------------------------------------------------------------------------


def test_world_initialization():
    """Test World initialization."""
    # Ensure N > k_nn (default 8)
    N = 20
    sheep_xy = np.zeros((N, 2))
    drone_xy = np.zeros((1, 2))
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42)

    assert w.N == N
    assert w.drones.shape[0] == 1
    assert np.array_equal(w.target, target_xy)
    assert w.P.shape == (N, 2)


def test_world_step():
    """Test World stepping logic (basic physics check)."""
    # Setup: 20 sheep (N > k_nn), 1 drone far away
    N = 20
    sheep_xy = np.full((N, 2), 50.0)
    drone_xy = np.array([[0.0, 0.0]])  # Far away
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42)

    # Step with no action
    plan = plan_type.DoNothing()
    w.step(plan)

    # Sheep shouldn't move much (just jitter)
    new_pos = w.get_state().flock
    # With 20 sheep at same spot, repulsion forces might be high
    # But we just want to ensure it runs without error and produces numbers
    assert np.isfinite(new_pos).all()


def test_world_repulsion():
    """Test that sheep are repelled by drones."""
    # Setup: 20 sheep, 1 drone very close
    N = 20
    sheep_xy = np.full((N, 2), 50.0)
    drone_xy = np.array([[50.5, 50.0]])  # Very close, to the right
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42, flock_init=1.0)

    # Capture initial state from the world itself
    initial_pos = w.get_state().flock.copy()
    mean_x_before = np.mean(initial_pos[:, 0])

    # Force drone position update via plan
    # apply_repulsion must be an array matching number of drones
    apply_repulsion = np.ones(1, dtype=bool)

    plan = plan_type.DronePositions(
        positions=drone_xy,
        apply_repulsion=apply_repulsion,
        target_sheep_indices=[],
        gcm=np.array([0, 0]),
        radius=0,
    )
    w.step(plan)

    # Sheep should move AWAY from drone (to the left)
    # Check mean position shift
    new_pos = w.get_state().flock
    mean_x_after = np.mean(new_pos[:, 0])

    # Since drone is at 50.5 (right) and sheep at 50.0, they should move left (< 50.0)
    assert mean_x_after < mean_x_before


# -----------------------------------------------------------------------------
# 5. Herding Policy Tests
# -----------------------------------------------------------------------------


def test_policy_plan():
    """Test ShepherdPolicy planning."""
    # Setup state
    flock = np.random.rand(10, 2) * 50
    drones = np.array([[0.0, 0.0]])

    # Setup job
    target = state.Circle(center=np.array([100.0, 100.0]), radius=10.0)
    job = state.Job(
        target=target,
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

    s = state.State(flock=flock, drones=drones, polygons=[], jobs=[job])

    # Initialize policy with required args
    pol = policy.ShepherdPolicy(fN=20.0, umax=2.0, too_close=10.0, collect_standoff=5.0)

    # Generate plan
    plan = pol.plan(s, [job], dt=1.0)

    assert isinstance(plan, plan_type.DronePositions)
    assert plan.positions.shape == (1, 2)
    # Drone should move (not stay at 0,0)
    assert not np.array_equal(plan.positions, drones)


def test_policy_geometry():
    """Test geometric helpers in policy."""
    # Points inside polygon
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    pts = np.array([[5, 5], [15, 15], [5, -5]])
    inside = policy.points_inside_polygon(pts, poly)
    assert np.array_equal(inside, [True, False, False])

    # Closest point on polygon
    # Must pass 2D array for points
    pt = np.array([[15.0, 5.0]])  # Outside right edge
    closest = policy.closest_point_on_polygon(pt, poly)
    expected = np.array([[10.0, 5.0]])
    np.testing.assert_allclose(closest, expected)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
