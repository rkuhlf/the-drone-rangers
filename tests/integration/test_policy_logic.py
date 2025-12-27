import numpy as np

from planning.herding import policy
from planning import state, plan_type


def test_policy_plan_structure():
    """Test ShepherdPolicy.plan() returns correct structure."""
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

    # Initialize policy
    pol = policy.ShepherdPolicy(fN=20.0, umax=2.0, too_close=10.0, collect_standoff=5.0)

    # Generate plan
    plan = pol.plan(s, [job], dt=1.0)

    assert isinstance(plan, plan_type.DronePositions)
    assert plan.positions.shape == (1, 2)
    assert isinstance(plan.apply_repulsion, np.ndarray)
    assert plan.apply_repulsion.dtype == bool or plan.apply_repulsion.dtype == int


def test_policy_no_active_job():
    """Test policy returns DoNothing when no active job."""
    flock = np.random.rand(10, 2) * 50
    drones = np.array([[0.0, 0.0]])
    s = state.State(flock=flock, drones=drones, polygons=[], jobs=[])

    pol = policy.ShepherdPolicy(fN=20.0, umax=2.0, too_close=10.0, collect_standoff=5.0)

    plan = pol.plan(s, [], dt=1.0)
    assert isinstance(plan, plan_type.DoNothing)


def test_multi_drone_assignment():
    """Test policy with multiple drones."""
    flock = np.random.rand(20, 2) * 50
    drones = np.array([[0.0, 0.0], [10.0, 10.0]])  # 2 drones

    target = state.Circle(center=np.array([100.0, 100.0]), radius=10.0)
    job = state.Job(
        target=target,
        drone_count=2,
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

    pol = policy.ShepherdPolicy(fN=20.0, umax=2.0, too_close=10.0, collect_standoff=5.0)

    plan = pol.plan(s, [job], dt=1.0)

    assert isinstance(plan, plan_type.DronePositions)
    assert plan.positions.shape == (2, 2)
    assert len(plan.target_sheep_indices) == 2
    # Ensure they target different sheep (heuristic check, not strictly guaranteed but likely)
    # Actually, policy might assign same sheep if it's the only outlier, but usually distributes.
    # Just check indices are valid
    assert all(0 <= idx < 20 for idx in plan.target_sheep_indices)


def test_polygon_target():
    """Test policy with Polygon target."""
    flock = np.random.rand(10, 2) * 50
    drones = np.array([[0.0, 0.0]])

    # Polygon target
    poly_pts = np.array([[100, 100], [120, 100], [120, 120], [100, 120]])
    target = state.Polygon(points=poly_pts)

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

    pol = policy.ShepherdPolicy(fN=20.0, umax=2.0, too_close=10.0, collect_standoff=5.0)

    plan = pol.plan(s, [job], dt=1.0)

    assert isinstance(plan, plan_type.DronePositions)
    # Should produce valid positions
    assert np.isfinite(plan.positions).all()
