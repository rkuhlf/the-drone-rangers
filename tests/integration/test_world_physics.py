import numpy as np
import pytest
from simulation import world
from planning import plan_type


def test_world_initialization():
    """Test World initialization."""
    N = 20
    sheep_xy = np.zeros((N, 2))
    drone_xy = np.zeros((1, 2))
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42)

    assert w.N == N
    assert w.drones.shape[0] == 1
    assert np.array_equal(w.target, target_xy)
    assert w.P.shape == (N, 2)


def test_world_step_do_nothing():
    """Test World stepping with DoNothing plan."""
    N = 20
    sheep_xy = np.full((N, 2), 50.0)
    drone_xy = np.array([[0.0, 0.0]])
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42)

    # Step with no action
    plan = plan_type.DoNothing()
    w.step(plan)

    # Sheep shouldn't move much (just jitter/noise)
    new_pos = w.get_state().flock
    assert np.isfinite(new_pos).all()
    # Check that they haven't moved wildly
    assert np.all(np.abs(new_pos - sheep_xy) < 2.0)


def test_world_repulsion():
    """Test that sheep are repelled by drones."""
    N = 20
    # Sheep at (50, 50)
    sheep_xy = np.full((N, 2), 50.0)
    # Dog at (50.5, 50) -> Right of sheep
    drone_xy = np.array([[50.5, 50.0]])
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42, flock_init=1.0)

    # Capture initial state
    initial_pos = w.get_state().flock.copy()
    mean_x_before = np.mean(initial_pos[:, 0])

    # Force drone position update via plan with repulsion ON
    apply_repulsion = np.ones(1, dtype=bool)
    plan = plan_type.DronePositions(
        positions=drone_xy,
        apply_repulsion=apply_repulsion,
        target_sheep_indices=[],
        gcm=np.array([0, 0]),
        radius=0,
    )
    w.step(plan)

    # Sheep should move AWAY from drone (to the left, x < 50)
    new_pos = w.get_state().flock
    mean_x_after = np.mean(new_pos[:, 0])

    assert mean_x_after < mean_x_before


def test_world_bounds():
    """Test that sheep stay within bounds."""
    N = 10
    # Sheep near edge (0,0)
    sheep_xy = np.full((N, 2), 1.0)
    drone_xy = np.array([[0.0, 0.0]])
    target_xy = np.array([100.0, 100.0])

    # Bounds (0, 100, 0, 100)
    w = world.World(
        sheep_xy, drone_xy, target_xy, bounds=(0.0, 100.0, 0.0, 100.0), seed=42, k_nn=5
    )

    # Step multiple times to let them potentially hit the wall
    plan = plan_type.DoNothing()
    for _ in range(10):
        w.step(plan)

    final_pos = w.get_state().flock
    assert np.all(final_pos[:, 0] >= 0.0)
    assert np.all(final_pos[:, 1] >= 0.0)


def test_obstacle_collision():
    """Test that sheep avoid obstacles."""
    N = 10
    # Sheep at (50, 50) moving right
    sheep_xy = np.full((N, 2), 50.0)
    drone_xy = np.array([[0.0, 0.0]])
    target_xy = np.array([100.0, 100.0])

    # Obstacle at (60, 40) to (60, 60) blocking path to right
    obs = np.array([[60, 40], [70, 40], [70, 60], [60, 60]])

    w = world.World(
        sheep_xy, drone_xy, target_xy, obstacles_polygons=[obs], seed=42, k_nn=5
    )

    # Manually set velocity to move right towards obstacle
    w.V[:, 0] = 1.0  # Move right

    # Step
    plan = plan_type.DoNothing()
    for _ in range(5):
        w.step(plan)

    # Check that sheep didn't penetrate the obstacle (x < 60)
    # Allow small penetration due to soft repulsion, but should be pushed back
    final_pos = w.get_state().flock
    # Check if any sheep are significantly inside the obstacle box
    # Box x range: 60-70. Sheep started at 50.
    # They should be stopped around 60.
    assert np.all(final_pos[:, 0] < 61.0)


def test_vmax_clamping():
    """Test that velocity is clamped to vmax."""
    N = 10
    sheep_xy = np.zeros((N, 2))
    drone_xy = np.zeros((1, 2))

    w = world.World(sheep_xy, drone_xy, None, vmax=1.0, seed=42, k_nn=5)

    # Set huge velocity
    w.V[:] = 100.0

    # Step
    w.step(plan_type.DoNothing())

    # Check velocity magnitude
    speeds = np.linalg.norm(w.V, axis=1)
    assert np.all(speeds <= 1.0 + 1e-9)


def test_empty_world():
    """Test world with 0 sheep (should fail due to k_nn constraint)."""
    sheep_xy = np.zeros((0, 2))
    drone_xy = np.zeros((1, 2))

    # World requires N > k_nn, so N=0 should raise AssertionError
    with pytest.raises(AssertionError):
        world.World(sheep_xy, drone_xy, None, seed=42)
