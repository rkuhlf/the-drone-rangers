import pytest
import time
import numpy as np
from simulation import world, scenarios
from planning.herding import policy
from planning import state


def benchmark_simulation_step(N, steps=100):
    """Benchmark simulation stepping for N agents."""
    sheep_xy = scenarios.spawn_uniform(N, (0, 200, 0, 200), seed=42)
    drone_xy = np.array([[0.0, 0.0]])
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42, k_nn=5)
    pol = policy.ShepherdPolicy(
        fN=20.0, umax=w.umax, too_close=10.0, collect_standoff=5.0
    )

    # Create a dummy job
    target = state.Circle(center=target_xy, radius=10.0)
    job = state.Job(
        target=target,
        drone_count=1,
        status="running",
        is_active=True,
        remaining_time=None,
        start_at=None,
        completed_at=None,
        scenario_id="bench",
        maintain_until="target_is_reached",
        created_at=0,
        updated_at=0,
    )

    start_time = time.time()
    for _ in range(steps):
        s = w.get_state()
        s.jobs = [job]
        plan = pol.plan(s, [job], w.dt)
        w.step(plan)
    end_time = time.time()

    avg_time = (end_time - start_time) / steps
    return avg_time


def test_benchmark_small():
    """Benchmark N=50 (Standard)."""
    avg_time = benchmark_simulation_step(50, steps=50)
    print(f"\nN=50 Avg Step Time: {avg_time*1000:.2f} ms")
    # Should be fast (< 10ms usually)
    assert avg_time < 0.05


def test_benchmark_medium():
    """Benchmark N=200 (Large)."""
    avg_time = benchmark_simulation_step(200, steps=20)
    print(f"\nN=200 Avg Step Time: {avg_time*1000:.2f} ms")
    assert avg_time < 0.1


@pytest.mark.skip(reason="Too slow for regular CI")
def test_benchmark_large():
    """Benchmark N=1000 (Stress)."""
    avg_time = benchmark_simulation_step(1000, steps=10)
    print(f"\nN=1000 Avg Step Time: {avg_time*1000:.2f} ms")
    # Python implementation might struggle here, maybe < 500ms
    assert avg_time < 0.5
