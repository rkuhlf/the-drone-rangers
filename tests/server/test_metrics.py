import pytest

from server import metrics
from planning import state
import numpy as np


@pytest.fixture
def collector():
    """Get a fresh collector for each test."""
    # Reset singleton
    metrics._COLLECTOR = None
    return metrics.get_collector()


def test_start_stop_run(collector):
    run = collector.start_run("test-run")
    assert run.run_id == "test-run"
    assert collector.current_run is not None
    assert collector.current_run.run_id == "test-run"
    assert collector.get_current_run() == run

    # Stop
    stopped_run = collector.end_run()
    assert stopped_run == run
    assert collector.current_run is None
    assert "test-run" in collector.completed_runs


def test_record_step(collector):
    collector.start_run("test-run")

    # Create dummy state
    s = state.State(
        flock=np.array([[0, 0], [10, 10]]),
        drones=np.array([[5, 5]]),
        polygons=[],
        jobs=[],
    )

    # Record step
    # record_step(world_state, target, t, fN)
    collector.record_step(s, None, 1.0, 20.0)

    run = collector.get_current_run()
    assert len(run.steps) == 1
    step = run.steps[0]
    # Check metrics calculation
    # GCM of (0,0) and (10,10) is (5,5)
    # Cohesion: mean dist to GCM. dists: sqrt(50) ~= 7.07. mean = 7.07
    # cohesiveness = fN / spread_radius = 20.0 / 7.07 ~= 2.82
    assert abs(step.cohesiveness - 2.82) < 0.1


def test_summary_generation(collector):
    collector.start_run("test-run")

    # Step 1: spread out
    s1 = state.State(
        flock=np.array([[0, 0], [100, 100]]),
        drones=np.array([[50, 50]]),
        polygons=[],
        jobs=[],
    )
    collector.record_step(s1, None, 1.0, 20.0)

    # Step 2: closer
    s2 = state.State(
        flock=np.array([[40, 40], [60, 60]]),
        drones=np.array([[50, 50]]),
        polygons=[],
        jobs=[],
    )
    collector.record_step(s2, None, 2.0, 20.0)

    run = collector.end_run()
    summary = run.summary

    assert summary["num_steps"] == 2
    assert summary["avg_cohesiveness"] > 0
