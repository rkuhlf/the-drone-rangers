import numpy as np

from planning import state


def test_circle_serialization():
    """Test Circle to_dict."""
    c = state.Circle(center=np.array([10.0, 20.0]), radius=5.0)
    d = c.to_dict()
    assert d["type"] == "circle"
    assert d["center"] == [10.0, 20.0]
    assert d["radius"] == 5.0


def test_polygon_serialization():
    """Test Polygon to_dict."""
    pts = np.array([[0, 0], [10, 0], [0, 10]])
    p = state.Polygon(points=pts)
    d = p.to_dict()
    assert d["type"] == "polygon"
    assert d["points"] == [[0, 0], [10, 0], [0, 10]]


def test_job_serialization():
    """Test Job to_dict."""
    target = state.Circle(center=np.array([0, 0]), radius=10)
    job = state.Job(
        target=target,
        drone_count=2,
        status="pending",
        is_active=True,
        remaining_time=120.0,
        start_at=1000.0,
        completed_at=None,
        scenario_id="scen-1",
        maintain_until="target_is_reached",
        created_at=1000.0,
        updated_at=1000.0,
    )
    d = job.to_dict()
    assert d["status"] == "pending"
    assert d["target"]["type"] == "circle"
    assert d["start_at"] is not None  # Should be ISO string
    assert d["completed_at"] is None
    assert d["maintain_until"] == "target_is_reached"


def test_state_serialization():
    """Test full State to_dict."""
    flock = np.array([[1, 1], [2, 2]])
    drones = np.array([[0, 0]])
    s = state.State(flock=flock, drones=drones, polygons=[], jobs=[])
    d = s.to_dict()
    assert len(d["flock"]) == 2
    assert len(d["drones"]) == 1
    assert d["jobs"] == []
