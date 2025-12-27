import pytest
import numpy as np

from server import main
from planning import state


# Mock dependencies
class MockWorldAdapter:
    def __init__(self):
        self.paused = True
        self.target = None
        self.polys = []
        self.N = 10
        self.P = np.zeros((10, 2))
        self.drones = np.zeros((1, 2))

    def get_state(self):
        return state.State(
            flock=self.P, drones=self.drones, polygons=self.polys, jobs=[]
        )

    def pause(self):
        self.paused = not self.paused

    def clear_polygons(self):
        self.polys = []

    def add_polygons(self, polys):
        self.polys.extend(polys)

    def add_polygon(self, poly):
        self.polys.append(poly)


class MockJobCache:
    def __init__(self):
        self.list = []

    def reset_with(self, jobs):
        self.list = jobs


@pytest.fixture
def app_and_world():
    mock_world = MockWorldAdapter()
    mock_cache = MockJobCache()

    original_adapter = main.backend_adapter
    original_cache = main.jobs_cache

    main.backend_adapter = mock_world
    main.jobs_cache = mock_cache

    yield main.app, mock_world

    main.backend_adapter = original_adapter
    main.jobs_cache = original_cache


def test_patch_state_multi_field(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    # Update target, polygon, and pause all at once
    target = [80.0, 80.0]
    poly = [[0, 0], [10, 0], [0, 10]]
    payload = {"target": target, "polygon": poly, "pause": False}

    response = client.patch("/state", json=payload)
    assert response.status_code == 200

    # Verify all effects
    assert np.array_equal(w.target, target)
    assert len(w.polys) == 1
    assert w.paused is False


def test_patch_state_clear_and_add(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    # Add one first
    w.add_polygon([[0, 0], [1, 1], [1, 0]])
    assert len(w.polys) == 1

    # Clear and add new
    payload = {"clear": True, "polygon": [[10, 10], [20, 20], [20, 10]]}

    response = client.patch("/state", json=payload)
    assert response.status_code == 200

    # Should have cleared old and added new -> total 1
    assert len(w.polys) == 1
    assert w.polys[0][0][0] == 10


def test_patch_state_invalid_target(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    # Invalid target format
    payload = {"target": "invalid"}
    # The current implementation might crash or ignore.
    # Let's see if it handles it gracefully (500 or 400).
    # Ideally it should be 400.
    # main.py: target = np.array(data['target']) -> might raise ValueError

    # If it raises exception, Flask returns 500.
    # We'll assert it's not 200.
    response = client.patch("/state", json=payload)
    assert response.status_code != 200
