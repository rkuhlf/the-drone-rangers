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
    """Create a Flask app with mocked world."""
    # Mock globals in main
    mock_world = MockWorldAdapter()
    mock_cache = MockJobCache()

    original_adapter = main.backend_adapter
    original_cache = main.jobs_cache

    main.backend_adapter = mock_world
    main.jobs_cache = mock_cache

    yield main.app, mock_world

    # Restore
    main.backend_adapter = original_adapter
    main.jobs_cache = original_cache


def test_get_state(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    response = client.get("/state")
    assert response.status_code == 200
    data = response.get_json()

    assert "flock" in data
    assert "drones" in data
    assert data["paused"] is True


def test_patch_state_pause(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    # Unpause
    response = client.patch("/state", json={"pause": False})
    assert response.status_code == 200
    assert w.paused is False

    # Pause
    response = client.patch("/state", json={"pause": True})
    assert response.status_code == 200
    assert w.paused is True


def test_patch_state_target(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    target = [50.0, 50.0]
    response = client.patch("/state", json={"target": target})
    assert response.status_code == 200
    assert np.array_equal(w.target, target)


def test_patch_state_polygons(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    poly = [[0, 0], [10, 0], [0, 10]]
    response = client.patch("/state", json={"polygon": poly})
    assert response.status_code == 200
    assert len(w.polys) == 1

    # Clear
    response = client.patch("/state", json={"clear": True})
    assert response.status_code == 200
    assert len(w.polys) == 0


def test_play_pause_toggle(app_and_world):
    app, w = app_and_world
    client = app.test_client()

    # Initial: paused=True
    response = client.post("/pause")
    assert response.status_code == 200
    assert response.get_json()["paused"] is False
    assert w.paused is False

    response = client.post("/pause")
    assert response.status_code == 200
    assert response.get_json()["paused"] is True
    assert w.paused is True
