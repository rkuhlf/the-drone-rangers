import pytest
import threading
import time
import random

from server import main, jobs_api
from planning import state
import numpy as np


# Mock dependencies
class MockWorldAdapter:
    def __init__(self):
        self.paused = True
        self.target = None
        self.polys = []
        self.N = 10
        self.P = np.zeros((10, 2))
        self.drones = np.zeros((1, 2))
        self._lock = threading.Lock()

    def get_state(self):
        with self._lock:
            # Simulate some work
            time.sleep(0.001)
            return state.State(
                flock=self.P, drones=self.drones, polygons=self.polys, jobs=[]
            )

    def pause(self):
        with self._lock:
            self.paused = not self.paused


@pytest.fixture
def app_concurrent():
    mock_world = MockWorldAdapter()

    # Use real jobs cache and repo (with temp file)
    import tempfile
    import os
    import pickle
    from pathlib import Path

    fd, db_path = tempfile.mkstemp()
    os.close(fd)

    original_db_path = jobs_api.DB_PATH
    jobs_api.DB_PATH = Path(db_path)
    with open(db_path, "wb") as f:
        pickle.dump([], f)
    jobs_api._REPO = None

    # Use real cache
    real_cache = main.JobCache()
    main.jobs_cache = real_cache

    original_adapter = main.backend_adapter
    main.backend_adapter = mock_world

    yield main.app, mock_world, db_path

    main.backend_adapter = original_adapter
    os.unlink(db_path)
    jobs_api.DB_PATH = original_db_path
    jobs_api._REPO = None


def test_concurrent_state_access(app_concurrent):
    """Test multiple threads hitting /state concurrently."""
    app, w, _ = app_concurrent
    client = app.test_client()

    errors = []

    def worker():
        for _ in range(50):
            try:
                res = client.get("/state")
                if res.status_code != 200:
                    errors.append(f"Status {res.status_code}")
                # Also try to patch occasionally
                if random.random() < 0.1:
                    client.patch("/state", json={"pause": True})
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent access: {errors[:5]}"


def test_concurrent_job_creation(app_concurrent):
    """Test multiple threads creating jobs concurrently."""
    app, _, _ = app_concurrent
    client = app.test_client()

    errors = []
    job_ids = []
    lock = threading.Lock()

    def worker():
        for i in range(10):
            try:
                payload = {
                    "target": {"type": "circle", "center": [0, 0], "radius": 10},
                    "drones": 1,
                    "job_type": "immediate",
                }
                res = client.post("/api/jobs", json=payload)
                if res.status_code == 201:
                    with lock:
                        job_ids.append(res.get_json()["id"])
                else:
                    errors.append(f"Status {res.status_code}")
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Should have created 50 jobs
    assert len(job_ids) == 50
    # Verify all are unique
    assert len(set(job_ids)) == 50

    # Verify list returns all
    res = client.get("/api/jobs")
    data = res.get_json()
    assert data["total"] == 50
