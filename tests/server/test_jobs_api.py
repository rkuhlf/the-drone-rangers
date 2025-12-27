import pytest
import tempfile
import os
import pickle

from pathlib import Path
from flask import Flask
from server import jobs_api


class MockJobCache:
    def __init__(self):
        self.list = []
        self.map = {}

    def add(self, job):
        self.list.append(job)
        self.map[job.id] = job

    def get(self, job_id):
        return self.map.get(job_id)

    def remove(self, job_id):
        if job_id in self.map:
            job = self.map.pop(job_id)
            self.list.remove(job)
            return job
        return None


class MockWorldAdapter:
    def __init__(self):
        self.paused = True

    def unpause(self):
        self.paused = False


@pytest.fixture
def app_and_repo():
    """Create a Flask app with a temporary jobs repo."""
    # Create temp file for pickle DB
    fd, db_path = tempfile.mkstemp()
    os.close(fd)

    # Mock DB_PATH
    original_db_path = jobs_api.DB_PATH
    jobs_api.DB_PATH = Path(db_path)

    # Initialize empty repo
    with open(db_path, "wb") as f:
        pickle.dump([], f)

    # Reset global repo
    jobs_api._REPO = None

    # Mock other deps
    cache = MockJobCache()
    import threading

    world_lock = threading.Lock()
    adapter = MockWorldAdapter()

    app = Flask(__name__)
    bp = jobs_api.create_jobs_blueprint(world_lock, cache, lambda: adapter)
    app.register_blueprint(bp)

    yield app, cache, db_path

    # Cleanup
    os.unlink(db_path)
    jobs_api.DB_PATH = original_db_path
    jobs_api._REPO = None


def test_list_jobs_empty(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    response = client.get("/api/jobs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 0
    assert data["jobs"] == []


def test_create_job(app_and_repo):
    app, cache, _ = app_and_repo
    client = app.test_client()

    payload = {
        "target": {"type": "circle", "center": [100, 100], "radius": 10},
        "drone_count": 2,
        "job_type": "immediate",
    }

    response = client.post("/api/jobs", json=payload)
    if response.status_code != 201:
        print(f"Create Job Failed: {response.get_json()}")
    assert response.status_code == 201
    data = response.get_json()

    assert data["status"] == "running"
    assert data["drone_count"] == 2
    assert data["target"]["type"] == "circle"

    # Check cache updated
    assert len(cache.list) == 1
    assert str(cache.list[0].id) == data["id"]


def test_create_job_invalid(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    # Missing target
    response = client.post("/api/jobs", json={"drone_count": 1})
    assert response.status_code == 400

    # Invalid drone count
    payload = {
        "target": {"type": "circle", "center": [0, 0], "radius": 5},
        "drone_count": 0,
    }
    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 400


def test_get_job(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    # Create
    payload = {
        "target": {"type": "circle", "center": [100, 100], "radius": 10},
        "drone_count": 1,
    }
    create_res = client.post("/api/jobs", json=payload)
    assert create_res.status_code == 201
    job_id = create_res.get_json()["id"]

    # Get
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.get_json()["id"] == job_id


def test_update_job_status(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    # Create
    payload = {
        "target": {"type": "circle", "center": [100, 100], "radius": 10},
        "drone_count": 1,
    }
    create_res = client.post("/api/jobs", json=payload)
    assert create_res.status_code == 201
    job_id = create_res.get_json()["id"]

    # Update to paused (pending)
    response = client.patch(
        f"/api/jobs/{job_id}", json={"status": "pending", "is_active": False}
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "pending"
    assert response.get_json()["is_active"] is False


def test_delete_job(app_and_repo):
    app, cache, _ = app_and_repo
    client = app.test_client()

    # Create
    payload = {
        "target": {"type": "circle", "center": [100, 100], "radius": 10},
        "drone_count": 1,
    }
    create_res = client.post("/api/jobs", json=payload)
    assert create_res.status_code == 201
    job_id = create_res.get_json()["id"]

    # Delete
    response = client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200

    # Verify removed from cache
    assert len(cache.list) == 0

    # Verify 404 on get
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 404
