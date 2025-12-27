import pytest

from datetime import datetime, timezone, timedelta
from flask import Flask
from server import jobs_api

import threading


# Mock dependencies
class MockJobCache:
    def __init__(self):
        self.list = []
        self.map = {}

    def add(self, job):
        self.list.append(job)
        self.map[job.id] = job

    def get(self, job_id):
        return self.map.get(job_id)


class MockWorldAdapter:
    pass


@pytest.fixture
def app_and_repo():
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

    cache = MockJobCache()
    world_lock = threading.Lock()
    adapter = MockWorldAdapter()

    app = Flask(__name__)
    bp = jobs_api.create_jobs_blueprint(world_lock, cache, lambda: adapter)
    app.register_blueprint(bp)

    yield app, jobs_api.get_repo(), db_path

    os.unlink(db_path)
    jobs_api.DB_PATH = original_db_path
    jobs_api._REPO = None


def test_create_scheduled_job(app_and_repo):
    app, repo, _ = app_and_repo
    client = app.test_client()

    # Schedule for 1 hour in future
    future_time = datetime.now(timezone.utc) + timedelta(hours=1)
    future_iso = future_time.isoformat().replace("+00:00", "Z")

    payload = {
        "target": {"type": "circle", "center": [0, 0], "radius": 10},
        "drone_count": 1,
        "job_type": "scheduled",
        "scheduled_time": future_iso,
    }

    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 201
    data = response.get_json()

    assert data["status"] == "scheduled"
    assert data["is_active"] is False
    assert data["start_at"] is not None


def test_create_scheduled_job_missing_time(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    payload = {
        "target": {"type": "circle", "center": [0, 0], "radius": 10},
        "drone_count": 1,
        "job_type": "scheduled",
        # Missing scheduled_time
    }

    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 400
    assert "scheduled_time" in response.get_json()["error"]


def test_create_immediate_job_explicit(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    payload = {
        "target": {"type": "circle", "center": [0, 0], "radius": 10},
        "drone_count": 1,
        "job_type": "immediate",
    }

    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 201
    data = response.get_json()

    assert data["status"] == "running"
    assert data["is_active"] is True
