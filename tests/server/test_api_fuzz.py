import pytest
import random
import string

from flask import Flask
from server import jobs_api, scenarios_api
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
def app_fuzz():
    import tempfile
    import os
    import pickle
    from pathlib import Path

    # Setup Jobs DB
    fd1, jobs_db = tempfile.mkstemp()
    os.close(fd1)
    jobs_api.DB_PATH = Path(jobs_db)
    with open(jobs_db, "wb") as f:
        pickle.dump([], f)
    jobs_api._REPO = None

    # Setup Scenarios DB
    fd2, scen_db = tempfile.mkstemp()
    os.close(fd2)
    scenarios_api.DB_PATH = Path(scen_db)
    with open(scen_db, "wb") as f:
        pickle.dump([], f)
    scenarios_api.REPO = scenarios_api.ScenarioRepo()  # Reset repo

    # Setup App
    cache = MockJobCache()
    world_lock = threading.Lock()
    adapter = MockWorldAdapter()

    app = Flask(__name__)
    jobs_bp = jobs_api.create_jobs_blueprint(world_lock, cache, lambda: adapter)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(scenarios_api.scenarios_bp)

    yield app

    os.unlink(jobs_db)
    os.unlink(scen_db)


def random_string(length=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def random_json(depth=2):
    if depth == 0:
        return random.choice(
            [random_string(), random.randint(0, 100), True, False, None]
        )

    if random.random() < 0.5:
        return {random_string(): random_json(depth - 1) for _ in range(3)}
    else:
        return [random_json(depth - 1) for _ in range(3)]


def test_fuzz_jobs_endpoint(app_fuzz):
    client = app_fuzz.test_client()

    for _ in range(50):
        # Send random JSON
        payload = random_json()
        try:
            res = client.post("/api/jobs", json=payload)
            # Should be 400 (Bad Request) or 422 (Validation Error)
            # Or 201 if we accidentally hit a valid schema (unlikely)
            assert res.status_code in [400, 422, 201]
        except Exception as e:
            pytest.fail(f"Server crashed on payload {payload}: {e}")


def test_fuzz_scenarios_endpoint(app_fuzz):
    client = app_fuzz.test_client()

    for _ in range(50):
        payload = random_json()
        try:
            res = client.post("/scenarios", json=payload)
            assert res.status_code in [400, 422, 201]
        except Exception as e:
            pytest.fail(f"Server crashed on payload {payload}: {e}")


def test_fuzz_invalid_json(app_fuzz):
    client = app_fuzz.test_client()

    # Send malformed JSON (raw string)
    res = client.post(
        "/api/jobs", data="{invalid_json:", content_type="application/json"
    )
    # Flask usually handles this with 400
    assert res.status_code == 400
