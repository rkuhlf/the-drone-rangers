import pytest
import os

import tempfile
from pathlib import Path
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


class MockWorldAdapter:
    pass


@pytest.fixture
def app_with_corrupt_db():
    fd, db_path = tempfile.mkstemp()
    os.close(fd)

    # Write garbage to DB
    with open(db_path, "wb") as f:
        f.write(b"NOT A PICKLE FILE")

    original_db_path = jobs_api.DB_PATH
    jobs_api.DB_PATH = Path(db_path)
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


def test_recover_from_corrupt_db(app_with_corrupt_db):
    app, repo, db_path = app_with_corrupt_db

    # The repo should handle corruption.
    # Current implementation: pickle.load(f) might raise UnpicklingError or EOFError.
    # Let's see if it crashes or returns empty list.
    # If it crashes, we should fix the implementation to be robust.

    try:
        repo.list()
        # If it returns, great.
    except Exception as e:
        # If it fails, we should modify the code to handle it.
        # But for this test, we assert failure if the code isn't robust yet.
        # Ideally, we want it to be robust.
        pytest.fail(f"Repo crashed on corrupt DB: {e}")

    # If we are here, it handled it (or we fixed it).
    # If the current code doesn't handle it, this test will fail,
    # and I will fix the code.


def test_recover_from_empty_file(app_with_corrupt_db):
    # Test empty file
    app, repo, db_path = app_with_corrupt_db
    with open(db_path, "wb"):
        pass  # Empty

    try:
        repo.list()
    except Exception as e:
        pytest.fail(f"Repo crashed on empty DB: {e}")
