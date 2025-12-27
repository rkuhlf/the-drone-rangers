import pytest
import sqlite3
import os
import tempfile

import numpy as np
from flask import Flask
from server.drone_management import create_drones_blueprint, _generate_next_drone_id


class MockWorld:
    def __init__(self):
        self.drones = np.zeros((0, 2))


@pytest.fixture
def app_and_db():
    """Create a Flask app with a temporary database."""
    # Create a temporary file for the database
    db_fd, db_path = tempfile.mkstemp()

    # Mock the DB_PATH in the module
    import server.drone_management

    original_db_path = server.drone_management.DB_PATH
    server.drone_management.DB_PATH = db_path

    # Mock World
    w = MockWorld()

    # Create app
    app = Flask(__name__)
    app.register_blueprint(create_drones_blueprint(w))

    yield app, w, db_path

    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)
    server.drone_management.DB_PATH = original_db_path


def test_list_drones_empty(app_and_db):
    app, w, _ = app_and_db
    client = app.test_client()

    response = client.get("/drones")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 0
    assert data["items"] == []


def test_create_drone(app_and_db):
    app, w, _ = app_and_db
    client = app.test_client()

    # Create drone
    payload = {"make": "DJI", "model": "Phantom"}
    response = client.post("/drones", json=payload)
    assert response.status_code == 201
    data = response.get_json()
    assert data["make"] == "DJI"
    assert data["model"] == "Phantom"
    assert data["id"] == "DR-001"

    # Check World updated
    assert w.drones.shape[0] == 1

    # Create another
    response = client.post("/drones", json=payload)
    assert response.status_code == 201
    assert response.get_json()["id"] == "DR-002"
    assert w.drones.shape[0] == 2


def test_create_drone_invalid(app_and_db):
    app, _, _ = app_and_db
    client = app.test_client()

    # Missing fields
    response = client.post("/drones", json={"make": "DJI"})
    assert response.status_code == 400

    response = client.post("/drones", json={"model": "Phantom"})
    assert response.status_code == 400


def test_delete_drone(app_and_db):
    app, w, _ = app_and_db
    client = app.test_client()

    # Create drone first
    client.post("/drones", json={"make": "DJI", "model": "Phantom"})
    assert w.drones.shape[0] == 1

    # Delete it
    response = client.delete("/drones/DR-001")
    assert response.status_code == 204

    # Check World updated
    assert w.drones.shape[0] == 0

    # Verify DB empty
    response = client.get("/drones")
    assert response.get_json()["total"] == 0


def test_delete_nonexistent_drone(app_and_db):
    app, _, _ = app_and_db
    client = app.test_client()

    response = client.delete("/drones/DR-999")
    assert response.status_code == 404


def test_id_generation():
    """Test ID generation logic directly."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE drones (id TEXT)")

    # Empty
    assert _generate_next_drone_id(conn) == "DR-001"

    # With existing
    conn.execute("INSERT INTO drones (id) VALUES ('DR-005')")
    assert _generate_next_drone_id(conn) == "DR-006"

    # With gaps
    conn.execute("INSERT INTO drones (id) VALUES ('DR-002')")
    assert _generate_next_drone_id(conn) == "DR-006"  # Still max + 1
