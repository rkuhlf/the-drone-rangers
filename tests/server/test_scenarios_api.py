import pytest
import tempfile
import os
import pickle
from pathlib import Path
from flask import Flask
from server import scenarios_api


@pytest.fixture
def app_and_repo():
    """Create a Flask app with a temporary scenarios repo."""
    # Create temp file
    fd, db_path = tempfile.mkstemp()
    os.close(fd)

    # Mock DB_PATH
    original_db_path = scenarios_api.DB_PATH
    scenarios_api.DB_PATH = Path(db_path)

    # Initialize empty repo
    with open(db_path, "wb") as f:
        pickle.dump([], f)

    # Reset global repo
    scenarios_api.REPO = scenarios_api.ScenarioRepo()  # Re-init with new path

    app = Flask(__name__)
    app.register_blueprint(scenarios_api.scenarios_bp)

    yield app, scenarios_api.REPO, db_path

    # Cleanup
    os.unlink(db_path)
    scenarios_api.DB_PATH = original_db_path
    # We can't easily reset REPO to original state without reloading module,
    # but for tests it's fine as long as we don't run parallel tests on same process


def test_list_scenarios_empty(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    response = client.get("/scenarios")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 0
    assert data["items"] == []


def test_create_scenario(app_and_repo):
    app, repo, _ = app_and_repo
    client = app.test_client()

    payload = {
        "name": "Test Scenario",
        "entities": {
            "sheep": [[0, 0], [1, 1]],
            "drones": [[5, 5]],
            "targets": [[10, 10]],
        },
    }

    response = client.post("/scenarios", json=payload)
    if response.status_code != 201:
        print(f"Create Scenario Failed: {response.get_json()}")
    assert response.status_code == 201
    data = response.get_json()

    assert data["name"] == "Test Scenario"
    assert len(data["sheep"]) == 2

    # Check via API
    list_res = client.get("/scenarios")
    assert list_res.status_code == 200
    list_data = list_res.get_json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["id"] == data["id"]


def test_get_scenario(app_and_repo):
    app, _, _ = app_and_repo
    client = app.test_client()

    # Create
    payload = {"name": "Test", "entities": {"sheep": [[0, 0]], "drones": [[0, 0]]}}
    create_res = client.post("/scenarios", json=payload)
    assert create_res.status_code == 201
    scen_id = create_res.get_json()["id"]

    # Get
    response = client.get(f"/scenarios/{scen_id}")
    assert response.status_code == 200
    assert response.get_json()["id"] == scen_id


# DELETE route not implemented in API yet
# def test_delete_scenario(app_and_repo):
#     ...
