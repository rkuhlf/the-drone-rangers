import pytest
import json

import threading

from server import main
from planning import state
import numpy as np


# Mock World Adapter that supports blocking/yielding for stream
class MockStreamWorld:
    def __init__(self):
        self.paused = False
        self.N = 10
        self.P = np.zeros((10, 2))
        self.drones = np.zeros((1, 2))
        self.polys = []
        self._lock = threading.Lock()

    def get_state(self):
        with self._lock:
            return state.State(
                flock=self.P, drones=self.drones, polygons=self.polys, jobs=[]
            )


@pytest.fixture
def app_and_stream_world():
    mock_world = MockStreamWorld()

    # Mock the stream generator in main.py
    # The actual implementation uses a generator that yields SSE format
    # We need to ensure main.py's stream route uses this mock

    original_adapter = main.backend_adapter
    main.backend_adapter = mock_world

    yield main.app, mock_world

    main.backend_adapter = original_adapter


def test_stream_state(app_and_stream_world):
    app, w = app_and_stream_world
    client = app.test_client()

    # The stream endpoint is likely /stream/state
    # We need to verify it returns a generator

    # Note: Testing infinite streams with test_client can be tricky.
    # We usually check the first few chunks.

    response = client.get("/stream/state")
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"

    # Read a few lines
    # The generator yields bytes
    # We expect "data: {...}\n\n"

    chunks = []
    # Using direct iterator on response.response (which is the generator)
    # But test_client response might buffer.
    # Let's try to iterate response.response

    count = 0
    for chunk in response.response:
        if chunk:
            chunks.append(chunk.decode("utf-8"))
            count += 1
        if count >= 3:
            break

    assert len(chunks) > 0
    assert "data: {" in chunks[0]

    # Parse one json
    line = chunks[0].strip()
    if line.startswith("data: "):
        json_str = line[6:]
        data = json.loads(json_str)
        assert "flock" in data
        assert "drones" in data


def test_stream_headers(app_and_stream_world):
    app, _ = app_and_stream_world
    client = app.test_client()

    response = client.get("/stream/state")
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"
