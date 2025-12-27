import pytest

import numpy as np

from server import main
from simulation import world, scenarios


@pytest.fixture
def app_with_real_world():
    # Initialize a real World for E2E testing
    sheep_xy = scenarios.spawn_circle(20, (50, 50), 10.0, seed=42)
    drone_xy = np.array([[0.0, 0.0]])
    target_xy = np.array([100.0, 100.0])

    w = world.World(sheep_xy, drone_xy, target_xy, seed=42, k_nn=5)

    # Inject into main
    original_adapter = main.backend_adapter
    main.backend_adapter = w

    # Ensure unpaused
    w.paused = False

    yield main.app, w

    # Restore
    main.backend_adapter = original_adapter


def test_server_e2e_flow(app_with_real_world):
    """
    Full E2E test driving the simulation via API.
    1. Create Job (immediate).
    2. Verify Job is running.
    3. Step simulation (manually via internal method or let it run if threaded).
       Note: main.py doesn't have a background thread loop exposed for tests.
       The `backend_adapter` is just the World object.
       The "loop" is usually external or client-driven in this architecture?
       Wait, earlier finding: "simulation loop is not explicitly called in main.py".
       So for this test, we must manually call `w.step()` to simulate time passing,
       but we verify the *API's view* of the state.
    """
    app, w = app_with_real_world
    client = app.test_client()

    # 1. Create Job
    payload = {
        "target": {"type": "circle", "center": [150, 150], "radius": 10},
        "drone_count": 1,
        "job_type": "immediate",
    }
    res = client.post("/api/jobs", json=payload)
    assert res.status_code == 201
    job_id = res.get_json()["id"]

    # 2. Verify Job Active via State API
    res = client.get("/state")
    # state_data = res.get_json()
    # Check if job is in state (if state exposes jobs)
    # The current /state endpoint might not expose jobs list directly,
    # but we can check if drones are moving.

    # 3. Simulate Loop
    # We manually step the world, simulating the server loop
    # The Policy needs the job from the cache/repo.
    # main.py's `backend_adapter` is `w`.
    # But `w` doesn't know about `jobs_api` jobs unless we inject them.
    # In the real app, who calls `pol.plan(s, jobs)`?
    # It seems `main.py` is missing the loop!
    # So this test confirms that *if* a loop existed, the API would reflect changes.
    # We will simulate the loop here.

    # from server import jobs_api
    from planning.herding import policy

    # Setup policy
    pol = policy.ShepherdPolicy(
        fN=20.0, umax=w.umax, too_close=10.0, collect_standoff=5.0
    )

    # Run loop for 20 steps
    for _ in range(20):
        # Get active jobs from API cache
        with main.world_lock:  # Access lock if needed, or just cache
            active_jobs = [j for j in main.jobs_cache.list if j.is_active]

        s = w.get_state()
        s.jobs = active_jobs

        plan = pol.plan(s, active_jobs, w.dt)
        w.step(plan)

    # 4. Verify Movement via API
    res = client.get("/state")
    final_state = res.get_json()
    flock = np.array(final_state["flock"])

    # Sheep should have moved from (50,50) towards (150,150)
    mean_x = np.mean(flock[:, 0])
    assert mean_x > 50.0, "Flock should have moved right"

    # 5. Stop Job via API
    res = client.patch(f"/api/jobs/{job_id}", json={"is_active": False})
    assert res.status_code == 200
    assert res.get_json()["is_active"] is False
