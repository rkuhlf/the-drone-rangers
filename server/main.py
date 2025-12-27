# flake8: noqa: E402
"""
Drone Rangers Simulation Server

This is the main entry point for the Flask backend server.
It handles:
- Simulation state management
- API endpoints for scenarios, jobs, and metrics
- Real-time state streaming via SSE
- CORS configuration
"""


import sys
from pathlib import Path

# Add project root to Python path so imports work
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json  # noqa: E402
import random
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

from planning import herding, state
from planning.policy_configs import POLICY_PRESETS, build_policy
from server import jobs_api
from server.drone_management import create_drones_blueprint
from server.metrics import end_metrics_run, get_collector, start_metrics_run
from server.scenario_types import (
    SCENARIO_TYPES,
    generate_initial_layout,
    get_scenario_type,
)
from server.scenarios_api import REPO, Scenario, scenarios_bp
from simulation import world

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_FLOCK_SIZE = 50
DEFAULT_SHEEP_X_RANGE = (0, 200)
DEFAULT_SHEEP_Y_RANGE = (0, 200)
# -----------------------------------------------------------------------------
# Simulation Timing & Speed Configuration
# -----------------------------------------------------------------------------

# Simulation Loop Speed (Heartbeat)
# How many times per second the server "wakes up" to update the world.
# - Higher (e.g. 60) = Smoother motion, more frequent updates, higher CPU usage.
# - Lower (e.g. 30) = Choppier motion, saves CPU.
SIMULATION_LOOP_FPS = 60
FRAME_TIME = 1.0 / SIMULATION_LOOP_FPS

# Simulation Speed Multiplier (Physics Steps per Heartbeat)
# How much "simulation time" passes for every "heartbeat".
#
# Formula: Simulation Speed = (SIMULATION_LOOP_FPS * STEPS_PER_FRAME * dt) / 1 second
#
# - Loop FPS: 60 (wakes up 60 times/sec)
# - Steps/Frame: 3 (does 3 physics steps each time)
# - Physics dt: 0.07s (each step advances sim time by 0.07s)
#
# Result: 60 * 3 = 180 steps per real second.
# Total Sim Time: 180 * 0.07s = 12.6 seconds of sim time per 1 real second.
# Speed: ~12.6x real-time.
STEPS_PER_FRAME = 3

#  Broadcast Speed (Network/Rendering)
# How many times per second we send updates to the frontend.
# Keep this lower (e.g., 30) to save network bandwidth and frontend rendering power.
BROADCAST_FPS = 30
BROADCAST_FRAME_TIME = 1.0 / BROADCAST_FPS

# Allowed origins for CORS (development)
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------

# Thread-safe lock for world reinitialization
world_lock = threading.RLock()
current_scenario_id: Optional[str] = None  # Track what scenario is currently loaded

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


class JobCache:
    """Cache for jobs that maintains both list and O(1) dict lookup."""

    def __init__(self, initial: Optional[List[state.Job]] = None):
        self.list: List[state.Job] = []
        self.map: Dict[Any, state.Job] = {}
        if initial:
            for j in initial:
                self.add(j)

    def add(self, job: state.Job) -> state.Job:
        """Add a job, ensuring no duplicates by ID."""
        if job.id in self.map:
            return self.map[job.id]
        self.list.append(job)
        self.map[job.id] = job
        return job

    def get(self, job_id: Any) -> Optional[state.Job]:
        """Get job by ID in O(1) time."""
        return self.map.get(job_id)

    def remove(self, job_id: Any) -> Optional[state.Job]:
        """Remove job by ID."""
        # Remove from map first
        j = self.map.pop(job_id, None)

        if j is not None:
            # Remove from list by finding the index using ID comparison (not object comparison)
            # This avoids issues with numpy array comparisons in job objects
            for i, job in enumerate(self.list):
                if job.id == job_id:
                    del self.list[i]
                    break

        return j

    def clear(self):
        """Clear all jobs from the cache."""
        self.list.clear()
        self.map.clear()

    def reset_with(self, jobs: List[state.Job]):
        """Clear and replace with new jobs (maintains same cache object reference)."""
        self.clear()
        for j in jobs:
            self.add(j)


def _create_policy_for_world(w: world.World) -> herding.ShepherdPolicy:
    """
    Create a herding policy matched to the given world's flock size.
    """
    total_area = 0.5 * w.N * (w.ra**2)
    collected_herd_radius = np.sqrt(total_area)
    return herding.ShepherdPolicy(
        fN=collected_herd_radius,
        umax=w.umax,
        too_close=1.5 * w.ra,
        collect_standoff=1.0 * w.ra,
    )


def initialize_sim():
    """Initialize a new simulation with default parameters."""
    flock_size = DEFAULT_FLOCK_SIZE
    # Calculate appropriate k_nn based on flock size (must be <= N-1)
    k_nn = min(21, max(1, flock_size - 1))

    backend_adapter = world.World(
        sheep_xy=np.array(
            [
                [
                    random.uniform(*DEFAULT_SHEEP_X_RANGE),
                    random.uniform(*DEFAULT_SHEEP_Y_RANGE),
                ]
                for _ in range(flock_size)
            ]
        ),
        shepherd_xy=np.array([[0.0, 0.0]]),
        target_xy=None,  # No target by default - user must set via frontend
        boundary="none",
        k_nn=k_nn,
    )
    backend_adapter.paused = True  # Start paused since there's no target yet
    policy = _create_policy_for_world(backend_adapter)

    # Load existing jobs from database (for recovery after restart)
    repo = jobs_api.get_repo()
    db_jobs = repo.list()

    # Filter out jobs with invalid targets (must be Circle or Polygon, or None)
    valid_jobs = []
    for job in db_jobs:
        if job.target is None or isinstance(job.target, (state.Circle, state.Polygon)):
            valid_jobs.append(job)
        else:
            print(
                f"Warning: Discarding job {job.id} with invalid target type: {type(job.target)}"
            )

    # Filter: keep running and scheduled jobs for recovery
    # Completed/cancelled jobs are kept in DB but don't need to be in-memory
    active_jobs = [
        j for j in valid_jobs if j.status in ("running", "scheduled", "pending")
    ]

    return backend_adapter, policy, JobCache(active_jobs)


def _ensure_polygon_array(poly_like):
    """Validate and normalize polygon input to Nx2 array with N>=3."""
    arr = np.asarray(poly_like, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("polygon must be an Nx2 array with N>=3")
    # Accept closed rings; drop duplicate closing vertex if present
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def get_allowed_origin():
    """Get the allowed origin from the request, or return the first allowed origin."""
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        return origin
    # Default to first allowed origin if no match or no origin header
    return ALLOWED_ORIGINS[0]


# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------

backend_adapter, policy, jobs_cache = initialize_sim()

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(scenarios_bp)
app.register_blueprint(create_drones_blueprint(backend_adapter))

# Register jobs API blueprint
# Pass a lambda to always get the current global backend_adapter (which changes on scenario load)
api_jobs_bp = jobs_api.create_jobs_blueprint(
    world_lock, jobs_cache, lambda: backend_adapter
)
app.register_blueprint(api_jobs_bp)


# -----------------------------------------------------------------------------
# Middleware / CORS
# -----------------------------------------------------------------------------


@app.before_request
def handle_preflight():
    """Handle CORS preflight OPTIONS requests."""
    if request.method == "OPTIONS":
        response = Response()
        allowed_origin = get_allowed_origin()
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PATCH, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Cache-Control, Idempotency-Key"
        )
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response


@app.after_request
def after_request(response):
    """Add CORS headers to all responses."""
    # For streaming responses, headers are set directly in the Response object
    if response.mimetype == "text/event-stream":
        return response

    origin = request.headers.get("Origin")
    if origin:
        allowed_origin = get_allowed_origin()
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PATCH, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Cache-Control, Idempotency-Key"
        )
        response.headers["Access-Control-Expose-Headers"] = (
            "Content-Type, Cache-Control"
        )
    return response


# -----------------------------------------------------------------------------
# State & Control Endpoints
# -----------------------------------------------------------------------------


@app.route("/state", methods=["GET", "OPTIONS"])
def get_state():
    """Get current simulation state with pause status."""
    if request.method == "OPTIONS":
        return Response(status=200)
    with world_lock:
        state = backend_adapter.get_state()
        state.jobs = jobs_cache.list
        state_dict = state.to_dict()
        state_dict["paused"] = backend_adapter.paused
        return jsonify(state_dict)


@app.route("/stream/state", methods=["GET", "OPTIONS"])
def stream_state():
    """Stream simulation state via Server-Sent Events (SSE)."""
    if request.method == "OPTIONS":
        return Response(status=200)

    def generate():
        while True:
            try:
                # Acquire lock for thread-safe state access
                with world_lock:
                    state = backend_adapter.get_state()
                    state.jobs = jobs_cache.list
                    state_dict = state.to_dict()
                    state_dict["paused"] = backend_adapter.paused

                # Format as SSE event with event type
                event_data = f"event: stateUpdate\ndata: {json.dumps(state_dict)}\n\n"
                yield event_data

                # Sleep to maintain target FPS (30Hz for network/rendering)
                time.sleep(BROADCAST_FRAME_TIME)

            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                print("Error in stream_state:", e)
                # Log error but continue streaming with keepalive
                yield ": keepalive\n\n"
                time.sleep(FRAME_TIME)

    allowed_origin = get_allowed_origin()
    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Cache-Control, Idempotency-Key",
            "Access-Control-Expose-Headers": "Content-Type, Cache-Control",
        },
    )
    return response


@app.route("/state", methods=["PATCH"])
def patch_state():
    """Update simulation state (polygons, target, pause)."""
    data = request.get_json(silent=True) or {}
    print(data)

    with world_lock:
        # 1) Clear polygons
        try:
            if data.get("clear") is True:
                backend_adapter.clear_polygons()
        except Exception as e:
            return (
                jsonify({"ok": False, "error": f"failed to clear polygons: {e}"}),
                400,
            )

        # 2) Add multiple polygons
        if "polygons" in data:
            polys_in = data["polygons"]
            if not isinstance(polys_in, list) or len(polys_in) == 0:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "'polygons' must be a non-empty list of Nx2 arrays",
                        }
                    ),
                    400,
                )
            try:
                polys = [_ensure_polygon_array(p) for p in polys_in]
                backend_adapter.add_polygons(polys)
            except ValueError as ve:
                return jsonify({"ok": False, "error": str(ve)}), 400
            except Exception as e:
                return (
                    jsonify({"ok": False, "error": f"failed to add polygons: {e}"}),
                    400,
                )

        # 3) Add single polygon
        if "polygon" in data:
            try:
                poly = _ensure_polygon_array(data["polygon"])
                backend_adapter.add_polygon(poly)
            except ValueError as ve:
                return jsonify({"ok": False, "error": str(ve)}), 400
            except Exception as e:
                return (
                    jsonify({"ok": False, "error": f"failed to add polygon: {e}"}),
                    400,
                )

        # 4) Set target
        if "target" in data:
            try:
                pos = np.asarray(data["target"], dtype=float).reshape(2)
            except Exception:
                return jsonify({"ok": False, "error": "'target' must be [x, y]"}), 400
            try:
                backend_adapter.target = (
                    pos  # let world clamp/bounds as it already does
                )
            except Exception as e:
                return (
                    jsonify({"ok": False, "error": f"failed to set target: {e}"}),
                    400,
                )

        # 5) Set pause (boolean setter, not toggle)
        if "pause" in data:
            try:
                backend_adapter.paused = bool(data["pause"])
            except Exception as e:
                return jsonify({"ok": False, "error": f"failed to set pause: {e}"}), 400

        # Return the same shape as GET /state to avoid regressions
        try:
            return jsonify(backend_adapter.get_state().to_dict()), 200
        except Exception as e:
            return (
                jsonify({"ok": False, "error": f"failed to serialize state: {e}"}),
                500,
            )


@app.route("/pause", methods=["POST"])
def play_pause():
    """Toggle pause state (play/pause)."""
    with world_lock:
        backend_adapter.pause()
        paused_state = backend_adapter.paused
    return jsonify({"paused": paused_state}), 200


@app.route("/restart", methods=["POST"])
def restart_sim():
    """Restart simulation with default parameters."""
    global backend_adapter, policy, current_scenario_id

    with world_lock:
        backend_adapter, policy, new_cache = initialize_sim()
        # Reset the existing cache with new jobs (maintains same cache object reference)
        jobs_cache.reset_with(new_cache.list)
        current_scenario_id = None  # Clear any loaded scenario
        # Ensure it starts paused since there's no target
        backend_adapter.paused = True

    return jsonify(backend_adapter.get_state().to_dict())


# -----------------------------------------------------------------------------
# Scenario Management
# -----------------------------------------------------------------------------


@app.route("/load-scenario/<uuid:scenario_id>", methods=["POST"])
def load_scenario(scenario_id):
    """Load a scenario from the repository into the running simulation."""
    global backend_adapter, policy, current_scenario_id, jobs_cache

    scenario = REPO.get(scenario_id)
    if not scenario:
        return (
            jsonify({"error": {"type": "NotFound", "message": "scenario not found"}}),
            404,
        )

    # Validate scenario has required entities
    if not scenario.sheep or len(scenario.sheep) == 0:
        return (
            jsonify(
                {
                    "error": {
                        "type": "Validation",
                        "message": "scenario must have at least one sheep",
                    }
                }
            ),
            422,
        )
    if not scenario.drones or len(scenario.drones) == 0:
        return (
            jsonify(
                {
                    "error": {
                        "type": "Validation",
                        "message": "scenario must have at least one drone",
                    }
                }
            ),
            422,
        )

    try:
        with world_lock:
            # Calculate appropriate k_nn based on flock size
            # k_nn must be <= N-1, where N is number of sheep
            num_sheep = len(scenario.sheep)
            k_nn = min(21, max(1, num_sheep - 1))

            # Build world_params: start with defaults
            world_params = {
                "k_nn": k_nn,
                "boundary": scenario.boundary,
                "bounds": scenario.bounds,
            }

            # 1. Overlay base config from scenario type (if any)
            if scenario.scenario_type:
                st = get_scenario_type(scenario.scenario_type)
                if st and st.default_world_config:
                    world_params.update(st.default_world_config)

            # 2. Apply overrides and multipliers from scenario.world_config
            if scenario.world_config:
                # Separate multipliers from direct overrides
                multipliers = {
                    k: v
                    for k, v in scenario.world_config.items()
                    if k.endswith("_multiplier")
                }
                overrides = {
                    k: v
                    for k, v in scenario.world_config.items()
                    if not k.endswith("_multiplier")
                }

                # Apply direct overrides first
                world_params.update(overrides)

                # Apply multipliers to the current values (using World defaults as fallback)
                if "wa_multiplier" in multipliers:
                    base = world_params.get("wa", 1.05)
                    world_params["wa"] = base * multipliers["wa_multiplier"]

                if "wr_multiplier" in multipliers:
                    base = world_params.get("wr", 50.0)  # Default wr is high for sheep
                    world_params["wr"] = base * multipliers["wr_multiplier"]

                if "wd_multiplier" in multipliers:
                    # Map wd_multiplier to w_align
                    base = world_params.get("w_align", 0.0)
                    world_params["w_align"] = base * multipliers["wd_multiplier"]

            # Convert obstacles to world format
            obstacles_polygons = []
            if scenario.obstacles:
                for obs in scenario.obstacles:
                    # Handle both list-of-points and dict-with-polygon-key formats
                    if isinstance(obs, list):
                        poly = np.array(obs, dtype=float)
                        obstacles_polygons.append(poly)
                    elif isinstance(obs, dict) and "polygon" in obs:
                        poly = np.array(obs["polygon"], dtype=float)
                        obstacles_polygons.append(poly)

            # Initialize world with scenario data and config
            backend_adapter = world.World(
                sheep_xy=np.array(scenario.sheep, dtype=float),
                shepherd_xy=np.array(scenario.drones, dtype=float),
                target_xy=(
                    np.array(scenario.targets[0], dtype=float)
                    if scenario.targets
                    else None
                ),
                obstacles_polygons=obstacles_polygons if obstacles_polygons else None,
                **world_params,
            )

            # Build policy using the new config system
            policy = build_policy(backend_adapter, scenario.policy_config)

            # Initialize jobs for the scenario
            # Convert target position to a Circle target (required by Job type)
            target = None
            if scenario.targets and len(scenario.targets) > 0:
                target_pos = np.array(scenario.targets[0], dtype=float)
                # Create a Circle target with the position and a reasonable radius
                target = state.Circle(center=target_pos, radius=policy.fN * 3.0)

            # Create an in-memory job for the simulation (NOT persisted to database)
            # This keeps simulation state separate from farm jobs
            now = datetime.now(timezone.utc).timestamp()
            scenario_job = state.Job(
                id=uuid4(),
                target=target,
                remaining_time=None,
                is_active=False,  # Inactive until user clicks "Start Job"
                drone_count=len(scenario.drones),
                status="pending",  # Pending until user starts the job
                start_at=None,
                completed_at=None,
                scenario_id=str(scenario_id),  # Link job to the loaded scenario
                maintain_until="target_is_reached",
                created_at=now,
                updated_at=now,
            )
            # Reset jobs cache with only the scenario job (fresh start)
            # Important: use reset_with() to maintain the same cache object reference
            jobs_cache.reset_with([scenario_job])

            # Track what's loaded
            current_scenario_id = str(scenario_id)

            # Don't pause - let the simulation start immediately
            # Users can set a target on the map to begin herding
            backend_adapter.paused = False

        return (
            jsonify(
                {
                    "ok": True,
                    "loaded_scenario_id": str(scenario_id),
                    "scenario_name": scenario.name,
                    "num_sheep": len(scenario.sheep),
                    "drone_count": len(scenario.drones),
                    "boundary": scenario.boundary,
                    "has_target": (
                        len(scenario.targets) > 0 if scenario.targets else False
                    ),
                    "world_config_applied": scenario.world_config is not None,
                    "policy_config_applied": scenario.policy_config is not None,
                    "paused": backend_adapter.paused,
                }
            ),
            200,
        )

    except Exception as e:
        error_details = {
            "type": "ServerError",
            "message": f"Failed to load scenario: {str(e)}",
            "scenario_id": str(scenario_id),
            "scenario_found": scenario is not None,
            "scenario_data": {
                "name": scenario.name if scenario else None,
                "sheep_count": (
                    len(scenario.sheep) if scenario and scenario.sheep else 0
                ),
                "drone_count": (
                    len(scenario.drones) if scenario and scenario.drones else 0
                ),
                "targets_count": (
                    len(scenario.targets) if scenario and scenario.targets else 0
                ),
                "boundary": scenario.boundary if scenario else None,
                "bounds": scenario.bounds if scenario else None,
                "sheep_sample": (
                    scenario.sheep[:3] if scenario and scenario.sheep else None
                ),
                "drones_sample": (
                    scenario.drones[:3] if scenario and scenario.drones else None
                ),
                "targets_sample": (
                    scenario.targets[:3] if scenario and scenario.targets else None
                ),
            },
            "traceback": traceback.format_exc(),
        }
        print(f"Load scenario error: {error_details}")
        return jsonify({"error": error_details}), 500


@app.route("/current-scenario", methods=["GET"])
def get_current_scenario():
    """Get information about the currently loaded scenario (if any)."""
    with world_lock:
        scenario_id = current_scenario_id

    if scenario_id is None:
        return (
            jsonify(
                {
                    "loaded": False,
                    "scenario_id": None,
                    "message": "No scenario loaded (using default initialization)",
                }
            ),
            200,
        )

    scenario = REPO.get(scenario_id)
    if not scenario:
        return (
            jsonify(
                {
                    "loaded": False,
                    "scenario_id": scenario_id,
                    "message": "Previously loaded scenario no longer exists in repository",
                }
            ),
            200,
        )

    return (
        jsonify(
            {
                "loaded": True,
                "scenario_id": str(scenario.id),
                "scenario_name": scenario.name,
                "num_sheep": len(scenario.sheep),
                "drone_count": len(scenario.drones),
            }
        ),
        200,
    )


@app.route("/policy-presets", methods=["GET"])
def get_policy_presets():
    """Get all available policy presets."""
    presets_dict = {key: config.to_dict() for key, config in POLICY_PRESETS.items()}
    return jsonify(presets_dict), 200


@app.route("/scenario-types", methods=["GET"])
def get_scenario_types():
    """Get all available scenario type definitions."""
    types_list = [st.to_dict() for st in SCENARIO_TYPES.values()]
    return jsonify(types_list), 200


@app.route("/scenario-types/<key>", methods=["GET"])
def get_scenario_type_endpoint(key: str):
    """Get a specific scenario type definition by key."""
    scenario_type = get_scenario_type(key)
    if not scenario_type:
        return (
            jsonify(
                {
                    "error": {
                        "type": "NotFound",
                        "message": f"scenario type '{key}' not found",
                    }
                }
            ),
            404,
        )

    return jsonify(scenario_type.to_dict()), 200


@app.route("/scenario-types/<key>/instantiate", methods=["POST"])
def instantiate_scenario_type(key: str):
    """
    Create a new scenario from a scenario type template.
    Returns the created scenario.
    """
    scenario_type = get_scenario_type(key)
    if not scenario_type:
        return (
            jsonify(
                {
                    "error": {
                        "type": "NotFound",
                        "message": f"scenario type '{key}' not found",
                    }
                }
            ),
            404,
        )

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}

    # Get optional overrides
    name = (
        body.get("name")
        or f"{scenario_type.name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    num_agents = body.get("num_agents")
    num_controllers = body.get("num_controllers")
    seed = body.get("seed")
    overrides = body.get("overrides") or {}

    # Generate initial layout
    bounds = (0.0, 250.0, 0.0, 250.0)
    layout = generate_initial_layout(
        scenario_type,
        num_agents=num_agents,
        num_controllers=num_controllers,
        bounds=bounds,
        seed=seed,
    )

    # Merge world_config and policy_config from type with any overrides
    world_config = (
        scenario_type.default_world_config.copy()
        if scenario_type.default_world_config
        else {}
    )
    if overrides.get("world_config"):
        world_config.update(overrides["world_config"])

    policy_config = (
        scenario_type.default_policy_config.copy()
        if scenario_type.default_policy_config
        else {}
    )
    if overrides.get("policy_config"):
        policy_config.update(overrides["policy_config"])

    appearance = (
        {"themeKey": scenario_type.default_theme_key}
        if scenario_type.default_theme_key
        else None
    )
    if overrides.get("appearance"):
        if appearance:
            appearance.update(overrides["appearance"])
        else:
            appearance = overrides["appearance"]

    scenario = Scenario(
        id=uuid4(),
        name=name,
        description=overrides.get("description") or scenario_type.description,
        tags=scenario_type.tags.copy() if scenario_type.tags else [],
        visibility="public",
        seed=seed,
        sheep=layout["sheep"],
        drones=layout["drones"],
        targets=layout["targets"],
        obstacles=overrides.get("obstacles") or layout.get("obstacles") or [],
        goals=overrides.get("goals") or [],
        boundary=world_config.get("boundary", "none"),
        bounds=bounds,
        world_config=world_config if world_config else None,
        policy_config=policy_config if policy_config else None,
        target_sequence=None,
        scenario_type=key,
        appearance=appearance,
    )

    REPO.create(scenario)

    return jsonify(asdict(scenario)), 201, {"Location": f"/scenarios/{scenario.id}"}


# -----------------------------------------------------------------------------
# Metrics & Evaluation API
# -----------------------------------------------------------------------------


@app.route("/metrics/current", methods=["GET"])
def get_current_metrics():
    """Get metrics for the currently running simulation."""
    collector = get_collector()
    current = collector.get_current_run()

    if current is None:
        return (
            jsonify(
                {"active": False, "run_id": None, "message": "No active metrics run"}
            ),
            200,
        )

    return (
        jsonify(
            {
                "active": True,
                "run_id": current.run_id,
                "num_steps": len(current.steps),
                "started_at": current.started_at,
                "latest_step": current.steps[-1].to_dict() if current.steps else None,
            }
        ),
        200,
    )


@app.route("/metrics/runs", methods=["GET"])
def list_metrics_runs():
    """List all completed metrics runs."""
    collector = get_collector()
    runs = [
        {
            "run_id": run.run_id,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "num_steps": len(run.steps),
            "summary": run.summary,
        }
        for run in collector.completed_runs.values()
    ]

    return jsonify({"runs": runs, "count": len(runs)}), 200


@app.route("/metrics/runs/<run_id>", methods=["GET"])
def get_metrics_run(run_id: str):
    """Get detailed metrics for a specific run."""
    collector = get_collector()
    run = collector.get_run(run_id)

    if run is None:
        return (
            jsonify(
                {"error": {"type": "NotFound", "message": f"run '{run_id}' not found"}}
            ),
            404,
        )

    return jsonify(run.to_dict()), 200


@app.route("/metrics/start", methods=["POST"])
def start_metrics_collection():
    """Manually start metrics collection for the current simulation."""
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}

    run_id = body.get("run_id") or str(uuid4())[:8]
    run = start_metrics_run(run_id)

    return (
        jsonify(
            {
                "started": True,
                "run_id": run.run_id,
                "started_at": run.started_at,
            }
        ),
        200,
    )


@app.route("/metrics/stop", methods=["POST"])
def stop_metrics_collection():
    """Stop the current metrics collection and compute summary."""
    run = end_metrics_run()

    if run is None:
        return (
            jsonify({"stopped": False, "message": "No active metrics run to stop"}),
            200,
        )

    return (
        jsonify(
            {
                "stopped": True,
                "run_id": run.run_id,
                "summary": run.summary,
            }
        ),
        200,
    )


def run_flask():
    app.run(debug=True, use_reloader=False, port=5001, host="127.0.0.1")


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nCaught Ctrl+C, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start Flask in a background thread.
    threading.Thread(target=run_flask, daemon=True).start()

    # Throttle remaining_time DB writes (once per second)
    last_rem_sync_ts = 0.0

    while True:
        # Get current jobs from cache
        jobs = jobs_cache.list


        time.sleep(FRAME_TIME)
        # Update job statuses and remaining times
        jobs_to_sync = set()  # Use set to avoid duplicate syncs
        with world_lock:
            # Promote any scheduled jobs that should now start
            now = datetime.now().timestamp()
            for j in jobs:
                if (
                    j.status == "scheduled"
                    and j.start_at is not None
                    and j.start_at <= now
                ):
                    # Check if there's currently an active job
                    has_active_job = any(
                        other.is_active and other.status == "running"
                        for other in jobs
                        if other.id != j.id
                    )

                    if not has_active_job:
                        # No active job - promote to running and activate immediately
                        j.status = "running"
                        j.is_active = True
                        jobs_to_sync.add(j.id)
                    else:
                        # There's an active job - add this scheduled job to the queue as pending
                        j.status = "pending"
                        jobs_to_sync.add(j.id)

            # Check goal satisfaction and update remaining_time
            for job in jobs:
                if job.target is None:
                    job.remaining_time = None
                elif job.status == "completed" or job.status == "cancelled":
                    # Don't update completed/cancelled jobs
                    pass
                elif job.status == "running" and job.is_active:
                    # Only check goal for running+active jobs
                    if herding.policy.is_goal_satisfied(
                        backend_adapter.get_state(), job.target
                    ):
                        job.remaining_time = 0
                        # Double-check status before marking completed (race protection)
                        if job.status == "running" and job.is_active:
                            job.status = "completed"
                            job.is_active = False
                            job.completed_at = datetime.now(timezone.utc).timestamp()
                            jobs_to_sync.add(job.id)

                            # Activate the next pending job in the queue
                            # Order: scheduled jobs by start_at, then regular pending jobs by created_at
                            pending_jobs = [
                                j
                                for j in jobs
                                if j.status == "pending" and j.target is not None
                            ]
                            if pending_jobs:
                                # Sort: scheduled jobs (with start_at) first by start_at,
                                # then regular pending jobs by created_at
                                pending_jobs.sort(
                                    key=lambda j: (
                                        (
                                            j.start_at
                                            if j.start_at is not None
                                            else float("inf")
                                        ),
                                        j.created_at,
                                    )
                                )
                                next_job = pending_jobs[0]
                                next_job.status = "running"
                                next_job.is_active = True
                                jobs_to_sync.add(next_job.id)
                    else:
                        job.remaining_time = None
                else:
                    job.remaining_time = None

        # Persist status changes to database (outside lock to avoid blocking)
        if jobs_to_sync:
            for job_id in jobs_to_sync:
                job_obj = jobs_cache.get(job_id)  # O(1) lookup instead of linear search
                if job_obj:
                    try:
                        jobs_api.sync_job_status_to_db(job_obj)
                    except Exception as e:
                        # Don't crash - DB sync failure shouldn't stop simulation
                        print(f"Warning: Failed to sync job {job_id} to DB: {e}")

        # Throttled remaining_time sync (once per second for running jobs)
        current_time = time.time()
        if current_time - last_rem_sync_ts >= 1.0:
            for job in jobs:
                if job.status == "running" and job.remaining_time is not None:
                    try:
                        jobs_api.sync_job_status_to_db(job)
                    except Exception as e:
                        print(
                            f"Warning: Failed to sync remaining_time for job {job.id}: {e}"
                        )
            last_rem_sync_ts = current_time

        # We receive the new state of the world from the backend adapter, and we compute what we should do based on the planner.
        # We send that back to the backend adapter.
        with world_lock:
            # Sync target from active job to world, and auto-unpause if there's an active job with a target
            active_job = None
            for job in jobs:
                if job.is_active and job.status == "running":
                    active_job = job
                    break

            if active_job and active_job.target is not None:
                # Sync job target to world target
                # Extract center coordinates from Circle/Polygon for the World (which expects numpy array)
                if isinstance(active_job.target, state.Circle):
                    backend_adapter.target = active_job.target.center
                elif isinstance(active_job.target, state.Polygon):
                    # For polygon, use centroid as target
                    backend_adapter.target = active_job.target.points.mean(axis=0)
                else:
                    raise TypeError(
                        f"Job target must be Circle or Polygon, got {type(active_job.target)}"
                    )

                # Sync drone count from job to world
                if active_job.drone_count != backend_adapter.num_controllers:
                    backend_adapter.set_drone_count(active_job.drone_count)

                # Auto-unpause if paused
                if backend_adapter.paused:
                    backend_adapter.paused = False
            elif active_job and active_job.target is None:
                # Active job but no target yet - keep running for visualization (sheep will graze)
                # Also sync drone count
                if active_job.drone_count != backend_adapter.num_controllers:
                    backend_adapter.set_drone_count(active_job.drone_count)
                if backend_adapter.paused:
                    backend_adapter.paused = False
                backend_adapter.target = None
            else:
                # No active job - allow simulation to run for live monitoring
                if backend_adapter.paused:
                    backend_adapter.paused = False

                # If we have a pending job with a target, use it for the world target
                # This ensures the policy (and visualization) knows where to go even if the job isn't "started"
                pending_target_job = None
                for job in jobs:
                    if job.status == "pending" and job.target is not None:
                        pending_target_job = job
                        break

                if pending_target_job:
                    # Sync pending job target to world target
                    if isinstance(pending_target_job.target, state.Circle):
                        backend_adapter.target = pending_target_job.target.center
                    elif isinstance(pending_target_job.target, state.Polygon):
                        backend_adapter.target = pending_target_job.target.points.mean(
                            axis=0
                        )
                else:
                    # Only clear target if we really have no target source
                    backend_adapter.target = None

            for job in list(jobs):  # Copy to avoid modifying list while iterating
                if job.completed_at is not None:
                    jobs_cache.remove(job.id)

            # Run multiple simulation steps per frame to speed up simulation
            # while maintaining smooth 60Hz updates
            for _ in range(STEPS_PER_FRAME):
                plan = policy.plan(
                    backend_adapter.get_state(), jobs, backend_adapter.dt
                )
                backend_adapter.step(plan)

            # Record metrics if collection is active
            # from server.metrics import get_collector

            collector = get_collector()
            if collector.get_current_run() is not None:
                world_state = backend_adapter.get_state()

                # Determine target for metrics
                target = None
                if active_job and active_job.target:
                    target = active_job.target
                else:
                    # Try to find a pending job with a target
                    for job in jobs:
                        if job.status == "pending" and job.target:
                            target = job.target
                            break

                    # Fallback to world target if still no job target found
                    if target is None and backend_adapter.target is not None:
                        from planning import state

                        radius = policy.fN if hasattr(policy, "fN") else 10.0
                        target = state.Circle(
                            center=backend_adapter.target, radius=radius
                        )

                t = backend_adapter.t if hasattr(backend_adapter, "t") else 0.0
                fN = policy.fN if hasattr(policy, "fN") else 50.0

                collector.record_step(world_state, target, t, fN)
