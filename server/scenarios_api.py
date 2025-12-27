"""
Scenarios API

This module handles the creation, listing, retrieval, and persistence of scenarios.
It supports both preset scenarios (in-memory) and custom scenarios (persisted to disk).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from flask import Blueprint, jsonify, request

from simulation.scenarios import (
    spawn_circle,
    spawn_clusters,
    spawn_corners,
    spawn_line,
    spawn_uniform,
)

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "tmp" / "scenarios.pkl"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

Vec2 = Tuple[float, float]
Visibility = Literal["private", "public", "preset"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class Scenario:
    """
    Represents a simulation scenario configuration.
    Contains all necessary data to initialize a world state.
    """

    # Identity & Metadata
    id: UUID
    name: str
    description: Optional[str]
    tags: List[str]
    visibility: Visibility
    seed: Optional[int]

    # Resolved Entities (Positions)
    sheep: List[Vec2]
    drones: List[Vec2]
    targets: List[Vec2]

    # Passthrough fields (not strictly enforced here but used by simulation)
    obstacles: List[dict] = field(default_factory=list)
    goals: List[dict] = field(default_factory=list)

    # World Parameters
    boundary: Literal["none", "wrap", "reflect"] = "none"
    bounds: Tuple[float, float, float, float] = (0.0, 250.0, 0.0, 250.0)

    # Configuration fields for world/physics and policy behavior
    world_config: Optional[dict] = None  # World/physics parameter overrides
    policy_config: Optional[dict] = (
        None  # Policy configuration (preset key or custom dict)
    )
    target_sequence: Optional[List[Vec2]] = None  # Multi-waypoint goals

    # Scenario type (behavior pack) reference
    scenario_type: Optional[str] = None  # Key from SCENARIO_TYPES registry

    # Appearance/theme configuration
    appearance: Optional[dict] = None  # Contains themeKey and other visual settings

    # Domain tag for future extensibility
    domain: Optional[str] = None  # "herding", "evacuation", etc.

    # Environment (farm, city, ocean)
    environment: str = "farm"

    # Versioning
    version: int = 1
    schema_version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# -----------------------------------------------------------------------------
# Repository
# -----------------------------------------------------------------------------


class ScenarioRepo:
    """
    Repository for persisting and retrieving scenarios from PKL.

    Every mutating method loads from disk, modifies, and saves back.
    Preset scenarios are kept in memory and merged with custom scenarios from disk.
    """

    def __init__(self, cap: int = 500):
        self._lock = threading.RLock()
        self._presets: Dict[UUID, Scenario] = {}  # Preset scenarios (in-memory only)
        self._idem: Dict[str, UUID] = {}
        self._cap = cap

    def _load_custom_scenarios(self) -> List[Scenario]:
        """Load custom scenarios from the database."""
        if not DB_PATH.exists():
            return []
        try:
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, Exception) as e:
            print(f"Warning: Failed to load scenarios from {DB_PATH}: {e}")
            return []

    def _save_custom_scenarios(self, scenarios: List[Scenario]):
        """Save custom scenarios to the database."""
        # Only save non-preset scenarios
        custom_scenarios = [s for s in scenarios if s.visibility != "preset"]
        # Evict oldest if over capacity
        while len(custom_scenarios) > self._cap:
            custom_scenarios.pop(0)
        try:
            with open(DB_PATH, "wb") as f:
                pickle.dump(custom_scenarios, f)
        except Exception as e:
            print(f"Warning: Failed to save scenarios to {DB_PATH}: {e}")

    def add_preset(self, s: Scenario) -> Scenario:
        """Add a preset scenario (in-memory only, not persisted)."""
        with self._lock:
            self._presets[s.id] = s
            return s

    def create(self, s: Scenario) -> Scenario:
        """Create a new scenario in the database."""
        with self._lock:
            if s.visibility == "preset":
                # Presets are stored in memory only
                self._presets[s.id] = s
                return s

            # Load, add, and save custom scenarios
            scenarios = self._load_custom_scenarios()
            # Check if already exists (by ID)
            existing_ids = {sc.id for sc in scenarios}
            if s.id not in existing_ids:
                scenarios.append(s)
            self._save_custom_scenarios(scenarios)
            return s

    def get(self, sid: UUID) -> Optional[Scenario]:
        """Retrieve a scenario by ID."""
        with self._lock:
            # Check presets first
            if sid in self._presets:
                return self._presets[sid]
            # Check custom scenarios
            for s in self._load_custom_scenarios():
                if s.id == sid:
                    return s
            return None

    def list(
        self,
        *,
        q: Optional[str],
        tag: Optional[str],
        visibility: Optional[Visibility],
        created_after: Optional[str],
        created_before: Optional[str],
        sort: str,
        limit: int,
        offset: int,
    ) -> Tuple[List[Scenario], int]:
        """List scenarios with filtering and pagination."""
        with self._lock:
            # Combine presets and custom scenarios
            items = list(self._presets.values()) + self._load_custom_scenarios()

        def match(s: Scenario) -> bool:
            if q:
                qq = q.lower()
                if (
                    qq not in (s.name or "").lower()
                    and qq not in (s.description or "").lower()
                    and qq not in " ".join(s.tags).lower()
                ):
                    return False
            if tag and tag not in s.tags:
                return False
            if visibility and s.visibility != visibility:
                return False
            if created_after and s.created_at <= created_after:
                return False
            if created_before and s.created_at >= created_before:
                return False
            return True

        items = [s for s in items if match(s)]

        reverse = sort.startswith("-")
        key = sort.lstrip("-")
        if key not in {"created_at", "updated_at", "name"}:
            key = "created_at"
        items.sort(key=lambda s: getattr(s, key), reverse=reverse)

        total = len(items)
        limit = max(1, min(100, int(limit)))
        offset = max(0, int(offset))
        return items[offset : offset + limit], total

    def delete(self, sid: UUID) -> bool:
        """Delete a custom scenario by ID. Presets cannot be deleted."""
        with self._lock:
            if sid in self._presets:
                return False  # Cannot delete presets

            scenarios = self._load_custom_scenarios()
            original_len = len(scenarios)
            scenarios = [s for s in scenarios if s.id != sid]
            if len(scenarios) < original_len:
                self._save_custom_scenarios(scenarios)
                return True
            return False

    def save_idem(self, key: str, sid: UUID):
        with self._lock:
            self._idem[key] = sid

    def get_idem(self, key: str) -> Optional[UUID]:
        with self._lock:
            return self._idem.get(key)


# Global Repository Instance
REPO = ScenarioRepo()


# -----------------------------------------------------------------------------
# Preset Seeding
# -----------------------------------------------------------------------------


def _seed_presets():
    """Add default preset scenarios on startup."""
    presets = [
        # City evacuation scenario with urban intersection background
        Scenario(
            id=uuid4(),
            name="City Evacuation - 40 People",
            description=(
                "Urban evacuation at a city intersection: guide 40 people to the designated safe zone "
                "using coordinated drone robots"
            ),
            tags=["preset", "evacuation", "urban", "city", "intersection"],
            visibility="preset",
            seed=123,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_uniform(40, (50, 200, 200, 240), seed=123).tolist()
            ],
            drones=[
                (30.0, 125.0),
                (220.0, 125.0),
                (125.0, 30.0),
            ],  # Three drones surrounding
            targets=[(125.0, 30.0)],  # Safe zone at the bottom
            # Accident in the intersection (central obstacle) + side debris
            obstacles=[
                {
                    "polygon": [
                        [115.0, 115.0],
                        [135.0, 115.0],
                        [135.0, 135.0],
                        [115.0, 135.0],
                    ]
                },  # Center (smaller)
                {
                    "polygon": [
                        [30.0, 110.0],
                        [50.0, 110.0],
                        [50.0, 140.0],
                        [30.0, 140.0],
                    ]
                },  # Left (shifted left)
                {
                    "polygon": [
                        [200.0, 110.0],
                        [220.0, 110.0],
                        [220.0, 140.0],
                        [200.0, 140.0],
                    ]
                },  # Right (shifted right)
            ],
            boundary="reflect",
            bounds=(-250.0, 400.0, 0.0, 500.0),
            scenario_type="evacuation_prototype",
            policy_config={"key": "evacuation-prototype"},
            appearance={"themeKey": "evacuation-prototype", "iconSet": "evacuation"},
            environment="city",
        ),
        Scenario(
            id=uuid4(),
            name="Default - 50 Sheep Uniform",
            description="The default simulation: 50 sheep uniformly distributed (0-200 range), 1 drone at origin",
            tags=["preset", "default", "uniform", "small"],
            visibility="preset",
            seed=42,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_uniform(50, (0, 200, 0, 200), seed=42).tolist()
            ],
            drones=[(0.0, 0.0)],
            targets=[],  # No default target
            boundary="none",
            bounds=(0.0, 250.0, 0.0, 250.0),
            environment="farm",
        ),
        Scenario(
            id=uuid4(),
            name="Large Flock - 200 in 2 Clusters",
            description="200 sheep clustered in 2 groups, drone at center",
            tags=["preset", "large", "clusters", "challenge"],
            visibility="preset",
            seed=7,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_clusters(
                    200, 2, (0, 250, 0, 250), spread=3.5, seed=7
                ).tolist()
            ],
            drones=[(125.0, 125.0)],
            targets=[],  # No default target
            boundary="none",
            bounds=(0.0, 250.0, 0.0, 250.0),
            environment="farm",
        ),
        Scenario(
            id=uuid4(),
            name="Corner Challenge - 80 Sheep",
            description="80 sheep scattered in 4 corners, requires strategic herding",
            tags=["preset", "corners", "medium", "challenge"],
            visibility="preset",
            seed=3,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_corners(
                    80, (0, 250, 0, 250), jitter=2.0, seed=3
                ).tolist()
            ],
            drones=[(125.0, 125.0)],
            targets=[],  # No default target
            boundary="none",
            bounds=(0.0, 250.0, 0.0, 250.0),
            environment="farm",
        ),
        Scenario(
            id=uuid4(),
            name="Line Formation - 60 Sheep",
            description="60 sheep in a horizontal line, testing linear herding",
            tags=["preset", "line", "medium"],
            visibility="preset",
            seed=5,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_line(60, (0, 250, 0, 250), seed=5).tolist()
            ],
            drones=[(0.0, 0.0)],
            targets=[],  # No default target
            boundary="none",
            bounds=(0.0, 250.0, 0.0, 250.0),
            environment="farm",
        ),
        # Oil Spill Cleanup Scenario
        Scenario(
            id=uuid4(),
            name="Oil Spill Cleanup - 100 Droplets",
            description="Contain and clean up oil spills in the ocean using boom-equipped boats.",
            tags=["preset", "oil", "ocean", "cleanup"],
            visibility="preset",
            seed=99,
            sheep=[
                (float(x), float(y))
                for x, y in spawn_clusters(
                    100, 5, (0, 250, 0, 250), spread=20.0, seed=99
                ).tolist()
            ],
            drones=[(50.0, 50.0), (200.0, 50.0), (125.0, 200.0)],
            targets=[],
            boundary="none",
            bounds=(0.0, 300.0, 0.0, 300.0),
            scenario_type="oil_spill_cleanup",
            policy_config={"key": "default"},
            appearance={"themeKey": "oil-spill", "iconSet": "oil"},
            environment="ocean",
        ),
    ]

    for preset in presets:
        REPO.add_preset(preset)
        print(f"✓ Created preset: {preset.name} ({len(preset.sheep)} sheep)")


# Seed presets on module load
_seed_presets()

# Load persisted custom scenarios count
_custom_count = len(REPO._load_custom_scenarios())
print(
    f"Scenarios API ready with {len(REPO._presets)} presets + {_custom_count} custom scenarios"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _finite_pair(p) -> Vec2:
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        raise ValueError("coordinate must be [x, y]")
    x, y = float(p[0]), float(p[1])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("coordinate must be finite")
    return (x, y)


def _normalize_points(pts) -> List[Vec2]:
    return [_finite_pair(p) for p in (pts or [])]


def _hash_body(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _round_pts(pts: List[Vec2], nd=9) -> List[Vec2]:
    return [(round(float(x), nd), round(float(y), nd)) for x, y in pts]


# -----------------------------------------------------------------------------
# Spawn Dispatcher
# -----------------------------------------------------------------------------


def _spawn_entities(body: dict) -> Tuple[List[Vec2], List[Vec2], List[Vec2]]:
    """
    Returns (sheep, drones, targets) using simulation/scenarios.py helpers.
    Accepts either 'entities' directly or a 'spawn' block:
      spawn: {
        kind: "uniform"|"clusters"|"corners"|"line"|"circle",
        params inline (see below),
        drones: [{ position:[x,y] } or { around:[x,y], radius, count }],
        targets: [[x,y], ...]
      }
    """
    entities = body.get("entities")
    spawn = body.get("spawn")

    if entities and spawn:
        raise ValueError("Provide either 'entities' or 'spawn', not both.")

    # Direct entities
    if entities:
        sheep = _normalize_points(entities.get("sheep"))
        drones = _normalize_points(entities.get("drones"))
        targets = _normalize_points(entities.get("targets"))
        return _round_pts(sheep), _round_pts(drones), _round_pts(targets)

    if not spawn:
        raise ValueError("Provide 'entities' or 'spawn'.")

    kind = (spawn.get("kind") or "uniform").lower()
    seed = spawn.get("seed")
    bounds = tuple(spawn.get("bounds", (0.0, 250.0, 0.0, 250.0)))
    if len(bounds) != 4:
        raise ValueError("'spawn.bounds' must be [xmin, xmax, ymin, ymax]")
    xmin, xmax, ymin, ymax = map(float, bounds)

    N = int(spawn.get("num_sheep", 0))
    N = max(0, N)

    # Sheep via your helpers
    if kind == "uniform":
        sheep_xy = spawn_uniform(N, (xmin, xmax, ymin, ymax), seed=seed)
    elif kind == "clusters":
        k = int(spawn.get("k", 2))
        spread = float(spawn.get("spread", 3.5))
        sheep_xy = spawn_clusters(
            N, k, (xmin, xmax, ymin, ymax), spread=spread, seed=seed
        )
    elif kind == "corners":
        jitter = float(spawn.get("jitter", 2.0))
        sheep_xy = spawn_corners(N, (xmin, xmax, ymin, ymax), jitter=jitter, seed=seed)
    elif kind == "line":
        y_fixed = spawn.get("y")
        sheep_xy = spawn_line(N, (xmin, xmax, ymin, ymax), seed=seed, y=y_fixed)
    elif kind == "circle":
        center = _finite_pair(
            spawn.get("center", [(xmin + xmax) / 2, (ymin + ymax) / 2])
        )
        radius = float(spawn.get("radius", 5.0))
        sheep_xy = spawn_circle(N, center=center, radius=radius, seed=seed)
    else:
        raise ValueError(f"unknown spawn.kind '{kind}'")

    sheep = [(float(x), float(y)) for x, y in sheep_xy.tolist()]

    # Drones
    drones_in = spawn.get("drones") or []
    drones_list: List[Vec2] = []
    for d in drones_in:
        if "position" in d:
            drones_list.append(_finite_pair(d["position"]))
        elif "around" in d and "radius" in d:
            around = _finite_pair(d["around"])
            count = int(d.get("count", 1))
            r = float(d["radius"])
            rng = np.random.default_rng(seed)
            th = rng.random(count) * 2 * np.pi
            xs = around[0] + r * np.cos(th)
            ys = around[1] + r * np.sin(th)
            drones_list.extend([(float(x), float(y)) for x, y in zip(xs, ys)])
        else:
            # fallback: center
            drones_list.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))

    targets = _normalize_points(spawn.get("targets") or [])

    return _round_pts(sheep), _round_pts(drones_list), _round_pts(targets)


# -----------------------------------------------------------------------------
# Blueprint
# -----------------------------------------------------------------------------

scenarios_bp = Blueprint("scenarios", __name__, url_prefix="/scenarios")


@scenarios_bp.route("", methods=["POST"])
def create_scenario() -> Any:
    try:
        body = request.get_json(force=True, silent=False) or {}
    except Exception:
        return (
            jsonify({"error": {"type": "BadRequest", "message": "Invalid JSON"}}),
            400,
        )

    if not isinstance(body, dict):
        return (
            jsonify(
                {
                    "error": {
                        "type": "BadRequest",
                        "message": "JSON body must be an object",
                    }
                }
            ),
            400,
        )

    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        sid_existing = REPO.get_idem(idem_key)
        if sid_existing:
            existing = REPO.get(sid_existing)
            if existing:
                return (
                    jsonify(asdict(existing)),
                    201,
                    {"Location": f"/scenarios/{existing.id}"},
                )

    name = str(body.get("name", "")).strip()
    if not name:
        return (
            jsonify({"error": {"type": "Validation", "message": "'name' is required"}}),
            422,
        )

    description = body.get("description")
    tags = [str(t).strip().lower() for t in (body.get("tags") or []) if str(t).strip()]
    visibility = body.get("visibility", "private")
    if visibility not in ("private", "public", "preset"):
        return (
            jsonify(
                {"error": {"type": "Validation", "message": "invalid 'visibility'"}}
            ),
            422,
        )

    w = body.get("world") or {}
    boundary = w.get("boundary", "none")
    if boundary not in ("none", "wrap", "reflect"):
        return (
            jsonify(
                {"error": {"type": "Validation", "message": "invalid world.boundary"}}
            ),
            422,
        )
    bounds = tuple(w.get("bounds", (0.0, 250.0, 0.0, 250.0)))
    if len(bounds) != 4:
        return (
            jsonify(
                {
                    "error": {
                        "type": "Validation",
                        "message": "world.bounds must be [xmin,xmax,ymin,ymax]",
                    }
                }
            ),
            422,
        )
    seed = w.get("seed")

    try:
        sheep, drones, targets = _spawn_entities(body)
    except ValueError as ex:
        return jsonify({"error": {"type": "Validation", "message": str(ex)}}), 422

    s = Scenario(
        id=uuid4(),
        name=name,
        description=description,
        tags=list(dict.fromkeys(tags)),
        visibility=visibility,  # type: ignore
        seed=seed,
        sheep=sheep,
        drones=drones if drones else [(0.0, 0.0)],  # World expects >=1 shepherd
        targets=targets if targets else [],
        obstacles=body.get("obstacles") or [],
        goals=body.get("goals") or [],
        boundary=boundary,
        bounds=(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
        # Configuration fields
        world_config=body.get("world_config"),
        policy_config=body.get("policy_config"),
        target_sequence=body.get("target_sequence"),
        scenario_type=body.get("scenario_type"),
        appearance=body.get("appearance"),
    )

    REPO.create(s)
    if idem_key:
        REPO.save_idem(idem_key, s.id)

    return jsonify(asdict(s)), 201, {"Location": f"/scenarios/{s.id}"}


@scenarios_bp.route("", methods=["GET"])
def list_scenarios() -> Any:
    q = request.args.get("q")
    tag = request.args.get("tag")
    visibility = request.args.get("visibility")
    if visibility not in (None, "private", "public", "preset"):
        return (
            jsonify(
                {"error": {"type": "Validation", "message": "invalid 'visibility'"}}
            ),
            422,
        )

    created_after = request.args.get("created_after")
    created_before = request.args.get("created_before")
    sort = request.args.get("sort", "-created_at")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))

    items, total = REPO.list(
        q=q,
        tag=tag,
        visibility=visibility,  # type: ignore
        created_after=created_after,
        created_before=created_before,
        sort=sort,
        limit=limit,
        offset=offset,
    )

    return (
        jsonify(
            {
                "items": [asdict(s) for s in items],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        ),
        200,
    )


@scenarios_bp.route("/<uuid:sid>", methods=["GET"])
def get_scenario(sid: UUID) -> Any:
    s = REPO.get(sid)
    if not s:
        return (
            jsonify({"error": {"type": "NotFound", "message": "scenario not found"}}),
            404,
        )
    return jsonify(asdict(s)), 200
