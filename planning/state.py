"""
Simulation State Definitions

This module defines the core data structures representing the state of the simulation,
including the flock, drones, obstacles, and active jobs. It serves as the data contract
between the simulation engine, the planner, and the API.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional, Union

import numpy as np

# -----------------------------------------------------------------------------
# Constants & Types
# -----------------------------------------------------------------------------

JobStatus = Literal["pending", "scheduled", "running", "completed", "cancelled"]
MaintainUntil = Union[
    Literal["target_is_reached"], float
]  # "target_is_reached" or UNIX timestamp


# -----------------------------------------------------------------------------
# Geometry Primitives
# -----------------------------------------------------------------------------


@dataclass
class Circle:
    """Represents a circular target or zone."""

    center: np.ndarray
    radius: Optional[float]

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "radius": self.radius,
            "type": "circle",
        }


@dataclass
class Polygon:
    """Represents a polygonal target or zone (e.g., a pen)."""

    points: np.ndarray

    def to_dict(self) -> dict:
        return {
            "points": self.points.tolist(),
            "type": "polygon",
        }


Target = Union[Circle, Polygon]


# -----------------------------------------------------------------------------
# Job State
# -----------------------------------------------------------------------------


@dataclass
class Job:
    """
    Represents a high-level task for the herding system (e.g., "move flock to X").
    Tracks lifecycle, scheduling, and progress.
    """

    # Core configuration
    target: Optional[Target]
    drone_count: int
    scenario_id: Optional[str]  # UUID pointing to a scenario

    # Status & Progress
    status: JobStatus  # "pending", "scheduled", "running", "completed", "cancelled"
    is_active: bool  # If the user pauses a job, this becomes false.
    remaining_time: Optional[float]  # Estimate of seconds remaining

    # Scheduling
    start_at: Optional[float]  # UNIX timestamp; None = immediate
    completed_at: Optional[float]  # UNIX timestamp; None = not completed

    # Termination condition:
    # - "target_is_reached": maintain until target condition is satisfied
    # - float: maintain until this UNIX timestamp
    maintain_until: MaintainUntil

    # Metadata
    created_at: float  # UNIX timestamp
    updated_at: float  # UNIX timestamp
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def to_dict(self) -> dict:
        """Convert job state to a dictionary for API response."""

        def ts_to_iso(ts: Optional[float]) -> Optional[str]:
            if ts is None:
                return None
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        def maintain_until_to_dict(mu: MaintainUntil) -> str:
            if mu == "target_is_reached":
                return "target_is_reached"
            else:
                # mu is a float timestamp here
                result = ts_to_iso(mu)
                return result if result is not None else ""

        drone_count_value = getattr(self, "drone_count", 1)

        return {
            "id": str(self.id),
            "target": self.target.to_dict() if self.target is not None else None,
            "remaining_time": self.remaining_time,
            "is_active": self.is_active,
            "drone_count": drone_count_value,
            "status": self.status,
            "start_at": ts_to_iso(self.start_at),
            "completed_at": ts_to_iso(self.completed_at),
            "scenario_id": self.scenario_id,
            "maintain_until": maintain_until_to_dict(self.maintain_until),
            "created_at": ts_to_iso(self.created_at),
            "updated_at": ts_to_iso(self.updated_at),
        }


# -----------------------------------------------------------------------------
# World State
# -----------------------------------------------------------------------------


@dataclass
class State:
    """
    Snapshot of the entire simulation state at a specific time step.
    Includes agent positions, drone positions, obstacles, and active jobs.
    """

    # n-by-2 array of sheep positions
    flock: np.ndarray

    # n-by-2 array of drone positions
    drones: np.ndarray

    # List of polygon obstacles, each is an (m,2) array of vertices
    polygons: List[np.ndarray]

    # Active jobs
    jobs: List[Job]

    def to_dict(self) -> dict:
        """Convert world state to a dictionary for API response."""
        return {
            "flock": self.flock.tolist(),
            "drones": self.drones.tolist(),
            "jobs": [j.to_dict() for j in self.jobs],
            "polygons": [poly.tolist() for poly in self.polygons],
        }
