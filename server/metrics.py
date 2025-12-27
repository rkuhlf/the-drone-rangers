"""
Metrics & Evaluation System

This module provides lightweight metrics collection for simulation runs,
enabling performance analysis and evaluation without impacting simulation speed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


import numpy as np

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_STEPS_IN_MEMORY = 1000
STEPS_TO_KEEP_HEAD = 100
MAX_COMPLETED_RUNS = 50
EPSILON = 1e-6


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class StepMetrics:
    """Metrics captured at each simulation step."""

    t: float  # Simulation time
    fraction_in_goal: float  # Fraction of agents within goal region
    spread_radius: float  # Max distance from agent GCM (flock spread)
    min_obstacle_distance: float  # Min distance from any agent to obstacles
    cohesiveness: float  # Cohesiveness metric (fN / max_radius)
    gcm_to_goal_distance: float  # Distance from GCM to goal

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunMetrics:
    """
    Complete metrics for a simulation run (episode).

    Captures step-level data and summary statistics.
    """

    run_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    # Step-level metrics (kept in memory, can be truncated for large runs)
    steps: List[StepMetrics] = field(default_factory=list)

    # Summary statistics (computed at end of run)
    summary: Dict[str, Any] = field(default_factory=dict)

    # Maximum steps to keep in memory (older steps are discarded)
    max_steps_in_memory: int = MAX_STEPS_IN_MEMORY

    def add_step(self, step: StepMetrics):
        """Add a step's metrics, potentially discarding old steps."""
        self.steps.append(step)
        if len(self.steps) > self.max_steps_in_memory:
            # Keep first 100 and last (max-100) steps for analysis
            keep_first = STEPS_TO_KEEP_HEAD
            keep_last = self.max_steps_in_memory - keep_first
            self.steps = self.steps[:keep_first] + self.steps[-keep_last:]

    def compute_summary(self):
        """Compute summary statistics from step data."""
        if not self.steps:
            return

        self.ended_at = time.time()

        # Extract arrays for analysis
        fractions = [s.fraction_in_goal for s in self.steps]
        spreads = [s.spread_radius for s in self.steps]
        cohesiveness = [s.cohesiveness for s in self.steps]
        gcm_distances = [s.gcm_to_goal_distance for s in self.steps]
        times = [s.t for s in self.steps]

        # Time to reach thresholds
        def time_to_threshold(values: List[float], threshold: float) -> Optional[float]:
            for i, v in enumerate(values):
                if v >= threshold:
                    return times[i]
            return None

        self.summary = {
            # Success metrics
            "time_to_reach_fraction_50": time_to_threshold(fractions, 0.5),
            "time_to_reach_fraction_90": time_to_threshold(fractions, 0.9),
            "time_to_reach_fraction_95": time_to_threshold(fractions, 0.95),
            "final_fraction_in_goal": fractions[-1] if fractions else 0,
            # Spread/cohesion metrics
            "max_spread_radius": max(spreads) if spreads else 0,
            "min_spread_radius": min(spreads) if spreads else 0,
            "avg_spread_radius": sum(spreads) / len(spreads) if spreads else 0,
            "avg_cohesiveness": (
                sum(cohesiveness) / len(cohesiveness) if cohesiveness else 0
            ),
            # Goal approach metrics
            "initial_gcm_to_goal": gcm_distances[0] if gcm_distances else 0,
            "final_gcm_to_goal": gcm_distances[-1] if gcm_distances else 0,
            # Run info
            "num_steps": len(self.steps),
            "total_simulation_time": times[-1] if times else 0,
            "wall_clock_duration": (
                (self.ended_at - self.started_at) if self.ended_at else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": self.summary,
            "steps": [
                s.to_dict() for s in self.steps[-STEPS_TO_KEEP_HEAD:]
            ],  # Last 100 steps only in API
            "num_steps_total": len(self.steps),
        }


# -----------------------------------------------------------------------------
# Metrics Collector
# -----------------------------------------------------------------------------


class MetricsCollector:
    """
    Collects metrics during simulation runs.

    Designed to be lightweight and not impact simulation performance.
    """

    def __init__(self):
        self.current_run: Optional[RunMetrics] = None
        self.completed_runs: Dict[str, RunMetrics] = {}
        self.max_completed_runs = MAX_COMPLETED_RUNS

    def start_run(self, run_id: str) -> RunMetrics:
        """Start a new metrics collection run."""
        # End any existing run
        if self.current_run is not None:
            self.end_run()

        self.current_run = RunMetrics(run_id=run_id)
        return self.current_run

    def end_run(self) -> Optional[RunMetrics]:
        """End the current run and compute summary statistics."""
        if self.current_run is None:
            return None

        self.current_run.compute_summary()

        # Store in completed runs
        self.completed_runs[self.current_run.run_id] = self.current_run

        # Evict old runs
        while len(self.completed_runs) > self.max_completed_runs:
            oldest_key = next(iter(self.completed_runs))
            del self.completed_runs[oldest_key]

        run = self.current_run
        self.current_run = None
        return run

    def record_step(
        self,
        world_state,
        target,
        t: float,
        fN: float,
    ):
        """
        Record metrics for a single simulation step.

        Args:
            world_state: The current world state (with flock, drones arrays)
            target: The target (Circle or Polygon from state module)
            t: Current simulation time
            fN: The flock radius parameter
        """
        if self.current_run is None:
            return

        flock = world_state.flock
        if flock.shape[0] == 0:
            return

        # Compute GCM
        gcm = np.mean(flock, axis=0)

        # Compute spread radius (max distance from GCM)
        distances_to_gcm = np.linalg.norm(flock - gcm, axis=1)
        spread_radius = float(np.max(distances_to_gcm))

        # Compute cohesiveness
        cohesiveness = float(fN / (spread_radius + EPSILON))

        # Compute fraction in goal
        fraction_in_goal = 0.0
        gcm_to_goal = 0.0

        if target is not None:
            from planning import state

            if isinstance(target, state.Circle):
                # Distance from each agent to goal center
                distances_to_goal = np.linalg.norm(
                    flock - np.array(target.center), axis=1
                )
                fraction_in_goal = float(
                    np.sum(distances_to_goal <= target.radius) / len(flock)
                )
                gcm_to_goal = float(np.linalg.norm(gcm - np.array(target.center)))
            elif isinstance(target, state.Polygon):
                # For polygons, use point-in-polygon check
                from planning.herding.utils import points_inside_polygon

                inside = points_inside_polygon(flock, target.points)
                fraction_in_goal = float(np.sum(inside) / len(flock))
                # Use centroid for distance
                poly_center = np.mean(target.points, axis=0)
                gcm_to_goal = float(np.linalg.norm(gcm - poly_center))

        # JSON doesn't support Infinity, so use -1.0 to indicate "no obstacles" or "unknown"
        min_obstacle_distance = -1.0

        step = StepMetrics(
            t=float(t),
            fraction_in_goal=fraction_in_goal,
            spread_radius=spread_radius,
            min_obstacle_distance=min_obstacle_distance,
            cohesiveness=cohesiveness,
            gcm_to_goal_distance=gcm_to_goal,
        )
        self.current_run.add_step(step)

    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Get a run by ID (current or completed)."""
        if self.current_run and self.current_run.run_id == run_id:
            return self.current_run
        return self.completed_runs.get(run_id)

    def get_current_run(self) -> Optional[RunMetrics]:
        """Get the currently active run."""
        return self.current_run


# -----------------------------------------------------------------------------
# Global Instance & Public API
# -----------------------------------------------------------------------------

_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def start_metrics_run(run_id: str) -> RunMetrics:
    """Start a new metrics collection run."""
    return get_collector().start_run(run_id)


def end_metrics_run() -> Optional[RunMetrics]:
    """End the current metrics run."""
    return get_collector().end_run()


def record_step_metrics(world_state, target, t: float, fN: float):
    """Record metrics for the current step."""
    get_collector().record_step(world_state, target, t, fN)
