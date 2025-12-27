"""
Plan Types

This module defines the types of plans that the planner can generate and send to the
simulation engine. Currently, this supports direct drone positioning or a no-op.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np

# -----------------------------------------------------------------------------
# Plan Definitions
# -----------------------------------------------------------------------------


@dataclass
class DronePositions:
    """
    A plan that specifies the exact target positions for all drones.
    """

    # n-by-2 array of target positions for all drones
    positions: np.ndarray

    # n-by-1 boolean array:
    # - 1 (True): Drone is low enough to apply repulsion
    # - 0 (False): Drone is high (flyover) and does not repel
    apply_repulsion: np.ndarray

    # --- DEBUGGING INFO ---
    # These fields are used for visualization and debugging purposes.
    target_sheep_indices: List[int]
    gcm: np.ndarray
    radius: float


@dataclass
class DoNothing:
    """A no-op plan where the simulation continues without new drone commands."""

    pass


# Union type for type hinting
Plan = Union[DronePositions, DoNothing]
