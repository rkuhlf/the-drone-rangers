"""
Scenario Types / Behavior Packs

This module provides reusable scenario type definitions that bundle:
- world_config defaults
- policy_config defaults
- appearance/theme defaults
- recommended entity counts

Scenario types serve as templates for quickly creating scenarios with
appropriate settings for different use cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_AGENT_COUNT = 50
DEFAULT_CONTROLLER_COUNT = 2
BOUNDS_PADDING = 5.0


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class ScenarioTypeDefinition:
    """
    Definition for a reusable scenario type/behavior pack.

    Provides defaults for world physics, policy behavior, and visual appearance
    that can be used as a starting point for new scenarios.
    """

    key: str
    name: str
    description: str

    # Configuration defaults
    default_world_config: Optional[Dict[str, Any]] = None
    default_policy_config: Optional[Dict[str, Any]] = None

    # Visual defaults
    default_theme_key: str = "default-herd"
    default_icon_set: str = (
        "herding"  # "herding" (sheep/drones) or "evacuation" (people/guides)
    )

    # Recommended entity counts
    recommended_agents: Optional[int] = None
    recommended_controllers: Optional[int] = None

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    # Environment (farm, city, ocean)
    environment: str = "farm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Scenario Registry
# -----------------------------------------------------------------------------

SCENARIO_TYPES: Dict[str, ScenarioTypeDefinition] = {
    "herding_standard": ScenarioTypeDefinition(
        key="herding_standard",
        name="Herding",
        description="Standard sheep herding scenario. Guide the flock to the target zone.",
        default_world_config={
            "boundary": "none",
        },
        default_policy_config={
            "key": "default",
        },
        default_theme_key="default-herd",
        default_icon_set="herding",
        recommended_agents=50,
        recommended_controllers=2,
        tags=["herding", "farm", "standard"],
        environment="farm",
    ),
    "evacuation_prototype": ScenarioTypeDefinition(
        key="evacuation_prototype",
        name="City Evacuation",
        description=(
            "Urban evacuation scenario at a city intersection. "
            "Guide people to safety zones using coordinated drone robots."
        ),
        default_world_config={
            "boundary": "reflect",
            "k_nn": 1,  # Minimal flocking - people don't herd like sheep
            "wa": 0.1,  # Very low attraction - people don't clump

            "w_align": 0.2,  # Low alignment - people move independently
            "graze_alpha": 0.0,  # No random grazing
            "vmax": 0.6,  # Realistic walking speed
            "umax": 1.0,  # Moderate drone speed
            "w_target": 8.0,  # Stronger attraction to goal
            "wr": 0.5,  # Reduced repulsion for smoother flow
            "ws": 10.0,  # Greatly reduced drone repulsion to prevent erratic movement
            "ra": 2.0,  # Smaller agent radius for tighter packing
            "sigma": 0.001,  # Reduced noise for less erratic movement
        },
        default_policy_config={
            "key": "evacuation-prototype",
        },
        default_theme_key="evacuation-prototype",
        default_icon_set="evacuation",
        recommended_agents=40,
        recommended_controllers=3,
        tags=["evacuation", "urban", "city", "research"],
        environment="city",
    ),
    "oil_spill_cleanup": ScenarioTypeDefinition(
        key="oil_spill_cleanup",
        name="Oil Spill Cleanup",
        description="Contain and clean up oil spills in the ocean using boom-equipped boats.",
        default_world_config={
            "boundary": "none",
            "k_nn": 0,  # No flocking/alignment
            "wa": 0.1,  # Minimal attraction - prevents clumping
            "wr": 5.0,  # Low repulsion - oil can overlap slightly
            "ws": 50.0,  # Strong response to boats
            "wm": 0.0,  # No inertia
            "vmax": 0.5,  # Slow movement
            "graze_alpha": 0.0,  # No random wandering
            "sigma": 0.0,  # No random noise
        },
        default_policy_config={
            "key": "default",
        },
        default_theme_key="oil-spill",
        default_icon_set="oil",
        recommended_agents=100,
        recommended_controllers=4,
        tags=["oil", "ocean", "cleanup", "slow"],
        environment="ocean",
    ),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def get_scenario_type(key: str) -> Optional[ScenarioTypeDefinition]:
    """Get a scenario type by key."""
    return SCENARIO_TYPES.get(key)


def list_scenario_types() -> List[ScenarioTypeDefinition]:
    """List all available scenario types."""
    return list(SCENARIO_TYPES.values())


def generate_initial_layout(
    scenario_type: ScenarioTypeDefinition,
    num_agents: Optional[int] = None,
    num_controllers: Optional[int] = None,
    bounds: tuple = (0.0, 250.0, 0.0, 250.0),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an initial entity layout based on scenario type.

    Args:
        scenario_type: The scenario type definition
        num_agents: Override for number of agents (uses recommended if None)
        num_controllers: Override for number of controllers (uses recommended if None)
        bounds: World bounds (xmin, xmax, ymin, ymax)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'sheep', 'drones', 'targets' keys containing position lists
    """
    rng = np.random.default_rng(seed)

    n_agents = num_agents or scenario_type.recommended_agents or DEFAULT_AGENT_COUNT
    n_controllers = (
        num_controllers
        or scenario_type.recommended_controllers
        or DEFAULT_CONTROLLER_COUNT
    )

    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # --- Generate Agents ---
    if "clusters" in scenario_type.tags:
        # Clusters: agents in 2-4 clusters
        n_clusters = min(4, max(2, n_agents // 30))
        agents_per_cluster = n_agents // n_clusters
        agent_groups = []

        for _ in range(n_clusters):
            cx = rng.uniform(xmin + width * 0.2, xmax - width * 0.2)
            cy = rng.uniform(ymin + height * 0.2, ymax - height * 0.2)
            cluster = rng.normal(
                loc=[cx, cy],
                scale=[width * 0.08, height * 0.08],
                size=(agents_per_cluster, 2),
            )
            agent_groups.append(cluster)

        agents = np.vstack(agent_groups)[:n_agents]
    else:
        # Default: moderately clustered in center region
        agents = rng.uniform(
            low=[center_x - width * 0.25, center_y - height * 0.25],
            high=[center_x + width * 0.25, center_y + height * 0.25],
            size=(n_agents, 2),
        )

    # Clip to bounds
    agents[:, 0] = np.clip(agents[:, 0], xmin + BOUNDS_PADDING, xmax - BOUNDS_PADDING)
    agents[:, 1] = np.clip(agents[:, 1], ymin + BOUNDS_PADDING, ymax - BOUNDS_PADDING)

    # --- Generate Controllers ---
    # Evenly spaced around the agents
    agent_center = agents.mean(axis=0)
    agent_radius = np.max(np.linalg.norm(agents - agent_center, axis=1))
    controller_radius = agent_radius * 1.5 + 20.0

    controllers = np.zeros((n_controllers, 2))
    for i in range(n_controllers):
        angle = 2 * np.pi * i / n_controllers
        controllers[i] = agent_center + controller_radius * np.array(
            [np.cos(angle), np.sin(angle)]
        )

    # Clip controllers to bounds
    controllers[:, 0] = np.clip(
        controllers[:, 0], xmin + BOUNDS_PADDING, xmax - BOUNDS_PADDING
    )
    controllers[:, 1] = np.clip(
        controllers[:, 1], ymin + BOUNDS_PADDING, ymax - BOUNDS_PADDING
    )

    # --- Generate Targets ---
    if scenario_type.key == "evacuation_prototype":
        targets = [[50.0, 50.0]]  # Top-Left for evacuation
    else:
        targets = [[center_x, center_y]]

    # --- Generate Obstacles ---
    obstacles: List[List[float]] = []

    return {
        "sheep": [[float(x), float(y)] for x, y in agents],
        "drones": [[float(x), float(y)] for x, y in controllers],
        "targets": targets,
        "obstacles": obstacles,
    }
