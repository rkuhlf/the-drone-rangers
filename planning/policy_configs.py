"""
PolicyConfig system for configurable herding behavior.

This module provides a single source of truth for herding behavior parameters,
with named presets and support for custom overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

# -----------------------------------------------------------------------------
# Types & Constants
# -----------------------------------------------------------------------------

StrategyMode = Literal["gentle", "aggressive", "defensive", "patrol"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """
    Configuration for herding policy behavior.

    This dataclass defines all tunable parameters for the ShepherdPolicy,
    including strategy multipliers, mode-level behavior, multi-drone coordination,
    and target selection weights.
    """

    # Identity/meta
    key: str = "default"
    name: str = "Default Policy"
    description: str = "Standard gentle herding behavior"

    # Strategy multipliers (scale base world parameters)
    fN_multiplier: float = 1.0  # Collected herd radius multiplier
    too_close_multiplier: float = 1.5  # Minimum distance to sheep
    collect_standoff_multiplier: float = 1.0  # Collection standoff distance

    # Mode-level behavior
    strategy_mode: StrategyMode = "gentle"
    conditionally_apply_repulsion: bool = True

    # Multi-drone coordination
    sector_based_collection: bool = False
    drone_spacing_factor: float = 1.0

    # Target selection weights
    gcm_weight: float = 0.7  # Weight for distance to global center of mass
    goal_weight: float = 0.3  # Weight for distance to goal
    closeness_weight: float = 1.0  # Weight for drone-sheep closeness

    # Phase 3: Strategy multipliers for velocity/force adjustments
    drive_force_multiplier: float = 1.0  # Scales drive force toward target
    repulsion_weight_multiplier: float = 1.0  # Scales repulsion between drones/agents
    goal_bias_multiplier: float = 1.0  # Biases movement toward goal vs GCM
    max_speed_multiplier: float = 1.0  # Scales maximum drone speed

    def to_dict(self) -> Dict[str, Any]:
        """Convert PolicyConfig to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyConfig:
        """
        Create PolicyConfig from dictionary, ignoring unknown keys.

        This is backwards-compatible: extra keys in the dict are silently ignored,
        and missing keys will use the default values from the dataclass.
        """
        # Extract only the fields that PolicyConfig knows about
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

# Named presets for common herding strategies
POLICY_PRESETS: Dict[str, PolicyConfig] = {
    "default": PolicyConfig(
        key="default",
        name="Default (Gentle)",
        description="Standard gentle herding behavior with balanced parameters",
        strategy_mode="gentle",
        fN_multiplier=1.0,
        too_close_multiplier=1.5,
        collect_standoff_multiplier=1.0,
        conditionally_apply_repulsion=True,
        gcm_weight=0.7,
        goal_weight=0.3,
        closeness_weight=1.0,
        # All Phase 3 multipliers at 1.0 for unchanged behavior
        drive_force_multiplier=1.0,
        repulsion_weight_multiplier=1.0,
        goal_bias_multiplier=1.0,
        max_speed_multiplier=1.0,
    ),
    "aggressive": PolicyConfig(
        key="aggressive",
        name="Aggressive",
        description="Faster, more forceful herding - closer approach, maintains momentum",
        strategy_mode="aggressive",
        fN_multiplier=1.2,  # Larger collection radius
        too_close_multiplier=1.2,  # Slightly closer approach allowed
        collect_standoff_multiplier=0.8,  # Tighter standoff
        conditionally_apply_repulsion=True,
        gcm_weight=0.6,
        goal_weight=0.4,  # More goal-focused
        closeness_weight=1.2,
        # Aggressive: 25% faster movement for quicker herding
        drive_force_multiplier=1.25,
        repulsion_weight_multiplier=1.0,
        goal_bias_multiplier=1.2,
        max_speed_multiplier=1.15,
    ),
    "defensive": PolicyConfig(
        key="defensive",
        name="Defensive (Containment)",
        description="Slower, careful herding - prioritizes flock cohesion over speed",
        strategy_mode="defensive",
        fN_multiplier=0.9,  # Tighter collection radius
        too_close_multiplier=1.6,  # Keep more distance from sheep
        collect_standoff_multiplier=1.2,  # Larger standoff
        conditionally_apply_repulsion=True,
        gcm_weight=0.8,  # More focus on keeping flock together
        goal_weight=0.2,
        closeness_weight=0.8,
        # Defensive: 20% slower for gentler pressure
        drive_force_multiplier=0.8,
        repulsion_weight_multiplier=1.0,
        goal_bias_multiplier=0.8,
        max_speed_multiplier=0.85,
    ),
    "patrol": PolicyConfig(
        key="patrol",
        name="Patrol (Containment)",
        description="Perimeter containment - drones orbit flock to prevent wandering (does NOT drive to goal)",
        strategy_mode="patrol",
        fN_multiplier=1.1,
        too_close_multiplier=1.5,
        collect_standoff_multiplier=1.5,
        conditionally_apply_repulsion=True,  # Apply repulsion for containment
        gcm_weight=0.5,
        goal_weight=0.5,
        closeness_weight=0.5,
        # Patrol: moderate speed for smooth orbiting
        drive_force_multiplier=0.9,
        repulsion_weight_multiplier=1.0,
        goal_bias_multiplier=1.0,
        max_speed_multiplier=0.9,
    ),
    "evacuation-prototype": PolicyConfig(
        key="evacuation-prototype",
        name="Evacuation Prototype",
        description="Experimental mode for human evacuation scenarios: spread-aware, less direct",
        strategy_mode="gentle",
        fN_multiplier=1.0,
        too_close_multiplier=2.0,  # Keep more distance from agents
        collect_standoff_multiplier=1.3,
        conditionally_apply_repulsion=True,
        gcm_weight=0.5,
        goal_weight=0.5,
        closeness_weight=0.7,
        # Evacuation: lower goal bias, higher repulsion awareness
        drive_force_multiplier=0.9,
        repulsion_weight_multiplier=1.3,
        goal_bias_multiplier=0.7,
        max_speed_multiplier=0.95,
    ),
}


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_policy(world, policy_config: Optional[Union[str, dict, PolicyConfig]] = None):
    """
    Build a ShepherdPolicy from various config input formats.

    Args:
        world: The World instance to attach the policy to
        policy_config: Can be:
            - None: uses "default" preset
            - str: preset key (e.g. "aggressive"), falls back to "default" if not found
            - dict: either a preset override (if "key" field matches a preset)
                   or a complete custom config
            - PolicyConfig: use directly

    Returns:
        ShepherdPolicy instance configured with the specified parameters
    """
    from planning.herding.policy import ShepherdPolicy

    # Handle None -> default preset
    if policy_config is None:
        config = POLICY_PRESETS["default"]

    # Handle string -> preset lookup
    elif isinstance(policy_config, str):
        config = POLICY_PRESETS.get(policy_config, POLICY_PRESETS["default"])

    # Handle dict -> either preset override or custom config
    elif isinstance(policy_config, dict):
        # Check if this is a preset override (has "key" field matching a preset)
        key = policy_config.get("key")
        if key and key in POLICY_PRESETS:
            # Start from preset, overlay custom fields
            base_dict = POLICY_PRESETS[key].to_dict()
            base_dict.update(policy_config)
            config = PolicyConfig.from_dict(base_dict)
        else:
            # Construct PolicyConfig directly from dict
            config = PolicyConfig.from_dict(policy_config)

    # Handle PolicyConfig -> use as-is
    elif isinstance(policy_config, PolicyConfig):
        config = policy_config

    else:
        raise TypeError(f"Unsupported policy_config type: {type(policy_config)}")

    # Calculate derived parameters from config and world
    # Calculate base fN from world flock size
    total_area = 0.5 * world.N * (world.ra**2)
    base_fN = np.sqrt(total_area)

    # Apply config multipliers to world parameters
    fN = base_fN * config.fN_multiplier
    umax = world.umax * config.max_speed_multiplier
    too_close = config.too_close_multiplier * world.ra
    collect_standoff = config.collect_standoff_multiplier * world.ra

    # Create and return the policy with direct parameters
    return ShepherdPolicy(
        fN=fN,
        umax=umax,
        too_close=too_close,
        collect_standoff=collect_standoff,
        conditionally_apply_repulsion=config.conditionally_apply_repulsion,
    )
