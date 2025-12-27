"""
Shepherd Policy

This module implements the `ShepherdPolicy` class, which defines the multi-drone
herding strategy. It uses a combination of global center of mass (GCM) tracking,
goal attraction, and individual sheep targeting to drive the flock towards a destination.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from planning import state
from planning.herding.utils import (
    lerp_clamped,
    points_inside_polygon,
    closest_point_on_polygon,
)
from planning.plan_type import DoNothing, DronePositions, Plan

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EPSILON = 1e-9


def is_goal_satisfied(w: state.State, target: state.Target) -> bool:
    """
    Return True if every sheep in the world's flock is within the goal tolerance
    of the world's target.
    """
    if w.flock.size == 0:
        return True

    if isinstance(target, state.Circle) and target.radius is not None:
        # squared comparison for speed / numerical stability
        tol_sq = target.radius * target.radius

        # distances squared from each sheep to the target
        diffs = w.flock - target.center.reshape(1, 2)
        d2 = np.sum(diffs * diffs, axis=1)

        return bool(np.all(d2 <= tol_sq))
    elif isinstance(target, state.Polygon):
        return bool(np.all(points_inside_polygon(w.flock, target.points)))

    return False


# -----------------------------------------------------------------------------
# Shepherd Policy
# -----------------------------------------------------------------------------


class ShepherdPolicy:
    """
    Collect/drive policy modified for multiple drones.

    Strategy:
    - Drive: All drones drive to sector-assigned drive points behind the G-to-Target line.
    - Collect: Each drone collects an assigned outermost sheep.
    """

    def __init__(
        self,
        *,
        fN: float,
        umax: float,
        too_close: float,
        collect_standoff: float,
        conditionally_apply_repulsion: bool = True,
    ):
        self.fN = fN
        self.umax = umax
        self.too_close = too_close
        self.collect_standoff = collect_standoff
        self.conditionally_apply_repulsion = conditionally_apply_repulsion

    def _gcm(self, world: state.State) -> np.ndarray:
        """Calculate Global Center of Mass (GCM) of the flock."""
        return np.mean(world.flock, axis=0)

    def _cohesive(self, world: state.State, G: np.ndarray) -> bool:
        """Check for flock cohesiveness (boolean)."""
        return self._cohesiveness(world, G) > 1

    def _cohesiveness(self, world: state.State, G: np.ndarray) -> float:
        """
        Calculate flock cohesiveness metric based on max distance from GCM.
        Value > 1 means flock is contained within fN radius.
        """
        if world.flock.shape[0] == 0:
            return 100.0  # Arbitrary high value for empty flock
        r = np.max(np.linalg.norm(world.flock - G, axis=1))
        return self.fN / (r + EPSILON)

    def _mean_cohesiveness(self, world: state.State, G: np.ndarray) -> float:
        """
        Calculate flock cohesiveness metric based on mean distance from GCM.
        """
        if world.flock.shape[0] == 0:
            return 100.0
        r = np.mean(np.linalg.norm(world.flock - G, axis=1))
        return self.fN / (r + EPSILON)

    # ------------------ Multi-Drone Collect Logic ------------------

    def _collect_points(
        self, world: state.State, G: np.ndarray, target: Optional[state.Target]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Assigns each drone to collect an individual outermost sheep.
        Returns target points for all drones and the indices of the assigned sheep.
        The target point will be NaN if there is no sheep for that drone to target.
        """
        P = world.flock  # (N,2)
        D = world.drones  # (N_drones, 2)
        N_drones = D.shape[0]

        if target is None:
            # If no target is set, use a default position or skip goal-based calculations
            dGoal = np.zeros(P.shape[0])  # No goal distance when no target
        elif isinstance(target, state.Circle):
            dGoal = np.linalg.norm(P - target.center, axis=1)  # distance to goal
            # If the sheep is inside the goal, set the distance to -inf so it won't be targeted.
            if target.radius is not None:
                dGoal = np.where(dGoal < target.radius, -np.inf, dGoal)
        elif isinstance(target, state.Polygon):
            # Compute the distance to the closest point on the polygon (vectorized)
            closest_points = closest_point_on_polygon(P, target.points)
            dGoal = np.linalg.norm(P - closest_points, axis=1)
            # If the point is inside the polygon (or close enough), we consider it "inside"
            # for the purpose of pushing it out.
            # However, for the shepherd, we want to push the sheep *into* the polygon if it's outside?
            # The logic here seems to be: if the sheep is NOT in the polygon, we want to push it towards the polygon.
            # But the code says "if not points_inside_polygon...".
            # Let's assume the target is to be INSIDE the polygon.
            polygon_inside = points_inside_polygon(P, target.points)
            dGoal = np.where(polygon_inside, -np.inf, dGoal)
        else:
            dGoal = np.zeros(P.shape[0])

        dG = np.linalg.norm(P - G, axis=1)  # distance to global COM

        # Calculate dD for each sheep to ALL drones
        dD_all = np.zeros((P.shape[0], N_drones))
        for i in range(N_drones):
            dD_all[:, i] = np.linalg.norm(P - D[i], axis=1)

        # Dynamic weighting based on cohesiveness and goal progress
        cohesiveness = self._mean_cohesiveness(world, G)
        goal_distance_ratio = np.max(dGoal) / self.fN

        # The more cohesive it is, the less we care about the GCM.
        gcm_weight = lerp_clamped(0.8, 0.6, 0.3, 1.5, cohesiveness)
        # The closer we are to the goal, the less it matters how far from the GCM the sheep is.
        gcm_weight *= lerp_clamped(0.5, 1, 1, 3, goal_distance_ratio)

        # The more cohesive it is, the more we care about how far it is from the goal.
        goal_weight = lerp_clamped(0.2, 0.4, 0.3, 1.5, cohesiveness)

        # The more cohesive the herd is, the less it matters how far the drone is from the sheep.
        closeness_weight = lerp_clamped(1, 0.2, 0.3, 1.5, cohesiveness)
        # The closer the sheep are to the goal, the less it matters how far the drone is from the sheep.
        closeness_weight *= lerp_clamped(0.2, 1, 2, 4, goal_distance_ratio)

        intrinsic_score = gcm_weight * dG + goal_weight * dGoal

        # Build score matrix: (N_sheep, N_drones)
        # Each entry [j, i] is the score for assigning sheep j to drone i
        N_sheep = P.shape[0]
        score_matrix = np.zeros((N_sheep, N_drones))

        # Compute scores for each drone-sheep pair
        for i in range(N_drones):
            # If there are more drones, the closeness will get accounted for by the min_distance_other stuff.
            current_closeness_weight = closeness_weight * (1 / N_drones)

            # Make the score worse the farther away that sheep is from this drone.
            score = intrinsic_score - current_closeness_weight * dD_all[:, i]

            # Compute how close the other drones are to each sheep.
            d_other_drones = np.hstack((dD_all[:, :i], dD_all[:, i + 1 :]))
            if N_drones > 1:
                min_distance_other = np.min(d_other_drones, axis=1)
                # Give a bonus if this drone is the closest to the sheep.
                score += 30 * (dD_all[:, i] < min_distance_other)

            # Store scores for this drone in the matrix
            score_matrix[:, i] = score

        # Iteratively assign drones to sheep using greedy selection
        # Each drone must pick a different sheep
        # Assumes there are at least N_drones sheep in the flock
        target_sheep_indices: List[Optional[int]] = [None] * N_drones
        assigned_sheep = set()

        # TODO: idk why this isn't done using a numpy function.
        for _ in range(N_drones):
            # Find the absolute maximum score in the entire matrix
            # Ignore already assigned sheep
            max_score = -np.inf
            best_sheep_idx = None
            best_drone_idx = None

            for sheep_idx in range(N_sheep):
                if sheep_idx in assigned_sheep:
                    continue  # Skip already assigned sheep
                for drone_idx in range(N_drones):
                    if target_sheep_indices[drone_idx] is not None:
                        continue  # Skip already assigned drones
                    if score_matrix[sheep_idx, drone_idx] > max_score:
                        max_score = score_matrix[sheep_idx, drone_idx]
                        best_sheep_idx = sheep_idx
                        best_drone_idx = drone_idx

            # Assign this sheep-drone pair
            if best_sheep_idx is not None and best_drone_idx is not None:
                target_sheep_indices[best_drone_idx] = int(best_sheep_idx)
                assigned_sheep.add(best_sheep_idx)

                # Set this sheep's score to -inf for all drones (so no other drone can pick it)
                for remaining_drone_idx in range(N_drones):
                    score_matrix[best_sheep_idx, remaining_drone_idx] = -np.inf
            else:
                # No valid assignment found (shouldn't happen if there are enough sheep),
                # but when collection has finished this may trigger.
                break

        # Calculate the standoff point for each assigned sheep
        collect_points = np.full((N_drones, 2), np.nan)
        for i, target_index in enumerate(target_sheep_indices):
            if target_index is None:
                continue

            Pj = P[target_index]

            # Point behind that sheep, pointing toward G
            dir_to_G = G - Pj
            c = dir_to_G / (np.linalg.norm(dir_to_G) + EPSILON)
            collect_points[i] = Pj - c * self.collect_standoff

        return collect_points, target_sheep_indices  # type: ignore

    # ------------------ Flyover Logic (Per-Drone) ------------------

    def _should_apply_repulsion(
        self,
        world: state.State,
        drone_idx: int,
        gcm: np.ndarray,
        target: Optional[state.Target],
        target_positions: np.ndarray,
    ) -> bool:
        """
        Check if a specific drone should apply repulsion.
        Returns true if the drone is close to its collect point or based on cohesiveness.
        """
        if np.isnan(target_positions[drone_idx]).any():
            return False

        # If the drone is close to its collect point, then it should always apply repulsion
        # The more cohesive, the more likely we just want to apply repulsion.
        repulsion_threshold = lerp_clamped(
            2, 5, 0.8, 1.2, self._cohesiveness(world, gcm)
        )

        if (
            np.linalg.norm(target_positions[drone_idx] - world.drones[drone_idx])
            < repulsion_threshold
        ):
            return True

        return False

    # ------------------ Main Planning Method ------------------

    def plan(self, world: state.State, jobs: List[state.Job], dt: float) -> Plan:
        """Return the movement plan for all drones."""
        target = None
        all_jobs_satisfied = True

        for job in jobs:
            if job.is_active and job.target is not None:
                if not is_goal_satisfied(world, job.target):
                    all_jobs_satisfied = False

                # TODO: Assign drones to different jobs instead of just picking the first active one
                target = job.target
                break

        if all_jobs_satisfied:
            return DoNothing()

        N_drones = world.drones.shape[0]
        G = self._gcm(world)

        # Initialize arrays for the plan
        apply_repulsion = np.full(N_drones, 1)

        # COLLECT PHASE: Each drone targets an outermost sheep's standoff point
        target_positions, target_indices = self._collect_points(world, G, target)

        # Check flyover status for each drone individually
        if self.conditionally_apply_repulsion:
            for i in range(N_drones):
                # Check if the path from current drone position to its assigned collect point needs a flyover
                apply_repulsion[i] = self._should_apply_repulsion(
                    world, i, G, target, target_positions
                )

        # Vector from drone to target position
        dir_to_target = target_positions - world.drones

        # Calculate unit direction vector for all drones
        dist = np.linalg.norm(dir_to_target, axis=1)
        dir_unit = dir_to_target / (dist[:, None] + EPSILON)

        # If a drone is too close to a sheep, then stop that drone.
        too_close_sq = self.too_close**2
        new_positions = np.copy(world.drones)

        for i, d in enumerate(world.drones):
            dist_sq = np.min(np.sum((world.flock - d) ** 2, axis=1))
            if np.isnan(dir_unit[i]).any():
                continue
            if dist_sq >= too_close_sq or not apply_repulsion[i]:
                # Only move drones that are not too close, or if they are in flyover mode
                new_positions[i] += self.umax * dt * dir_unit[i]
            # else: drone is too close and applying repulsion, so it stays put

        return DronePositions(
            positions=new_positions,
            apply_repulsion=apply_repulsion,
            target_sheep_indices=target_indices,
            gcm=G,
            radius=self.fN,
        )
