"""
Herding Utilities

This module provides common mathematical utility functions used across the herding
planning and simulation logic, such as vector normalization and force smoothing.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Vector Operations
# -----------------------------------------------------------------------------


def norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normalize a vector `v` to unit length.

    Args:
        v: Input vector (NumPy array).
        eps: Small epsilon to prevent division by zero.

    Returns:
        The normalized unit vector.
    """
    x = np.linalg.norm(v)
    return v / (x + eps)


# -----------------------------------------------------------------------------
# Force Functions
# -----------------------------------------------------------------------------


def smooth_push(
    dist: float | np.ndarray, rs: float, eps: float = 1e-9
) -> float | np.ndarray:
    """
    Calculate a smooth influence scalar in [0, 1] based on distance.

    The influence is 1.0 when distance is 0, and linearly decreases to 0.0
    at distance `rs`. Beyond `rs`, the influence is 0.0.

    Args:
        dist: Distance from the source (float or NumPy array).
        rs: The sensing radius or maximum influence distance.
        eps: Small epsilon (unused in this formula but kept for consistency).

    Returns:
        Influence value in [0, 1].
    """
    # Note: eps is not strictly needed for division here but kept for signature consistency
    return np.maximum(0.0, 1.0 - dist / (rs + eps))


# -----------------------------------------------------------------------------
# Geometry Helpers
# -----------------------------------------------------------------------------


def lerp_clamped(a: float, b: float, t1: float, t2: float, t: float) -> float:
    """Linearly interpolate between a and b by t, but clamp t to [0,1]."""
    t_norm = (t - t1) / (t2 - t1)
    t_norm = max(0.0, min(1.0, t_norm))
    return a + (b - a) * t_norm


def points_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Check which points are inside a polygon using the ray casting algorithm (vectorized).

    Args:
        points: 2D points as np.ndarray of shape (n, 2)
        polygon: Polygon vertices as np.ndarray of shape (m, 2)

    Returns:
        Boolean array of shape (n,) indicating which points are inside the polygon
    """
    if polygon.shape[0] < 3:
        # Need at least 3 vertices for a valid polygon
        return np.zeros(points.shape[0], dtype=bool)

    n_points = points.shape[0]
    n_vertices = polygon.shape[0]
    inside = np.zeros(n_points, dtype=bool)

    # For each point, cast a ray to the right and count intersections
    for i in range(n_points):
        px, py = points[i]
        intersections = 0

        for j in range(n_vertices):
            v1 = polygon[j]
            v2 = polygon[(j + 1) % n_vertices]

            x1, y1 = v1
            x2, y2 = v2

            # Check if ray crosses this edge
            if (y1 > py) != (y2 > py):  # Edge crosses horizontal line through point
                # Compute x-coordinate of intersection
                if y2 != y1:  # Avoid division by zero
                    x_intersect = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                    if px < x_intersect:
                        intersections += 1

        # Odd number of intersections means point is inside
        inside[i] = (intersections % 2) == 1

    return inside


def closest_point_on_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Find the closest point on a polygon for each point in the input array (vectorized).

    Args:
        points: 2D points as np.ndarray of shape (n, 2)
        polygon: Polygon vertices as np.ndarray of shape (m, 2)

    Returns:
        The closest points on the polygon for each input point as np.ndarray of shape (n, 2)
    """
    if polygon.shape[0] < 2:
        # Degenerate polygon - return first vertex for all points
        if polygon.shape[0] > 0:
            return np.tile(polygon[0], (points.shape[0], 1))
        else:
            return points.copy()

    n_points = points.shape[0]
    n_vertices = polygon.shape[0]
    EPSILON = 1e-9

    # Initialize with large distances
    min_dist_sq = np.full(n_points, np.inf)
    closest_points = np.zeros((n_points, 2))

    # Check each edge of the polygon
    for i in range(n_vertices):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n_vertices]  # Wrap around to close the polygon

        # Vector along the edge
        edge_vec = v2 - v1
        edge_len_sq = np.dot(edge_vec, edge_vec)

        if edge_len_sq < EPSILON:
            # Degenerate edge (zero length) - check distance to vertex for all points
            to_vertex = points - v1  # (n_points, 2)
            dist_sq = np.sum(to_vertex**2, axis=1)  # (n_points,)

            # Update where this vertex is closer
            mask = dist_sq < min_dist_sq
            closest_points[mask] = v1
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
            continue

        # Vector from v1 to each point: (n_points, 2)
        to_points = points - v1

        # Project each point onto the edge line
        # t = dot(to_points, edge_vec) / dot(edge_vec, edge_vec)
        t = np.dot(to_points, edge_vec) / edge_len_sq  # (n_points,)

        # Clamp t to [0, 1] to stay on the segment
        t = np.clip(t, 0.0, 1.0)

        # Closest points on the edge segment: (n_points, 2)
        closest_on_edge = v1 + t[:, np.newaxis] * edge_vec

        # Distance squared from each point to closest point on edge: (n_points,)
        dist_sq = np.sum((points - closest_on_edge) ** 2, axis=1)

        # Update where this edge gives a closer point
        mask = dist_sq < min_dist_sq
        closest_points[mask] = closest_on_edge[mask]
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return closest_points
