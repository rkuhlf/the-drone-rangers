"""
Scenario Spawning Functions

This module provides functions to generate initial positions for agents (sheep)
in various geometric patterns such as uniform distribution, clusters, corners,
lines, and circles. These are used to initialize simulation scenarios.
"""

from typing import Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Spawning Functions
# -----------------------------------------------------------------------------


def spawn_uniform(
    N: int, bounds: Tuple[float, float, float, float], seed: int = 2
) -> np.ndarray:
    """
    Generates N points uniformly distributed within the given bounds.

    Args:
        N: Number of points to generate.
        bounds: Tuple of (xmin, xmax, ymin, ymax).
        seed: Random seed.

    Returns:
        (N, 2) array of point coordinates.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    # Add a small buffer from the edges
    x = rng.uniform(xmin + 1, xmax - 1, N)
    y = rng.uniform(ymin + 1, ymax - 1, N)

    return np.stack([x, y], axis=1)


def spawn_clusters(
    N: int,
    k: int,
    bounds: Tuple[float, float, float, float],
    spread: float = 3.5,
    seed: int = 2,
) -> np.ndarray:
    """
    Generates N points distributed in k Gaussian clusters.

    Args:
        N: Total number of points.
        k: Number of clusters.
        bounds: Tuple of (xmin, xmax, ymin, ymax).
        spread: Standard deviation of the clusters.
        seed: Random seed.

    Returns:
        (N, 2) array of point coordinates.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    # Generate cluster centers well within bounds
    centers = np.stack(
        [rng.uniform(xmin + 6, xmax - 6, k), rng.uniform(ymin + 6, ymax - 6, k)], axis=1
    )

    pts = []
    base = N // k
    extras = N - base * k

    # Distribute points among clusters
    sizes = [base + (1 if i < extras else 0) for i in range(k)]

    for i, c in enumerate(centers):
        cov = np.eye(2) * (spread**2)
        pts.append(rng.multivariate_normal(c, cov, sizes[i]))

    return np.vstack(pts)


def spawn_corners(
    N: int,
    bounds: Tuple[float, float, float, float],
    jitter: float = 2.0,
    seed: int = 2,
) -> np.ndarray:
    """
    Generates N points distributed among the four corners of the bounds.

    Args:
        N: Number of points.
        bounds: Tuple of (xmin, xmax, ymin, ymax).
        jitter: Standard deviation of the noise around corners.
        seed: Random seed.

    Returns:
        (N, 2) array of point coordinates.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    corners = np.array(
        [
            [xmin + 2, ymin + 2],
            [xmin + 2, ymax - 2],
            [xmax - 2, ymin + 2],
            [xmax - 2, ymax - 2],
        ],
        dtype=float,
    )

    pts = []
    for i in range(N):
        c = corners[i % 4]
        pts.append(c + rng.normal(scale=jitter, size=2))

    return np.array(pts)


def spawn_line(
    N: int,
    bounds: Tuple[float, float, float, float],
    seed: int = 2,
    y: Optional[float] = None,
) -> np.ndarray:
    """
    Generates N points along a horizontal line with some vertical jitter.

    Args:
        N: Number of points.
        bounds: Tuple of (xmin, xmax, ymin, ymax).
        seed: Random seed.
        y: Optional fixed y-coordinate. If None, chosen randomly.

    Returns:
        (N, 2) array of point coordinates.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    if y is None:
        y = rng.uniform(ymin + 5, ymax - 5)

    xs = np.linspace(xmin + 2, xmax - 2, N)
    ys = rng.normal(loc=y, scale=1.0, size=N)

    return np.stack([xs, ys], axis=1)


def spawn_circle(
    N: int, center: Tuple[float, float] = (0, 0), radius: float = 5.0, seed: int = 2
) -> np.ndarray:
    """
    Generates N points uniformly distributed within a circle.

    Args:
        N: Number of points.
        center: (x, y) center of the circle.
        radius: Radius of the circle.
        seed: Random seed.

    Returns:
        (N, 2) array of point coordinates.
    """
    rng = np.random.default_rng(seed)
    c = np.array(center, float)

    # Generate random angles and radii (sqrt for uniform area distribution)
    th = rng.random(N) * 2 * np.pi
    r = radius * np.sqrt(rng.random(N))

    return c + np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
