import numpy as np

from simulation import scenarios


def test_spawn_uniform():
    """Test uniform spawning within bounds."""
    N = 100
    bounds = (0, 100, 0, 100)
    pts = scenarios.spawn_uniform(N, bounds, seed=42)

    assert pts.shape == (N, 2)
    assert np.all(pts[:, 0] >= bounds[0])
    assert np.all(pts[:, 0] <= bounds[1])
    assert np.all(pts[:, 1] >= bounds[2])
    assert np.all(pts[:, 1] <= bounds[3])


def test_spawn_circle():
    """Test circle spawning."""
    N = 50
    center = (50, 50)
    radius = 10.0
    pts = scenarios.spawn_circle(N, center, radius, seed=42)

    assert pts.shape == (N, 2)
    # Check distances
    dists = np.linalg.norm(pts - np.array(center), axis=1)
    assert np.all(dists <= radius + 1e-9)


def test_spawn_line():
    """Test line spawning."""
    N = 20
    bounds = (0, 100, 0, 100)
    pts = scenarios.spawn_line(N, bounds, seed=42, y=50.0)

    assert pts.shape == (N, 2)
    # Check y spread (normal distribution, so most should be close to 50)
    # scale is 1.0, so 99.7% within +/- 3.0
    assert np.all(np.abs(pts[:, 1] - 50.0) < 5.0)
    # Check x spread (linspace)
    assert np.min(pts[:, 0]) >= bounds[0]
    assert np.max(pts[:, 0]) <= bounds[1]


def test_spawn_clusters():
    """Test cluster spawning."""
    N = 100
    k = 4
    bounds = (0, 100, 0, 100)
    pts = scenarios.spawn_clusters(N, k, bounds, seed=42)

    assert pts.shape == (N, 2)
    assert np.all(pts[:, 0] >= bounds[0] - 20)  # Allow some spillover due to spread
    assert np.all(pts[:, 0] <= bounds[1] + 20)

    # Check that we have roughly k clusters (hard to deterministicly verify without clustering algo)
    # But we can check basic properties
    assert len(pts) == N


def test_spawn_corners():
    """Test corner spawning."""
    N = 40
    bounds = (0, 100, 0, 100)
    pts = scenarios.spawn_corners(N, bounds, seed=42)

    assert pts.shape == (N, 2)
    # Points should be near corners (0,0), (0,100), (100,0), (100,100)
    # Check that points are NOT in the center
    center = np.array([50, 50])
    dists_to_center = np.linalg.norm(pts - center, axis=1)
    assert np.all(dists_to_center > 20.0)


def test_spawn_edge_cases():
    """Test spawning with N=0 or N=1."""
    bounds = (0, 100, 0, 100)

    # N=0
    pts0 = scenarios.spawn_uniform(0, bounds)
    assert pts0.shape == (0, 2)

    # N=1
    pts1 = scenarios.spawn_uniform(1, bounds)
    assert pts1.shape == (1, 2)
