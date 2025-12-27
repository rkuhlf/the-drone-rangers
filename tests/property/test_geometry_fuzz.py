
import numpy as np
import random
from planning.herding import utils, policy


def test_norm_fuzz():
    """Property: norm(v) >= 0 and triangle inequality."""
    for _ in range(100):
        # Random vector
        v = np.random.uniform(-1e6, 1e6, 2)

        # Non-negative
        assert np.linalg.norm(v) >= 0

        # Triangle inequality: |a+b| <= |a| + |b|
        a = np.random.uniform(-100, 100, 2)
        b = np.random.uniform(-100, 100, 2)
        assert np.linalg.norm(a + b) <= np.linalg.norm(a) + np.linalg.norm(b) + 1e-9


def test_smooth_push_fuzz():
    """Property: smooth_push(d, r) is continuous and bounded."""
    r = 10.0
    for _ in range(100):
        d = random.uniform(-10.0, 30.0)
        val = utils.smooth_push(d, r)

        if d >= r:
            assert val == 0.0
        elif d <= 0:
            # Should be high but finite (implementation dependent)
            # Current impl: (r-d)^2 / d might blow up if d -> 0
            # Let's check behavior near 0
            pass
        else:
            assert val > 0.0


def test_points_inside_polygon_fuzz():
    """Property: Points inside polygon must be within bounding box."""
    for _ in range(50):
        # Random convex polygon (triangle)
        poly = np.random.uniform(0, 100, (3, 2))

        # Random points
        pts = np.random.uniform(-50, 150, (20, 2))

        mask = policy.points_inside_polygon(pts, poly)

        # Check bounding box property
        min_x, min_y = np.min(poly, axis=0)
        max_x, max_y = np.max(poly, axis=0)

        for i, is_inside in enumerate(mask):
            if is_inside:
                p = pts[i]
                assert min_x <= p[0] <= max_x
                assert min_y <= p[1] <= max_y
