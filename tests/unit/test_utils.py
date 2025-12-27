import numpy as np

from planning.herding import utils
from planning.herding import policy

# -----------------------------------------------------------------------------
# Math Utils Tests
# -----------------------------------------------------------------------------


def test_norm():
    """Test vector normalization."""
    # Test single vector
    v = np.array([3.0, 4.0])
    n = utils.norm(v)
    expected = np.array([0.6, 0.8])
    np.testing.assert_allclose(n, expected, atol=1e-9)

    # Test zero vector (should handle epsilon)
    v_zero = np.array([0.0, 0.0])
    n_zero = utils.norm(v_zero)
    # With epsilon, it divides by epsilon, so it's 0/eps = 0
    np.testing.assert_allclose(n_zero, v_zero, atol=1e-9)

    # Test negative components
    v_neg = np.array([-3.0, -4.0])
    n_neg = utils.norm(v_neg)
    np.testing.assert_allclose(n_neg, [-0.6, -0.8], atol=1e-9)

    # Test batch of vectors (if supported, though implementation seems single-vector focused)
    # The current implementation uses np.linalg.norm(v) which returns a scalar if v is 1D,
    # or Frobenius norm if 2D. It doesn't seem designed for batch row-wise norm.
    # Let's verify that behavior:
    # v_batch = np.array([[3, 4], [1, 0]])
    # norm([[3,4],[1,0]]) -> norm is ~5.09 -> returns v / 5.09
    # This confirms it's NOT row-wise. So we won't test batch.


def test_smooth_push():
    """Test smooth push force."""
    # Test at 0 distance
    assert utils.smooth_push(0.0, 10.0) == 1.0

    # Test at > rs distance
    assert utils.smooth_push(11.0, 10.0) == 0.0

    # Test at rs distance
    assert utils.smooth_push(10.0, 10.0) < 1e-6

    # Test at half distance
    assert 0.4 < utils.smooth_push(5.0, 10.0) < 0.6

    # Test negative distance (should be clamped to 1.0 max? or > 1.0?)
    # Formula: max(0, 1 - dist/rs). If dist is negative, 1 - (-val) > 1.
    # Let's check if implementation clamps max.
    # Implementation: np.maximum(0.0, 1.0 - dist / (rs + eps))
    # It does NOT clamp max to 1.0.
    assert utils.smooth_push(-5.0, 10.0) > 1.0


def test_lerp_clamped():
    """Test linear interpolation with clamping."""
    # Normal range
    assert policy.lerp_clamped(0, 10, 0, 1, 0.5) == 5.0

    # Clamped low
    assert policy.lerp_clamped(0, 10, 0, 1, -0.5) == 0.0

    # Clamped high
    assert policy.lerp_clamped(0, 10, 0, 1, 1.5) == 10.0

    # Inverse range (t1 > t2) - usually not supported or behaves weirdly
    # Let's stick to standard usage


# -----------------------------------------------------------------------------
# Geometry Tests
# -----------------------------------------------------------------------------


def test_points_inside_polygon_robust():
    """Test point-in-polygon check with edge cases."""
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    # Inside
    assert policy.points_inside_polygon(np.array([[5, 5]]), poly)[0]

    # Outside
    assert not policy.points_inside_polygon(np.array([[15, 5]]), poly)[0]

    # On edge (implementation dependent)
    # Ray casting usually handles edges consistently (e.g., top-left rule)

    # Multiple points
    pts = np.array([[5, 5], [15, 15], [5, -5]])
    results = policy.points_inside_polygon(pts, poly)
    np.testing.assert_array_equal(results, [True, False, False])

    # Degenerate polygon (line)
    poly_line = np.array([[0, 0], [10, 0]])
    assert not policy.points_inside_polygon(np.array([[5, 0]]), poly_line)[0]

    # Empty polygon
    # poly_empty = np.array([])
    # Should handle gracefully (return False or raise error, implementation checks shape < 3)
    # The implementation returns zeros if shape[0] < 3
    assert not policy.points_inside_polygon(np.array([[5, 5]]), np.zeros((0, 2)))[0]


def test_closest_point_on_polygon_robust():
    """Test closest point on polygon with edge cases."""
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    # Point outside right edge
    pt = np.array([[15.0, 5.0]])
    closest = policy.closest_point_on_polygon(pt, poly)
    np.testing.assert_allclose(closest, [[10.0, 5.0]], atol=1e-9)

    # Point inside (closest is on edge)
    pt_in = np.array([[5.0, 1.0]])
    closest_in = policy.closest_point_on_polygon(pt_in, poly)
    np.testing.assert_allclose(closest_in, [[5.0, 0.0]], atol=1e-9)

    # Point on vertex
    pt_vert = np.array([[10.0, 10.0]])
    closest_vert = policy.closest_point_on_polygon(pt_vert, poly)
    np.testing.assert_allclose(closest_vert, [[10.0, 10.0]], atol=1e-9)

    # Degenerate polygon (single point)
    poly_pt = np.array([[5, 5]])
    closest_pt = policy.closest_point_on_polygon(np.array([[0, 0]]), poly_pt)
    np.testing.assert_allclose(closest_pt, [[5, 5]], atol=1e-9)

    # Degenerate polygon (empty)
    poly_empty = np.zeros((0, 2))
    closest_empty = policy.closest_point_on_polygon(np.array([[1, 1]]), poly_empty)
    # Should return original points if empty
    np.testing.assert_allclose(closest_empty, [[1, 1]], atol=1e-9)
