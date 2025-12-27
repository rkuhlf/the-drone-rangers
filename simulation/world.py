"""
World Simulation

This module defines the `World` class, which manages the simulation state,
including agents (sheep), controllers (drones/shepherds), obstacles, and physics.
It supports both NumPy and Numba (optional) for performance.
"""

from __future__ import annotations

import numpy as np

from planning import state
from planning.herding.utils import norm, smooth_push
from planning.plan_type import DoNothing, DronePositions, Plan

# -----------------------------------------------------------------------------
# Numba Support
# -----------------------------------------------------------------------------

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

DEFAULT_BOUNDS = (0.0, 250.0, 0.0, 250.0)
EPSILON = 1e-9
LARGE_DISTANCE = 1_000_000.0


# -----------------------------------------------------------------------------
# World Class
# -----------------------------------------------------------------------------


class World:
    """
    Simulates a flock of agents (sheep) and controllers (drones/shepherds)
    within a defined boundary, subject to physical forces and constraints.
    """

    def __init__(
        self,
        sheep_xy: np.ndarray,
        shepherd_xy: np.ndarray,  # n-by-2 of the positions of all of the shepherds.
        target_xy: list[float] | None = None,
        *,
        # Geometry
        ra: float = 4.0,  # Agent-agent repulsion radius
        rs: float = 65.0,  # Shepherd detection radius
        k_nn: int = 19,  # Nearest neighbors count
        # Timing & Speeds
        dt: float = 0.07,
        vmax: float = 1.0,  # Max agent speed
        umax: float = 1.5,  # Max controller speed
        # Weights
        wr: float = 50.0,  # Repulsion weight
        wa: float = 1.05,  # Attraction weight
        ws: float = 100.0,  # Shepherd repulsion weight
        wm: float = 20.0,  # Momentum/Inertia weight
        w_target: float = 0.0,  # Target attraction weight
        w_align: float = 0.0,  # Alignment weight
        # Grazing / Noise
        sigma: float = 0.3,  # Noise magnitude
        graze_p: float = 0.05,  # Grazing movement probability
        # Boundaries
        boundary: str = "none",
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        restitution: float = 0.85,
        # Obstacles
        obstacles_polygons: list[np.ndarray] | None = None,
        obstacle_influence: float = 15.0,
        w_obs: float = 10.0,
        w_tan: float = 12.0,
        keep_out: float = 8.0,
        world_keep_out: float = 8.0,
        wall_follow_boost: float = 6.0,
        stuck_speed_ratio: float = 0.08,
        near_wall_ratio: float = 0.8,
        # Initialization
        seed: int = 0,
        flock_init: float = 0.0,  # Initial flocking level (0=grazing, 1=flocking)
        # Optimization
        r_attr: float = 30.0,  # Attraction radius for local center of mass
        enforce_keepout: bool = True,
        use_neighbor_cache: bool | None = None,
        **_kw_ignore,
    ):
        self.N = sheep_xy.shape[0]

        # Initialize contiguous NumPy arrays for positions and velocities
        self.P = np.ascontiguousarray(sheep_xy, dtype=np.float64)  # shape (N, 2)
        self.V = np.zeros((self.N, 2), dtype=np.float64, order="C")  # shape (N, 2)

        # Internal storage for drones (controllers)
        # Note: 'drones' is used historically; aliased as 'shepherd_xy' via property
        self.drones = shepherd_xy

        # Initialize apply_repulsion array (all drones apply repulsion by default)
        self.apply_repulsion = np.ones(self.drones.shape[0], dtype=bool)

        self.target = np.asarray(target_xy, float) if target_xy is not None else None
        self.paused = False

        # Polygon obstacles
        self.polys: list[np.ndarray] = []
        self.poly_edges: list[dict] = []
        if obstacles_polygons is not None:
            for poly in obstacles_polygons:
                self.add_polygon(poly)

        # Simulation time
        self.t = 0.0

        # Parameters
        self.ra, self.rs, self.k_nn = ra, rs, k_nn
        self.dt, self.vmax, self.umax = dt, vmax, umax
        self.wr, self.wa, self.ws, self.wm, self.w_align, self.w_target = (
            wr,
            wa,
            ws,
            wm,
            w_align,
            w_target,
        )
        self.sigma = sigma

        # Cache squared distances for performance
        self.ra_sq = ra * ra
        self.rs_sq = rs * rs

        # Boundaries
        self.boundary = boundary
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.restitution = float(np.clip(restitution, 0.0, 1.0))

        # Obstacle parameters
        self.obstacle_influence = obstacle_influence
        self.w_obs = w_obs
        self.w_tan = w_tan
        self.keep_out = keep_out
        self.world_keep_out = world_keep_out
        self.wall_follow_boost = wall_follow_boost
        self.stuck_speed_ratio = stuck_speed_ratio
        self.near_wall_ratio = near_wall_ratio

        self.graze_p = graze_p

        # Hard enforcement flag
        self.enforce_keepout = enforce_keepout

        self.rng = np.random.default_rng(seed)

        # The -2 is because every sheep has n - 1 neighbors and k_nn is 0 indexed.
        assert self.k_nn <= self.N - 1

        # Sanitize initial positions if polygons exist
        if self.polys:
            self._sanitize_initial_positions()

        # Neighbor cache and movement tracking
        self.nb_idx = -np.ones((self.N, self.k_nn), dtype=np.int32)
        self.prev_P = self.P.copy()
        self._rr_cursor = 0
        self.eps_move = max(1e-6, 0.4 * self.ra)

        # Auto-enable neighbor cache for large flocks
        # Benchmark shows crossover at N=256 (+3.8%), significant gain at N=300 (+10%)
        if use_neighbor_cache is None:
            self.use_neighbor_cache = self.N >= 256
        else:
            self.use_neighbor_cache = use_neighbor_cache

        # Controls what fraction of the self.V velocity calculation for each sheep
        # should be attributed to flocking behavior vs grazing behavior.
        self.flock = np.full(
            self.N, float(np.clip(flock_init, 0.0, 1.0)), dtype=np.float64
        )

        self.r_attr = float(r_attr)
        self.r_attr_sq = self.r_attr * self.r_attr

    # -------------------------------------------------------------------------
    # Polygon Management
    # -------------------------------------------------------------------------

    def add_polygon(self, polygon: np.ndarray) -> None:
        """Add a polygon obstacle and precompute its edge data."""
        poly = np.asarray(polygon, float)
        if poly.ndim == 1:
            poly = poly.reshape(1, -1)

        # Ensure CCW winding
        # Signed area: 0.5 * sum(x1*y2 - x2*y1)
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

        # If area is negative (CW), reverse the order
        if area < 0:
            poly = poly[::-1]

        self.polys.append(poly)
        self.poly_edges.append(self._precompute_polygon_edges(poly))

    def add_polygons(self, polygons: list[np.ndarray]) -> None:
        """Add multiple polygon obstacles."""
        for poly in polygons:
            self.add_polygon(poly)

    def clear_polygons(self) -> None:
        """Remove all polygon obstacles."""
        self.polys = []
        self.poly_edges = []

    def get_polygons(self) -> list[np.ndarray]:
        """Get current polygon obstacles."""
        return [p.copy() for p in self.polys]

    # -------------------------------------------------------------------------
    # Domain-Neutral Accessors
    # -------------------------------------------------------------------------

    @property
    def agents_xy(self) -> np.ndarray:
        """Get agent positions (domain-neutral accessor)."""
        return self.sheep_xy

    @property
    def sheep_xy(self) -> np.ndarray:
        """Get sheep positions (herding-specific accessor)."""
        return self.P

    @property
    def controllers_xy(self) -> np.ndarray:
        """Get controller positions (domain-neutral accessor)."""
        return self.shepherd_xy

    @property
    def shepherd_xy(self) -> np.ndarray:
        """Get shepherd/drone positions (herding-specific accessor)."""
        return self.drones

    @property
    def num_agents(self) -> int:
        """Get number of agents."""
        return self.N

    @property
    def num_controllers(self) -> int:
        """Get number of controllers."""
        return self.drones.shape[0]

    def set_drone_count(self, count: int) -> None:
        """
        Dynamically set the number of drones/controllers.

        If increasing, new drones are spawned near existing drones.
        If decreasing, extra drones are removed.
        """
        current_count = self.drones.shape[0]
        if count == current_count:
            return

        if count < 1:
            count = 1  # Minimum 1 drone

        if count > current_count:
            # Add more drones - spawn them near existing drones
            new_drones = []
            for i in range(count - current_count):
                if current_count == 0:
                    ref_pos = np.array([0, 0])
                else:
                    # Pick a random existing drone to spawn near
                    ref_idx = i % current_count
                    ref_pos = self.drones[ref_idx]
                # Offset by small random amount
                offset = self.rng.uniform(-10, 10, size=2)
                new_pos = ref_pos + offset
                # Clamp to bounds
                new_pos[0] = np.clip(new_pos[0], self.xmin + 5, self.xmax - 5)
                new_pos[1] = np.clip(new_pos[1], self.ymin + 5, self.ymax - 5)
                new_drones.append(new_pos)
            self.drones = np.vstack([self.drones, new_drones])
            # Extend apply_repulsion array
            self.apply_repulsion = np.ones(self.drones.shape[0], dtype=bool)
        else:
            # Remove drones (keep the first 'count' drones)
            self.drones = self.drones[:count]
            self.apply_repulsion = self.apply_repulsion[:count]

    # -------------------------------------------------------------------------
    # Vectorized Geometry Kernels
    # -------------------------------------------------------------------------

    def _precompute_polygon_edges(self, poly: np.ndarray) -> dict:
        """Precompute edge vectors, normals, and lengths for a polygon."""
        V = poly
        E = np.roll(V, -1, axis=0) - V  # edge vectors
        L = np.sqrt(np.sum(E**2, axis=1))  # edge lengths
        # Unit normals pointing outward (rotate edge vector 90° CW: [x,y] -> [y,-x])
        N = np.column_stack([E[:, 1], -E[:, 0]])
        nonzero_mask = L > EPSILON
        N[nonzero_mask] /= L[nonzero_mask, None]

        return {"V": V, "E": E, "N": N, "L": L}

    def _point_in_poly_batch(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Vectorized ray casting for point-in-polygon test."""
        if NUMBA_AVAILABLE:
            return _point_in_poly_batch_numba(P, V)
        else:
            # Fallback NumPy implementation
            V1 = V
            V2 = np.roll(V, -1, axis=0)

            P_x = P[:, 0][:, np.newaxis]
            P_y = P[:, 1][:, np.newaxis]

            V1_x = V1[:, 0][np.newaxis, :]
            V1_y = V1[:, 1][np.newaxis, :]

            V2_x = V2[:, 0][np.newaxis, :]
            V2_y = V2[:, 1][np.newaxis, :]

            intersect_y = (V1_y > P_y) != (V2_y > P_y)

            denom = V2_y - V1_y
            denom = np.where(denom == 0, 1e-12, denom)

            x_inters = (V2_x - V1_x) * (P_y - V1_y) / denom + V1_x
            intersect_x = P_x < x_inters

            crossings = intersect_y & intersect_x
            inside = np.sum(crossings, axis=1) % 2 == 1

            return inside

    def _closest_point_on_polygon(
        self, P: np.ndarray, edges: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find closest points on polygon boundary for batch of points."""
        if NUMBA_AVAILABLE:
            return _closest_point_on_polygon_numba(
                P, edges["V"], edges["E"], edges["L"]
            )
        else:
            # Fallback NumPy implementation
            V, E, L = edges["V"], edges["E"], edges["L"]

            P_exp = P[:, np.newaxis, :]
            V_exp = V[np.newaxis, :, :]
            to_point = P_exp - V_exp

            E_exp = E[np.newaxis, :, :]
            L_sq = L**2
            L_sq = L_sq[np.newaxis, :]

            dot = np.sum(to_point * E_exp, axis=2)

            valid_edges = L_sq > 1e-18
            t = np.divide(dot, L_sq, out=np.zeros_like(dot), where=valid_edges)
            t = np.clip(t, 0.0, 1.0)

            closest_on_edges = V_exp + t[:, :, np.newaxis] * E_exp

            diff = P_exp - closest_on_edges
            dist_sq = np.sum(diff**2, axis=2)

            min_idx = np.argmin(dist_sq, axis=1)
            row_idx = np.arange(P.shape[0])

            Q = closest_on_edges[row_idx, min_idx]
            n = edges["N"][min_idx]
            s = np.sqrt(dist_sq[row_idx, min_idx])

            inside = self._point_in_poly_batch(P, edges["V"])
            s[inside] = -s[inside]

            return Q, n, s

    def _nearest_polygon(
        self, P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find nearest polygon for each point."""
        if not self.polys:
            return (
                np.zeros((P.shape[0], 2)),
                np.zeros((P.shape[0], 2)),
                np.full(P.shape[0], np.inf),
            )

        n_points = P.shape[0]
        best_Q = np.zeros((n_points, 2))
        best_n = np.zeros((n_points, 2))
        best_s = np.full(n_points, np.inf)

        for poly_edges in self.poly_edges:
            Q, n, s = self._closest_point_on_polygon(P, poly_edges)

            better_mask = np.abs(s) < np.abs(best_s)
            best_Q[better_mask] = Q[better_mask]
            best_n[better_mask] = n[better_mask]
            best_s[better_mask] = s[better_mask]

        return best_Q, best_n, best_s

    def _rect_signed_distance(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute signed distance to world rectangle boundary."""
        d_left = P[:, 0] - self.xmin
        d_right = self.xmax - P[:, 0]
        d_bottom = P[:, 1] - self.ymin
        d_top = self.ymax - P[:, 1]

        walls = np.column_stack([d_left, d_right, d_bottom, d_top])
        closest_wall = np.argmin(walls, axis=1)

        d_signed = np.min(walls, axis=1)
        n = np.zeros((P.shape[0], 2))

        n[closest_wall == 0] = [1, 0]  # left wall
        n[closest_wall == 1] = [-1, 0]  # right wall
        n[closest_wall == 2] = [0, 1]  # bottom wall
        n[closest_wall == 3] = [0, -1]  # top wall

        return d_signed, n

    def _obstacle_tangent_dir(self, n: np.ndarray) -> np.ndarray:
        """Rotate normals +90° to get tangent direction."""
        return np.column_stack([-n[:, 1], n[:, 0]])

    def _obstacle_avoid(
        self, P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute obstacle avoidance and tangential forces."""
        if not self.polys:
            return (
                np.zeros((P.shape[0], 2)),
                np.zeros((P.shape[0], 2)),
                np.full(P.shape[0], np.inf),
            )

        Q, n, s = self._nearest_polygon(P)

        # Avoidance weight: 1 at boundary, 0 at obstacle_influence distance
        w = np.clip(1 - np.abs(s) / self.obstacle_influence, 0, 1)
        avoid = w[:, None] * n

        # Tangent direction for wall following
        tan = self._obstacle_tangent_dir(n)

        return avoid, tan, s

    # -------------------------------------------------------------------------
    # Keep-Out & Anti-Pin
    # -------------------------------------------------------------------------

    def _sanitize_initial_positions(self):
        """Move agents outside polygon keep-out zones at initialization."""
        if not self.polys:
            return

        Q, n, s = self._nearest_polygon(self.P)
        inside_mask = s < -self.keep_out

        if np.any(inside_mask):
            self.P[inside_mask] = (
                Q[inside_mask] + (self.keep_out + 1e-3) * n[inside_mask]
            )

    def _resolve_polygon_penetration(self):
        """Project agents outward from polygon keep-out zones with capped correction."""
        if not self.polys:
            return np.zeros((self.N, 2)), np.zeros(self.N, dtype=bool)

        Q, n, s = self._nearest_polygon(self.P)
        penetration = self.keep_out - s
        mask = penetration > 0.0
        if not np.any(mask):
            return np.zeros((self.N, 2)), np.zeros(self.N, dtype=bool)

        max_corr_base = 0.25 * (self.vmax * self.dt)
        max_corr = np.where(
            penetration[mask] > 2 * max_corr_base, 2 * max_corr_base, max_corr_base
        )
        corr = np.minimum(penetration[mask], max_corr)

        n_unit = n[mask]
        Ln = np.sqrt(np.sum(n_unit * n_unit, axis=1)) + 1e-12
        n_unit = n_unit / Ln[:, None]

        correction_vec = corr[:, None] * n_unit
        self.P[mask] += correction_vec

        correction_velocity = correction_vec / (self.dt + 1e-12)

        v_into = np.sum(self.V[mask] * n_unit, axis=1)
        neg = v_into < 0.0

        if np.any(neg):
            self.V[mask][neg] -= (
                (1.0 + self.restitution) * v_into[neg, None] * n_unit[neg]
            )
            self.V[mask][neg] += 0.5 * correction_velocity[neg]

        pos_mask = ~neg
        if np.any(pos_mask):
            self.V[mask][pos_mask] += 0.3 * correction_velocity[pos_mask]

        correction_info = np.zeros((self.N, 2))
        correction_info[mask] = correction_vec
        return correction_info, mask

    def _resolve_world_keepout(self):
        """Push agents away from world walls with capped correction."""
        d_wall, n_wall = self._rect_signed_distance(self.P)
        penetration = self.world_keep_out - d_wall
        mask = penetration > 0.0
        if not np.any(mask):
            return np.zeros((self.N, 2)), np.zeros(self.N, dtype=bool)

        max_corr_base = 0.25 * (self.vmax * self.dt)
        max_corr = np.where(
            penetration[mask] > 2 * max_corr_base, 2 * max_corr_base, max_corr_base
        )
        corr = np.minimum(penetration[mask], max_corr)

        correction_vec = corr[:, None] * n_wall[mask]
        self.P[mask] += correction_vec

        correction_velocity = correction_vec / (self.dt + 1e-12)

        v_into = np.sum(self.V[mask] * n_wall[mask], axis=1)
        neg = v_into < 0.0

        if np.any(neg):
            self.V[mask][neg] -= (
                (1.0 + self.restitution) * v_into[neg, None] * n_wall[mask][neg]
            )
            self.V[mask][neg] += 0.5 * correction_velocity[neg]

        pos_mask = ~neg
        if np.any(pos_mask):
            self.V[mask][pos_mask] += 0.3 * correction_velocity[pos_mask]

        correction_info = np.zeros((self.N, 2))
        correction_info[mask] = correction_vec
        return correction_info, mask

    def enforce_keepout_all(self):
        """Enforce all keep-out constraints with conflict resolution."""
        poly_corr, poly_mask = self._resolve_polygon_penetration()

        if self.boundary == "none":
            wall_corr = np.zeros((self.N, 2))
            wall_mask = np.zeros(self.N, dtype=bool)
        else:
            wall_corr, wall_mask = self._resolve_world_keepout()

        conflict_mask = poly_mask & wall_mask
        if np.any(conflict_mask):
            poly_dir = poly_corr[conflict_mask]
            wall_dir = wall_corr[conflict_mask]

            poly_norm = np.linalg.norm(poly_dir, axis=1, keepdims=True) + 1e-12
            wall_norm = np.linalg.norm(wall_dir, axis=1, keepdims=True) + 1e-12

            poly_dir_norm = poly_dir / poly_norm
            wall_dir_norm = wall_dir / wall_norm

            dot_products = np.sum(poly_dir_norm * wall_dir_norm, axis=1)
            opposite = dot_products < -0.5

            if np.any(opposite):
                conflict_indices = np.where(conflict_mask)[0]
                opposite_indices = conflict_indices[opposite]

                poly_mag = np.linalg.norm(poly_corr[opposite_indices], axis=1)
                wall_mag = np.linalg.norm(wall_corr[opposite_indices], axis=1)

                larger_is_poly = poly_mag > wall_mag

                for i, idx in enumerate(opposite_indices):
                    if larger_is_poly[i]:
                        reduction = 0.5 * wall_corr[idx]
                        self.P[idx] -= reduction
                        self.V[idx] -= 0.3 * reduction / (self.dt + 1e-12)
                    else:
                        reduction = 0.5 * poly_corr[idx]
                        self.P[idx] -= reduction
                        self.V[idx] -= 0.3 * reduction / (self.dt + 1e-12)

    # -------------------------------------------------------------------------
    # Neighbor Operations
    # -------------------------------------------------------------------------

    def _kNN_vec(self, i: int, K: int) -> np.ndarray:
        """Vectorized k-nearest neighbors using contiguous arrays."""
        if NUMBA_AVAILABLE:
            return _kNN_numba(self.P, i, K)
        else:
            d2 = np.sum((self.P - self.P[i]) ** 2, axis=1)
            idx = np.argpartition(d2, K + 1)[: K + 1]
            idx = idx[d2[idx] > 0]
            return idx[:K]

    def _repel_close_vec(self, i: int) -> np.ndarray:
        """Vectorized repulsion from close neighbors using contiguous arrays."""
        if NUMBA_AVAILABLE:
            return _repel_close_numba(self.P, i, self.ra)
        else:
            d_sq = np.sum((self.P - self.P[i]) ** 2, axis=1)
            mask = (d_sq > 1e-18) & (d_sq < self.ra_sq)
            if not np.any(mask):
                return np.zeros(2)
            d_vec = self.P[mask] - self.P[i]
            d = np.sqrt(np.sum(d_vec**2, axis=1)) + 1e-9
            inv_d = 1.0 / d
            vecs = -(d_vec * inv_d[:, None])
            return vecs.sum(axis=0)

    def _neighbors_within(
        self, i: int, r_sq: float, max_k: int | None = None
    ) -> np.ndarray:
        """
        Return indices of neighbors within distance^2 <= r_sq (excluding i).
        If max_k is set, cap to that many nearest by distance.
        """
        if self.use_neighbor_cache:
            idx = self.nb_idx[i]
            k = np.count_nonzero(idx >= 0)
            if k == 0:
                # fall back to full scan
                P_i = self.P[i]
                d2 = np.sum((self.P - P_i) ** 2, axis=1)
                mask = (d2 > 0) & (d2 <= r_sq)
                cand = np.where(mask)[0]
                if cand.size == 0:
                    return cand
                if max_k is not None and cand.size > max_k:
                    order = np.argpartition(d2[cand], max_k - 1)[:max_k]
                    return cand[order]
                return cand
            else:
                cand = idx[:k]
                d2 = np.sum((self.P[cand] - self.P[i]) ** 2, axis=1)
                keep = d2 <= r_sq
                cand = cand[keep]
                if cand.size == 0:
                    return cand
                if max_k is not None and cand.size > max_k:
                    order = np.argpartition(d2[keep], max_k - 1)[:max_k]
                    cand = cand[order]
                return cand
        else:
            P_i = self.P[i]
            d2 = np.sum((self.P - P_i) ** 2, axis=1)
            mask = (d2 > 0) & (d2 <= r_sq)
            cand = np.where(mask)[0]
            if cand.size == 0:
                return cand
            if max_k is not None and cand.size > max_k:
                order = np.argpartition(d2[cand], max_k - 1)[:max_k]
                return cand[order]
            return cand

    def _lcm_vec(self, i: int) -> np.ndarray:
        """Local center of mass using only neighbors within r_attr."""
        idx = self._neighbors_within(i, self.r_attr_sq, self.k_nn)
        if idx.size == 0:
            return self.P[i].copy()
        return np.mean(self.P[idx], axis=0)

    def _align_vec(self, i: int) -> np.ndarray:
        """Alignment using neighbors within r_attr."""
        idx = self._neighbors_within(i, self.r_attr_sq, self.k_nn)
        if idx.size == 0:
            return np.zeros(2)
        vbar = np.mean(self.V[idx], axis=0)
        n = np.sqrt(np.sum(vbar * vbar))
        if n == 0.0:
            return np.zeros(2)
        return vbar / (n + 1e-9)

    # -------------------------------------------------------------------------
    # Boundaries
    # -------------------------------------------------------------------------

    def _apply_bounds_sheep_inplace(self):
        """In-place boundary application for all sheep positions and velocities."""
        if self.boundary == "none":
            return

        P, V = self.P, self.V

        if self.boundary == "wrap":
            P_before = P.copy()
            Lx, Ly = self.xmax - self.xmin, self.ymax - self.ymin
            P[:, 0] = self.xmin + ((P[:, 0] - self.xmin) % Lx)
            P[:, 1] = self.ymin + ((P[:, 1] - self.ymin) % Ly)
            displacement = P - P_before
            V += displacement / (self.dt + 1e-12)
            return

        # Reflection boundaries
        P_before = P.copy()

        m = P[:, 0] < self.xmin
        if np.any(m):
            P[m, 0] = self.xmin + (self.xmin - P[m, 0])
            V[m, 0] = np.abs(V[m, 0]) * self.restitution

        m = P[:, 0] > self.xmax
        if np.any(m):
            P[m, 0] = self.xmax - (P[m, 0] - self.xmax)
            V[m, 0] = -np.abs(V[m, 0]) * self.restitution

        m = P[:, 1] < self.ymin
        if np.any(m):
            P[m, 1] = self.ymin + (self.ymin - P[m, 1])
            V[m, 1] = np.abs(V[m, 1]) * self.restitution

        m = P[:, 1] > self.ymax
        if np.any(m):
            P[m, 1] = self.ymax - (P[m, 1] - self.ymax)
            V[m, 1] = -np.abs(V[m, 1]) * self.restitution

        displacement = P - P_before
        correction_velocity = displacement / (self.dt + 1e-12)
        moved = np.any(np.abs(displacement) > 1e-9, axis=1)
        if np.any(moved):
            V[moved] = 0.7 * V[moved] + 0.3 * correction_velocity[moved]

    def _apply_bounds_point(self, pos: np.ndarray) -> np.ndarray:
        """Apply boundaries to a list of positions (used for drone positions)."""
        if self.boundary == "none":
            return pos.copy()

        x, y = pos[:, 0], pos[:, 1]
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

        if self.boundary == "wrap":
            Lx, Ly = (xmax - xmin), (ymax - ymin)
            x = xmin + ((x - xmin) % Lx)
            y = ymin + ((y - ymin) % Ly)

        else:  # reflect
            x[x < xmin] = xmin + (xmin - x[x < xmin])
            x[x > xmax] = xmax - (x[x > xmax] - xmax)
            y[y < ymin] = ymin + (ymin - y[y < ymin])
            y[y > ymax] = ymax - (y[y > ymax] - ymax)

        return np.column_stack([x, y])

    # -------------------------------------------------------------------------
    # Sheep Step Logic
    # -------------------------------------------------------------------------

    def _gcm_vec(self) -> np.ndarray:
        """Vectorized global center of mass calculation."""
        return np.mean(self.P, axis=0)

    def _should_ignore_drone_repulsion(
        self, near_indices: np.ndarray, G: np.ndarray, tol: float = 0.0
    ) -> bool:
        """
        Return True if the summed local intent (wr*R + wa*A) of sheep near the drone
        points radially outward from the global COM G.
        """
        radial_sum = 0.0
        for i in near_indices:
            R = self._repel_close_vec(i)
            A = self._lcm_vec(i) - self.P[i]
            v_local = self.wr * R + self.wa * A

            g = self.P[i] - G
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-9:
                u_out = g / g_norm
                radial_sum += np.dot(v_local, u_out)

        return radial_sum > tol

    def _sheep_step(self):
        """Optimized sheep step using vectorized operations where possible."""
        G = self._gcm_vec()

        # Vectorized: compute all sheep-to-drone squared distances at once
        diff = self.P[:, None, :] - self.drones[None, :, :]
        drone_distances_sq = np.sum(diff**2, axis=2)

        # If a given drone isn't applying repulsion, make it seem far away
        drone_distances_sq[:, self.apply_repulsion == 0] = LARGE_DISTANCE

        # Continuous flock factor update
        d_all = np.sqrt(np.maximum(drone_distances_sq, 0.0))
        push_all = smooth_push(d_all, self.rs)

        # Combine multiple drones as union probability of "being pushed"
        repel_level = 1.0 - np.prod(1.0 - push_all, axis=1)

        # EMA smooth with previous value
        delta = repel_level - self.flock
        # Faster rate when entering flocking behavior than leaving
        rate = np.where(delta > 0, 1 / 2, 1 / 60)
        change_in_flocking = rate * delta * self.dt
        self.flock += change_in_flocking
        self.flock = np.clip(self.flock, 0.0, 1.0)

        v_far = self._handle_far_sheep(G)
        v_near = self._handle_near_sheep(G, drone_distances_sq)

        # Blend near and far behaviors based on flocking factor
        v_new = (
            v_near * self.flock[:, np.newaxis]
            + (1.0 - self.flock[:, np.newaxis]) * v_far
        )
        self.P += v_new * self.dt
        self.V = v_new

        # Hard enforcement: keep-out zones and boundaries
        if self.enforce_keepout:
            self.enforce_keepout_all()
            self._apply_bounds_sheep_inplace()

        # Safety check for NaNs
        bad = ~np.isfinite(self.P).all(axis=1)
        if np.any(bad):
            cx, cy = 0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax)
            self.P[bad] = np.array([cx, cy])
            self.V[bad] = 0.0

    def _handle_far_sheep(self, G: np.ndarray) -> np.ndarray:
        """Handle sheep that are far from the drone (grazing behavior)."""
        decay = 0.80

        V_new = np.zeros((self.N, 2))
        for i in range(self.N):
            rnd = self.rng.normal(size=2) * 0.2
            R = self._repel_close_vec(i)
            H = self.wr * R + rnd

            h = norm(H)

            # Obstacle handling for far sheep
            if self.polys:
                avoid_far, tan_far, s_far = self._obstacle_avoid(self.P[i : i + 1])
                nrm_f = avoid_far[0]
                tng_f = tan_far[0]
            else:
                nrm_f = np.zeros(2)
                tng_f = np.zeros(2)
                s_far = np.array([np.inf])

            H = h.copy()

            if np.dot(H, nrm_f) < 0.0:
                H += self.w_tan * tng_f

            H += (0.5 * self.w_obs) * nrm_f

            if s_far[0] <= self.keep_out:
                n_unit = nrm_f
                L = np.sqrt(np.dot(n_unit, n_unit)) + 1e-12
                n_unit = n_unit / L
                into = np.dot(H, n_unit)
                if into < 0.0:
                    H = H - into * n_unit

            Hn = np.linalg.norm(H)
            if Hn > 0:
                h = H / Hn

            v_des = self.vmax * h
            v_new = decay * self.V[i] + (1.0 - decay) * v_des
            sp = np.linalg.norm(v_new)
            if sp > self.vmax:
                v_new *= self.vmax / sp

            V_new[i] = v_new

        return V_new

    def _handle_near_sheep(
        self, G: np.ndarray, drone_distances_sq: np.ndarray
    ) -> np.ndarray:
        """Handle sheep that are near a drone (flocking behavior)."""

        if self.polys:
            avoid, tan, s = self._obstacle_avoid(self.P)
        else:
            avoid = np.zeros((self.N, 2))
            tan = np.zeros((self.N, 2))
            s = np.full(self.N, np.inf)

        nrm = avoid
        tng = tan

        # Drone Repulsion: SUM over all drones
        S_total = np.zeros((self.N, 2))

        for j in range(self.drones.shape[0]):
            D = self.drones[j]
            d_sq = drone_distances_sq[:, j]
            d = np.sqrt(d_sq)

            push = np.asarray(smooth_push(d, self.rs))
            inv_d = 1.0 / d

            vec_DP = self.P - D
            S_j = push[:, None] * vec_DP * inv_d[:, None]
            S_total += S_j

        V_new = np.zeros((self.N, 2))
        for i in range(self.N):

            R = self._repel_close_vec(i)
            A = self._lcm_vec(i) - self.P[i]
            S = S_total[i]
            AL = self._align_vec(i)

            vel_sq = np.sum(self.V[i] ** 2)
            if vel_sq > 0:
                vel_norm = np.sqrt(vel_sq)
                inv_vel_norm = 1.0 / (vel_norm + 1e-9)
                prev = self.V[i] * inv_vel_norm
            else:
                vel_norm = 0.0
                prev = np.zeros(2)

            H = (
                self.wr * R
                + self.wa * A
                + self.ws * S
                + self.wm * prev
                + self.w_align * AL
            )

            if self.w_target > 0.0 and self.target is not None:
                T_vec = self.target - self.P[i]
                dist_t = np.linalg.norm(T_vec)
                if dist_t > 0:
                    T = T_vec / dist_t
                    H += self.w_target * T

            H += self.w_obs * nrm[i]
            if np.dot(tng[i], tng[i]) > 0.0 and np.dot(H, nrm[i]) < 0.0:
                H += self.w_tan * tng[i]

            if s[i] <= self.keep_out:
                n_unit = nrm[i]
                L = np.sqrt(np.dot(n_unit, n_unit)) + 1e-12
                n_unit = n_unit / L
                into = np.dot(H, n_unit)
                if into < 0.0:
                    H = H - into * n_unit

            noise = self.sigma * np.sqrt(self.dt) * self.rng.normal(size=2)
            if vel_norm > 0.3 * self.vmax:
                noise *= 0.5
            H = H + noise

            h_sq = np.sum(H**2)
            if h_sq > 0:
                h_norm = np.sqrt(h_sq)
                inv_h_norm = 1.0 / (h_norm + 1e-9)
                h = H * inv_h_norm
            else:
                h = np.zeros(2)

            v_des = h * self.vmax

            smoothing = 0.8
            V_new[i] = smoothing * self.V[i] + (1.0 - smoothing) * v_des

        return V_new

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def step(self, plan: Plan):
        """Advance the simulation by one time step."""
        if self.paused:
            return

        if self.use_neighbor_cache:
            self._refresh_neighbors()

        # Apply planner output
        if isinstance(plan, DoNothing):
            self.apply_repulsion = np.zeros(self.drones.shape[0])
        elif isinstance(plan, DronePositions):
            pos = plan.positions
            apply = plan.apply_repulsion

            drone_count = self.drones.shape[0]
            if pos.shape[0] != drone_count or apply.size != drone_count:
                raise ValueError(
                    f"DronePositions plan must have {drone_count} positions and repulsion flags."
                )

            # Apply bounds to all drone positions
            new_drones_pos = self._apply_bounds_point(pos)
            self.drones = new_drones_pos
            self.apply_repulsion = apply.copy()
        else:
            raise Exception("Unexpected plan type", plan)

        # Then move sheep using the new drone pos + flag
        self._sheep_step()

        # Advance time
        self.t += self.dt

    def get_state(self) -> state.State:
        """Get the current simulation state."""
        return state.State(
            flock=self.P.copy(),
            drones=self.drones.copy(),
            polygons=[p.copy() for p in self.polys],
            jobs=[],
        )

    def pause(self):
        """Toggle simulation pause state."""
        self.paused = not self.paused

    def _refresh_neighbors(self):
        """Update neighbor cache if agents have moved significantly."""
        if not self.use_neighbor_cache:
            return
        moved = np.sqrt(np.sum((self.P - self.prev_P) ** 2, axis=1)) > self.eps_move
        batch = np.zeros(self.N, dtype=bool)
        half = max(1, self.N // 12)
        # If very few moved, shrink batch further
        if np.count_nonzero(moved) < max(2, self.N // 20):
            half = max(1, self.N // 16)
        start = self._rr_cursor
        end = min(self._rr_cursor + half, self.N)
        batch[start:end] = True
        self._rr_cursor = 0 if end >= self.N else end
        need = moved | batch
        if not np.any(need):
            return
        idxs = np.where(need)[0]
        # Use fast NumPy argpartition path for kNN refresh to minimize overhead
        for i in idxs:
            self.nb_idx[i, : self.k_nn] = self._kNN_vec(i, self.k_nn)
        self.prev_P[need] = self.P[need]


# -----------------------------------------------------------------------------
# Numba Optimized Functions
# -----------------------------------------------------------------------------


@njit
def _point_in_poly_batch_numba(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Numba-optimized batch point-in-polygon test using ray casting."""
    n_points = P.shape[0]
    n_vertices = V.shape[0]
    inside = np.zeros(n_points, dtype=np.bool_)

    for i in range(n_points):
        px, py = P[i, 0], P[i, 1]
        j = n_vertices - 1

        for k in range(n_vertices):
            xi, yi = V[k, 0], V[k, 1]
            xj, yj = V[j, 0], V[j, 1]

            denom = yj - yi
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (denom + 1e-12) + xi
            ):
                inside[i] = not inside[i]
            j = k

    return inside


@njit
def _closest_point_on_polygon_numba(
    P: np.ndarray, V: np.ndarray, E: np.ndarray, L: np.ndarray
) -> tuple:
    """Numba-optimized closest point calculation on polygon boundary."""
    n_points = P.shape[0]
    n_edges = V.shape[0]

    Q = np.zeros((n_points, 2), dtype=np.float64)
    n = np.zeros((n_points, 2), dtype=np.float64)
    s = np.zeros(n_points, dtype=np.float64)

    # Precompute normals
    N = np.zeros((n_edges, 2), dtype=np.float64)
    for j in range(n_edges):
        if L[j] > EPSILON:
            N[j, 0] = E[j, 1] / L[j]
            N[j, 1] = -E[j, 0] / L[j]

    for i in range(n_points):
        min_dist_sq = np.inf
        closest_point = np.array([P[i, 0], P[i, 1]])
        closest_normal = np.array([0.0, 0.0])

        for j in range(n_edges):
            v0 = V[j]
            v1 = V[(j + 1) % n_edges]
            edge_vec = v1 - v0
            to_point = P[i] - v0

            if L[j] > EPSILON:
                t = max(0.0, min(1.0, np.dot(to_point, edge_vec) / (L[j] * L[j])))
            else:
                t = 0.0

            closest_on_edge = v0 + t * edge_vec
            diff = P[i] - closest_on_edge
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1]

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = closest_on_edge
                closest_normal = N[j]

        Q[i] = closest_point
        n[i] = closest_normal
        s[i] = np.sqrt(min_dist_sq)

    # Apply sign based on inside/outside
    inside = _point_in_poly_batch_numba(P, V)
    for i in range(n_points):
        if inside[i]:
            s[i] = -s[i]

    return Q, n, s


@njit
def _kNN_numba(P: np.ndarray, i: int, K: int) -> np.ndarray:
    """Numba-optimized k-nearest neighbors using argsort."""
    N = P.shape[0]
    distances = np.empty(N, dtype=np.float64)

    for j in range(N):
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        distances[j] = dx * dx + dy * dy

    distances[i] = np.inf
    sorted_indices = np.argsort(distances)
    return sorted_indices[:K]


@njit
def _repel_close_numba(P: np.ndarray, i: int, ra: float) -> np.ndarray:
    """Numba-optimized repulsion calculation using squared distances."""
    repulsion = np.zeros(2, dtype=np.float64)
    pos_i = P[i]
    ra_sq = ra * ra

    for j in range(P.shape[0]):
        if i == j:
            continue

        dx = pos_i[0] - P[j, 0]
        dy = pos_i[1] - P[j, 1]
        d_sq = dx * dx + dy * dy

        if d_sq > 1e-18 and d_sq < ra_sq:
            d = np.sqrt(d_sq)
            inv_d = 1.0 / (d + 1e-9)
            repulsion[0] += dx * inv_d
            repulsion[1] += dy * inv_d

    return repulsion
