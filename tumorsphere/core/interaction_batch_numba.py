"""Fixed-N, order-preserving, cached and vectorized Grosmann interactions.

Implements the backend="numba" interaction sweep, with or without
translational/rotational noise. Pair-force arithmetic is unchanged by cleanup.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Saturation used by the NumPy reference for core amplification.
XI_POWER_MAX = float(np.nextafter(1.0, 0.0))


@njit(cache=True, fastmath=False)
def _batch_pair_forces(
    positions,
    phies,
    diagonals,
    anisotropies,
    qxx,
    qxy,
    ptr,
    indices,
    side,
    overlap_threshold,
    k_rep,
    gamma,
    lambda_core,
    xi_power_max,
):
    """Accumulate ordered pair forces and torques in a compiled sweep.

    ``ptr`` and ``indices`` encode candidate neighbors in CSR form.
    The first stage computes their normalized overlaps using the
    reference minimum-image rule; the second stage accumulates force
    and torque only where the overlap strictly exceeds the threshold.
    The supplied candidate order determines floating-point summation
    order. The routine does not generate noise or mutate cell state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Total pair forces ``(N, 3)``, torques ``(N,)`` and normalized
        overlaps for all candidate entries, in CSR order.
    """
    n = len(ptr) - 1
    f = np.zeros((n, 3), dtype=np.float64)
    t = np.zeros(n, dtype=np.float64)
    xis = np.empty(len(indices), dtype=np.float64)
    rel = np.empty((len(indices), 3), dtype=np.float64)
    half_side = 0.5 * side
    for i in range(n):
        di = diagonals[i]
        ei = anisotropies[i]
        pi = phies[i]
        qix = qxx[i]
        qiy = qxy[i]
        wi = di * ei
        # Stage 1: exact ordered canonical overlap calculations.
        for k in range(ptr[i], ptr[i + 1]):
            j = indices[k]
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            if abs(dx) > half_side:
                if dx > 0.0:
                    dx -= side
                elif dx < 0.0:
                    dx += side
            if abs(dy) > half_side:
                if dy > 0.0:
                    dy -= side
                elif dy < 0.0:
                    dy += side
            rel[k, 0] = dx
            rel[k, 1] = dy
            rel[k, 2] = dz
            if j >= i:
                d0 = di
                d1 = diagonals[j]
                e0 = ei
                e1 = anisotropies[j]
                p0 = pi
                p1 = phies[j]
                x0 = qix
                y0 = qiy
                x1 = qxx[j]
                y1 = qxy[j]
                rx = dx
                ry = dy
                rz = dz
            else:
                d0 = diagonals[j]
                d1 = di
                e0 = anisotropies[j]
                e1 = ei
                p0 = phies[j]
                p1 = pi
                x0 = qxx[j]
                y0 = qxy[j]
                x1 = qix
                y1 = qiy
                rx = -dx
                ry = -dy
                # Legacy canonical-overlap backend does not flip z. Planar z=0.
                rz = dz
            ds = d0 + d1
            delta = p0 - p1
            cos_delta = np.cos(delta)
            beta = (
                ds**2
                - (d0 * e0 - d1 * e1) ** 2
                - 4.0 * d0 * d1 * e0 * e1 * (cos_delta**2)
            )
            w0 = d0 * e0
            w1 = d1 * e1
            mxx = (w0 * x0 + w1 * x1) / ds
            mxy = (w0 * y0 + w1 * y1) / ds
            quad = (
                (1.0 - mxx) * rx**2
                + (1.0 + mxx) * ry**2
                - 2.0 * mxy * rx * ry
                + rz**2
            )
            norm = np.exp(-(ds / beta) * quad)
            xis[k] = norm
        # Stage 2: exact force and torque accumulation in reference neighbor order.
        for k in range(ptr[i], ptr[i + 1]):
            if not xis[k] > overlap_threshold:
                continue
            j = indices[k]
            dj = diagonals[j]
            ej = anisotropies[j]
            wj = dj * ej
            ds = di + dj
            relative_angle = pi - phies[j]
            cos_rel = np.cos(relative_angle)
            mxx = (wi * qix + wj * qxx[j]) / ds
            mxy = (wi * qiy + wj * qxy[j]) / ds
            beta = (
                ds**2 - (wi - wj) ** 2 - 4.0 * di * ei * dj * ej * cos_rel**2
            )
            xi_power = xis[k] ** gamma
            if lambda_core == 0.0:
                amplification = 1.0
            else:
                safe = xi_power
                if safe < 0.0:
                    safe = 0.0
                elif safe > xi_power_max:
                    safe = xi_power_max
                amplification = 1.0 + lambda_core * safe / (1.0 - safe)
            kernel = (
                2.0 * k_rep * gamma * xi_power * (ds / beta) * amplification
            )
            rx = rel[k, 0]
            ry = rel[k, 1]
            rz = rel[k, 2]
            cx = (1.0 - mxx) * rx - mxy * ry
            cy = -mxy * rx + (1.0 + mxx) * ry
            cz = rz
            f[i, 0] += kernel * cx
            f[i, 1] += kernel * cy
            f[i, 2] += kernel * cz
            theta = np.arctan2(ry, rx)
            r_dot_c = rx * cx + ry * cy + rz * cz
            r_squared = rx * rx + ry * ry + rz * rz
            t[i] += kernel * (
                (
                    (2.0 * di * ei * dj * ej * np.sin(-2.0 * relative_angle))
                    / beta
                )
                * r_dot_c
                + (di * ei / ds) * r_squared * np.sin(2.0 * (pi - theta))
            )
    return f, t, xis


def _vectorized_compiled_noise_postprocess(
    self, delta_t, f, t, mobilities, translation, rotation, xis
):
    """Vectorized fixed-N postprocessing of the compiled-noise sweep.

    The ordered force/torque kernel and NumPy RNG generation are unchanged.
    Array arithmetic is algebraically equivalent to the scalar cell loop;
    it is not promised to be bitwise identical on every NumPy/BLAS version.
    """
    n = len(self.cells)
    if getattr(self.output, "collect_overlap_statistics", True) and xis.size:
        self.max_normalized_overlap_interval = max(
            self.max_normalized_overlap_interval, float(np.max(xis))
        )
        self.normalized_overlap_sum_interval += float(np.sum(xis))
        self.normalized_overlap_count_interval += int(xis.size)

    mP = mobilities[:, 0]
    mS = mobilities[:, 1]
    mR = mobilities[:, 2]
    a = (mP + mS) / 2
    b = (mP - mS) / 2
    matrix = (
        a[:, None, None] * np.eye(3) + b[:, None, None] * self.nematic_tensors
    )
    dif_velocity = np.matmul(matrix, f[:, :, None])[:, :, 0]

    aspect_ratios = np.fromiter(
        (cell.aspect_ratio for cell in self.cells),
        dtype=np.float64,
        count=n,
    )
    intrinsic_speeds = (
        self.cell_speed_max * (aspect_ratios - 1) / (self.aspect_ratio_max - 1)
    )
    direction = np.empty((n, 3), dtype=np.float64)
    direction[:, 0] = np.cos(self.cell_phies)
    direction[:, 1] = np.sin(self.cell_phies)
    direction[:, 2] = 0.0
    intrinsic_velocity = intrinsic_speeds[:, None] * direction

    dif_positions = (intrinsic_velocity + dif_velocity) * delta_t
    dif_phies = mR * t * delta_t
    if self.force.translational_noise_enabled:
        dif_positions += translation
    if self.force.rotational_noise_enabled:
        dif_phies += rotation

    if self.force.shrinking is True:
        speed_norms = np.linalg.norm(intrinsic_velocity, axis=1)
        projections = np.einsum("ij,ij->i", dif_velocity, intrinsic_velocity)
        # The scalar reference assigns projection=0 for zero-speed cells.
        # Avoid dividing by zero, including for round cells.
        np.divide(
            projections, speed_norms, out=projections, where=speed_norms > 0
        )
        projections[speed_norms == 0] = 0.0
        not_round = np.fromiter(
            (not cell.is_round for cell in self.cells),
            dtype=bool,
            count=n,
        )
        shrinking_flags = not_round & (speed_norms + projections <= 0)
        for i, cell in enumerate(self.cells):
            cell.shrink = bool(shrinking_flags[i])
    return dif_positions, dif_phies


def batch_interaction_step(self, delta_t):
    """Return all step displacements, preserving the original per-cell RNG order."""
    from tumorsphere.core.forces import Anisotropic_Grosmann

    if (
        not isinstance(self.force, Anisotropic_Grosmann)
        or self.backend != "numba"
        or not self.movement
        or self.reproduction
    ):
        raise RuntimeError(
            "batch_cached_noise_numba requires backend=numba and fixed-N Grosmann"
        )
    if list(self.active_cell_indexes) != list(range(len(self.cells))):
        raise RuntimeError("Requires cells active in ascending index order")
    # The validated canonical pair semantics assume planar cells.
    if np.any(self.cell_positions[:, 2] != 0.0):
        raise RuntimeError(
            "batch_cached_noise_numba requires planar positions (z=0)"
        )
    n = len(self.cells)
    from tumorsphere.core.neighbor_candidate_cache import OrderedNeighborCache

    cache = getattr(self.grid, "_batch_neighbor_cache", None)
    if cache is None:
        cache = OrderedNeighborCache()
        self.grid._batch_neighbor_cache = cache
    ptr, neighbors = cache.ordered_csr(self.grid, self.cell_positions)
    geometry_d = np.asarray(
        [c.squared_diagonal for c in self.cells], dtype=np.float64
    )
    geometry_e = np.asarray(
        [c.anisotropy for c in self.cells], dtype=np.float64
    )
    qxx = np.ascontiguousarray(self.nematic_tensors[:, 0, 0])
    qxy = np.ascontiguousarray(self.nematic_tensors[:, 0, 1])
    f, t, xis = _batch_pair_forces(
        self.cell_positions,
        self.cell_phies,
        geometry_d,
        geometry_e,
        qxx,
        qxy,
        np.asarray(ptr, dtype=np.intp),
        np.asarray(neighbors, dtype=np.intp),
        float(self.side),
        float(self.overlap_threshold_ratio),
        float(self.force.kRep),
        float(self.force.bExp),
        float(self.force.lambda_core),
        XI_POWER_MAX,
    )
    # Vectorized postprocessing is used both with and without noise.
    # Zero-noise sweeps do not consume any random numbers.
    mobilities = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        mobilities[i] = self.force.calculate_mobilities(
            self.cells[i],
            self.cell_area,
        )
    if (
        self.force.translational_noise_enabled
        or self.force.rotational_noise_enabled
    ):
        from tumorsphere.core.interaction_noise_numba import (
            compiled_noise_for_sweep,
        )

        translation, rotation = compiled_noise_for_sweep(
            self,
            mobilities,
            delta_t,
        )
    else:
        translation = rotation = None
    return _vectorized_compiled_noise_postprocess(
        self,
        delta_t,
        f,
        t,
        mobilities,
        translation,
        rotation,
        xis,
    )
