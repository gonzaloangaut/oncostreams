"""Optional fixed-N batch interaction backend for Grosmann dynamics.

Design constraints:
* does not alter Culture state except the same physical updates as direct_numba;
* identical neighbor order to direct_numba, including sorted lower-index neighbors;
* identical scalar algebra in overlap_canonical/grosmann_force_torque_kernel;
* same NumPy post-force mobility, noise and shrink updates in ascending cell order;
* no fastmath, parallel reductions or changes in RNG draws.

This backend is opt-in. Use interaction_backend="batch_numba" in Culture or Simulation.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _batch_pair_forces(positions, phies, diagonals, anisotropies, qxx, qxy,
                       ptr, indices, side, area, overlap_threshold,
                       k_rep, gamma, lambda_core, xi_power_max):
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
            beta = (ds ** 2 - (d0 * e0 - d1 * e1) ** 2
                    - 4.0 * d0 * d1 * e0 * e1 * (cos_delta ** 2))
            w0 = d0 * e0
            w1 = d1 * e1
            mxx = (w0 * x0 + w1 * x1) / ds
            mxy = (w0 * y0 + w1 * y1) / ds
            quad = ((1.0 - mxx) * rx ** 2 + (1.0 + mxx) * ry ** 2
                    - 2.0 * mxy * rx * ry + rz ** 2)
            imax = 4.0 * area ** 2 / (np.pi * np.sqrt(beta))
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
            beta = (ds ** 2 - (wi - wj) ** 2
                    - 4.0 * di * ei * dj * ej * cos_rel ** 2)
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
            kernel = (2.0 * k_rep * gamma * xi_power * (ds / beta)
                      * amplification)
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
                ((2.0 * di * ei * dj * ej * np.sin(-2.0 * relative_angle))
                 / beta) * r_dot_c
                + (di * ei / ds) * r_squared
                * np.sin(2.0 * (pi - theta))
            )
    return f, t, xis


def _vectorized_compiled_noise_postprocess(self, delta_t, f, t, mobilities,
                                           translation, rotation, xis):
    """Vectorized fixed-N postprocessing of the compiled-noise sweep.

    The ordered force/torque kernel and NumPy RNG generation are unchanged.
    Array arithmetic is algebraically equivalent to the scalar cell loop;
    it is not promised to be bitwise identical on every NumPy/BLAS version.
    """
    n = len(self.cells)
    if getattr(self.output, 'collect_overlap_statistics', True) and xis.size:
        self.max_normalized_overlap_interval = max(
            self.max_normalized_overlap_interval, float(np.max(xis)))
        self.normalized_overlap_sum_interval += float(np.sum(xis))
        self.normalized_overlap_count_interval += int(xis.size)

    mP = mobilities[:, 0]
    mS = mobilities[:, 1]
    mR = mobilities[:, 2]
    a = (mP + mS) / 2
    b = (mP - mS) / 2
    matrix = (a[:, None, None] * np.eye(3)
              + b[:, None, None] * self.nematic_tensors)
    dif_velocity = np.matmul(matrix, f[:, :, None])[:, :, 0]

    aspect_ratios = np.fromiter(
        (cell.aspect_ratio for cell in self.cells),
        dtype=np.float64, count=n,
    )
    intrinsic_speeds = (self.cell_speed_max * (aspect_ratios - 1)
                        / (self.aspect_ratio_max - 1))
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
        projections = np.einsum('ij,ij->i', dif_velocity, intrinsic_velocity)
        # The scalar reference assigns projection=0 for zero-speed cells.
        # Avoid dividing by zero, including for round cells.
        np.divide(projections, speed_norms, out=projections,
                  where=speed_norms > 0)
        projections[speed_norms == 0] = 0.0
        not_round = np.fromiter(
            (not cell.is_round for cell in self.cells),
            dtype=bool, count=n,
        )
        shrinking_flags = not_round & (speed_norms + projections <= 0)
        for i, cell in enumerate(self.cells):
            cell.shrink = bool(shrinking_flags[i])
    return dif_positions, dif_phies


def batch_interaction_step(self, delta_t):
    """Return all step displacements, preserving the original per-cell RNG order."""
    from tumorsphere.core.forces import Anisotropic_Grosmann
    from tumorsphere.core.forces_numba import XI_POWER_MAX
    if (not isinstance(self.force, Anisotropic_Grosmann)
            or getattr(self.force, 'force_backend', None) != 'numba'
            or getattr(self, 'interaction_backend', None) not in ('batch_numba', 'batch_cached_numba', 'batch_cached_noise_numba')
            or not self.movement or self.reproduction):
        raise RuntimeError('batch_numba requires fixed-N Grosmann force_backend=numba')
    if list(self.active_cell_indexes) != list(range(len(self.cells))):
        raise RuntimeError('Requires cells active in ascending index order')
    if self.relative_positions_backend != 'numba' or self.overlap_backend != 'numba':
        raise RuntimeError('Requires Numba relative positions and overlap backends')
    # The validated canonical pair semantics assume planar cells.
    if np.any(self.cell_positions[:, 2] != 0.0):
        raise RuntimeError('batch_numba requires planar positions (z=0)')
    n = len(self.cells)
    ptr = [0]
    neighbors = []
    if getattr(self, 'interaction_backend', None) in ('batch_cached_numba', 'batch_cached_noise_numba'):
        from tumorsphere.core.neighbor_candidate_cache import OrderedNeighborCache
        cache = getattr(self.grid, '_batch_neighbor_cache', None)
        if cache is None:
            cache = OrderedNeighborCache()
            self.grid._batch_neighbor_cache = cache
        for i in range(n):
            neighbors.extend(cache.candidates(self.grid, i, self.cell_positions[i]))
            ptr.append(len(neighbors))
    else:
        # Preserve the original batch_numba path without changing its algebra.
        for i in range(n):
            # EXACT same order as direct_interaction, including OrderedIndexSet iteration.
            candidates = [int(j) for j in self.grid.find_neighbors(
                position=self.cell_positions[i]) if int(j) != i]
            earlier = sorted(j for j in candidates if j < i)
            later = [j for j in candidates if j > i]
            neighbors.extend(earlier)
            neighbors.extend(later)
            ptr.append(len(neighbors))
    geometry_d = np.asarray([c.squared_diagonal for c in self.cells], dtype=np.float64)
    geometry_e = np.asarray([c.anisotropy for c in self.cells], dtype=np.float64)
    qxx = np.ascontiguousarray(self.nematic_tensors[:, 0, 0])
    qxy = np.ascontiguousarray(self.nematic_tensors[:, 0, 1])
    f, t, xis = _batch_pair_forces(
        self.cell_positions, self.cell_phies, geometry_d, geometry_e,
        qxx, qxy, np.asarray(ptr, dtype=np.intp),
        np.asarray(neighbors, dtype=np.intp),
        float(self.side), float(self.cell_area),
        float(self.overlap_threshold_ratio), float(self.force.kRep),
        float(self.force.bExp), float(self.force.lambda_core),
        XI_POWER_MAX,
    )
    # New opt-in path: collect the already-cached mobilities in the
    # reference cell order, then advance NumPy's RNG once for the sweep.
    # No RNG is consumed between these draws in the original mechanical loop.
    use_compiled_noise = (
        getattr(self, 'interaction_backend', None) == 'batch_cached_noise_numba'
        and (self.force.translational_noise_enabled
             or self.force.rotational_noise_enabled)
    )
    if use_compiled_noise:
        from tumorsphere.core.interaction_noise_numba import compiled_noise_for_sweep
        mobilities = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            mobilities[i] = self.force.calculate_mobilities(
                self.cells[i], self.cell_area,
            )
        translation, rotation = compiled_noise_for_sweep(
            self, mobilities, delta_t,
        )
        return _vectorized_compiled_noise_postprocess(
            self, delta_t, f, t, mobilities, translation, rotation, xis,
        )
    dif_positions = np.empty((n, 3), dtype=np.float64)
    dif_phies = np.empty(n, dtype=np.float64)
    # The identity matrix is independent of the cell index.
    identity = np.identity(3)
    collect = getattr(self.output, 'collect_overlap_statistics', True)
    # xi contains the overlaps of every directed candidate pair in cell order.
    # Collect identical statistics with a single NumPy reduction per sweep,
    # avoiding two small-array NumPy calls for each cell. The sum may differ
    # in its final floating-point bits because the reduction order changes.
    if collect and xis.size:
        self.max_normalized_overlap_interval = max(
            self.max_normalized_overlap_interval, float(np.max(xis)))
        self.normalized_overlap_sum_interval += float(np.sum(xis))
        self.normalized_overlap_count_interval += int(xis.size)
    # Keep noise and shrink calls in the original cell order.
    for i in range(n):
        cell = self.cells[i]
        if use_compiled_noise:
            mP, mS, mR = mobilities[i]
        else:
            mP, mS, mR = self.force.calculate_mobilities(cell, self.cell_area)
        # Deliberately retain the original NumPy matmul arithmetic.
        dif_velocity = np.matmul(
            ((mP + mS) / 2) * identity + ((mP - mS) / 2)
            * self.nematic_tensors[i], f[i],
        )
        # Reuse the same computed intrinsic velocity for displacement and shrink.
        intrinsic_velocity = cell.velocity()
        dif_position = (intrinsic_velocity + dif_velocity) * delta_t
        dif_phi = mR * t[i] * delta_t
        if self.force.translational_noise_enabled or self.force.rotational_noise_enabled:
            if use_compiled_noise:
                dif_position += translation[i] if self.force.translational_noise_enabled else 0
                dif_phi += rotation[i] if self.force.rotational_noise_enabled else 0
            else:
                translational_noise, rotational_noise = self.force.calculate_noise(
                    self.cells, self.cell_phies, i, self.cell_area, delta_t, mP, mS, mR)
                dif_position += translational_noise
                dif_phi += rotational_noise
        if self.force.shrinking is True:
            # Keep NumPy norm, dot and scalar operation order identical to the
            # reference shrink condition; avoid repeating cell.velocity().
            speed = np.linalg.norm(intrinsic_velocity)
            if speed > 0:
                dif_velocity_project = np.dot(dif_velocity, intrinsic_velocity) / speed
            else:
                dif_velocity_project = 0
            cell.shrink = not cell.is_round and speed + dif_velocity_project <= 0
        dif_positions[i] = dif_position
        dif_phies[i] = dif_phi
    return dif_positions, dif_phies
