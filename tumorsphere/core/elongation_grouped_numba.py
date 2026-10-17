"""Opt-in grouped round-cell elongation with safe early overlap rejection.

This implementation is based on the experimentally validated ``combined``
mode from ``oncostreams_elongation_probe.py``. It preserves, for each cell:
  * the NumPy RNG sequence and per-proposal position/orientation arithmetic;
  * all attempted orientations, including rejected ones;
  * the best-valid-score selection and the final tie-breaking RNG draw;
  * sequential deformation processing in the caller-defined cell order;
  * accepted cell updates and spatial-grid insertion order.

The expensive neighbor/overlap calculation is fused into one Numba call
per round-cell elongation. Invalid proposals stop at the first overlap
above threshold. Valid scores are evaluated completely in reference order.

Numba/NumPy bitwise equivalence is version-dependent and is verified by tests.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def _shared_elongation_geometry(culture):
    """Lazy shape arrays, kept current by Cell.update_shape_parameters.

    Owned by the grid so it can be excluded from checkpoints alongside
    the existing interaction candidate cache.
    """
    grid = culture.grid
    cached = getattr(grid, '_elongation_geometry_cache', None)
    if cached is None or len(cached[0]) != len(culture.cells):
        cached = (
            np.array([cell.squared_diagonal for cell in culture.cells],
                     dtype=np.float64),
            np.array([cell.anisotropy for cell in culture.cells],
                     dtype=np.float64),
        )
        grid._elongation_geometry_cache = cached
    return cached


def _cached_elongation_neighbors(grid, bucket_key, position):
    """Return the ORIGINAL ordered candidate list for one occupied bucket.

    Membership (and insertion order) is unchanged until an add/remove call.
    SpatialHashGrid clears this cache after every such mutation.
    """
    cache = getattr(grid, '_elongation_bucket_cache', None)
    if cache is None:
        cache = {}
        grid._elongation_bucket_cache = cache
    result = cache.get(bucket_key)
    if result is None:
        result = tuple(grid.find_neighbors(position=position))
        cache[bucket_key] = result
    return result


@njit(cache=True, fastmath=False)
def compiled_proposals(old_position, displacement_length, phis, side):
    """Build proposed centers and nematic components in the original order.

    The NumPy per-proposal arithmetic (including 3D modulus) is retained to
    preserve exact trajectories on the validated NumPy/Numba platforms.
    RNG draws stay in Python and are performed before calling this function.
    """
    count = len(phis)
    positions = np.empty((count, 3), dtype=np.float64)
    qxx = np.empty(count, dtype=np.float64)
    qxy = np.empty(count, dtype=np.float64)
    for a in range(count):
        phi = phis[a]
        positions[a] = np.mod(
            old_position + np.array([
                displacement_length * np.cos(phi),
                displacement_length * np.sin(phi),
                0.0,
            ]), side,
        )
        qxx[a] = np.cos(2 * phi)
        qxy[a] = np.sin(2 * phi)
    return positions, qxx, qxy

@njit(cache=True, fastmath=False)
def grouped_proposal_scores(
    proposed_positions, proposed_phies, proposed_qxx, proposed_qxy,
    trial_buckets, ptr, neighbor_indices,
    positions, phies, diagonals, anisotropies, qxx, qxy,
    d_i, eps_i, side, threshold,
):
    """Compute all independent proposals of ONE cell within one Numba call.

    Positions/angles/nematic components are produced using the original
    scalar NumPy math outside this kernel. Buckets are resolved in Python.
    Only minimum-image displacements and overlap evaluation are fused.
    """
    out = np.empty(len(proposed_phies), dtype=np.float64)
    half_side = 0.5 * side
    for a in range(len(proposed_phies)):
        bi = trial_buckets[a]
        left = ptr[bi]
        right = ptr[bi + 1]
        maximum = 0.0
        for k in range(left, right):
            j = neighbor_indices[k]
            dx = proposed_positions[a, 0] - positions[j, 0]
            dy = proposed_positions[a, 1] - positions[j, 1]
            dz = proposed_positions[a, 2] - positions[j, 2]
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
            dj = diagonals[j]
            ej = anisotropies[j]
            ds = d_i + dj
            delta = proposed_phies[a] - phies[j]
            cos_delta = np.cos(delta)
            beta = (ds ** 2 - (d_i * eps_i - dj * ej) ** 2
                    - 4.0 * d_i * dj * eps_i * ej * (cos_delta ** 2))
            wi = d_i * eps_i
            wj = dj * ej
            mxx = (wi * proposed_qxx[a] + wj * qxx[j]) / ds
            mxy = (wi * proposed_qxy[a] + wj * qxy[j]) / ds
            quad = ((1.0 - mxx) * dx ** 2 + (1.0 + mxx) * dy ** 2
                    - 2.0 * mxy * dx * dy + dz ** 2)
            norm = np.exp(-(ds / beta) * quad)
            if np.isnan(norm):
                maximum = np.nan
                break
            if norm > maximum:
                maximum = norm
            if norm > threshold:
                break
        out[a] = maximum
    return out


def elongate_from_round_grouped(self, cell_index: int) -> bool:
    """Replace elongate_from_round_fast; never modifies state until acceptance."""
    if self.backend != 'numba':
        raise RuntimeError('Grouped elongation requires backend=numba')
    cell = self.cells[cell_index]
    n_attempts = self.cell_max_def_attempts
    if n_attempts <= 0:
        return False

    old_position = self.cell_positions[cell_index].copy()
    old_aspect_ratio = cell.aspect_ratio
    old_bucket = self.grid.get_hash_key(old_position)
    new_aspect_ratio = min(old_aspect_ratio + self.delta_aspect_ratio,
                           self.aspect_ratio_max)
    displacement_length = np.sqrt(
        (self.cell_area * new_aspect_ratio) / np.pi
    ) - np.sqrt((self.cell_area * old_aspect_ratio) / np.pi)
    d_i = (self.cell_area / np.pi) * (
        new_aspect_ratio + 1 / new_aspect_ratio
    )
    eps_i = (new_aspect_ratio ** 2 - 1) / (new_aspect_ratio ** 2 + 1)

    # Bulk draws preserve the scalar NumPy RNG sequence and generator state.
    # Keep the scalar path for a single attempt to avoid an unnecessary array.
    # The compiled helper keeps trigonometry and 3D np.mod in original order.
    trial_buckets = np.empty(n_attempts, dtype=np.intp)
    bucket_ids = {}
    ptr = [0]
    flat_neighbors = []
    if n_attempts == 1:
        new_phis = (self.rng.uniform(low=0, high=2 * np.pi),)
    else:
        new_phis = self.rng.uniform(low=0, high=2 * np.pi, size=n_attempts)
    phis = np.asarray(new_phis, dtype=np.float64)
    positions, proposed_qxx, proposed_qxy = compiled_proposals(
        old_position, displacement_length, phis, float(self.side),
    )
    for a in range(n_attempts):
        new_position = positions[a]
        key = self.grid.get_hash_key(new_position)
        bi = bucket_ids.get(key)
        if bi is None:
            bi = len(bucket_ids)
            bucket_ids[key] = bi
            idx = [int(j) for j in _cached_elongation_neighbors(
                self.grid, key, new_position) if int(j) != cell_index]
            flat_neighbors.extend(idx)
            ptr.append(len(flat_neighbors))
        trial_buckets[a] = bi

    # Gather each neighbor's geometry ONCE per cell instead of per proposal.
    # Global arrays avoid packing repeated per-neighbor geometry for each bucket.
    indices = np.asarray(flat_neighbors, dtype=np.intp)
    diag, eps = _shared_elongation_geometry(self)
    qs = self.nematic_tensors
    scores = grouped_proposal_scores(
        positions, phis, proposed_qxx, proposed_qxy,
        trial_buckets, np.asarray(ptr, dtype=np.intp), indices,
        self.cell_positions, self.cell_phies, diag, eps,
        np.ascontiguousarray(qs[:, 0, 0]),
        np.ascontiguousarray(qs[:, 0, 1]),
        float(d_i), float(eps_i), float(self.side),
        float(self.overlap_threshold_ratio),
    )

    best_score = np.inf
    best_proposals = []
    for a in range(n_attempts):
        score = float(scores[a])
        if score <= self.overlap_threshold_ratio:
            if score < best_score:
                best_score = score
                best_proposals = [(phis[a], positions[a])]
            elif score == best_score:
                best_proposals.append((phis[a], positions[a]))

    if not best_proposals:
        return False

    chosen_index = self.rng.choice(len(best_proposals))
    chosen_phi, chosen_position = best_proposals[chosen_index]
    cell.set_aspect_ratio(new_aspect_ratio)
    self.cell_positions[cell_index] = chosen_position
    self.cell_phies[cell_index] = chosen_phi
    self.update_nematic_tensors([cell_index])
    chosen_bucket = self.grid.get_hash_key(chosen_position)
    if chosen_bucket != old_bucket:
        self.grid.remove_cell_from_hash_table(cell_index, old_position)
        self.grid.add_cell_to_hash_table(cell_index, chosen_position)
    return True

