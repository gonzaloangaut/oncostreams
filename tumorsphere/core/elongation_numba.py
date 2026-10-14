"""Experimental fixed-N round-cell elongation fastpath, in-memory only.

No mutation to the proposed cell before accepting a proposal. Reuses neighbor
geometry within each bucket. Keeps one-by-one RNG draws, exact tie breaking,
Numba overlap arithmetic and original commit/gridding behavior.

This is not production code until A/B trajectory/RNG validation passes.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def max_normalized_overlap_proposal(
    d_i, eps_i, phi_i, qi_xx, qi_xy,
    d_j, eps_j, phi_j, qj_xx, qj_xy,
    relative_positions, area,
):
    """Return max normalized Gaussian overlap for one trial geometry.

    Mathematical operation order mirrors calculate_overlap_numba's inner loop.
    `imax` is intentionally computed to retain the original kernel's sequence
    of floating point evaluations during first correctness comparison.
    """
    maximum = 0.0
    for k in range(len(d_j)):
        dj = d_j[k]
        ej = eps_j[k]
        ds = d_i + dj
        delta = phi_i - phi_j[k]
        cos_delta = np.cos(delta)
        beta = (
            ds ** 2
            - (d_i * eps_i - dj * ej) ** 2
            - 4.0 * d_i * dj * eps_i * ej * (cos_delta ** 2)
        )
        wi = d_i * eps_i
        wj = dj * ej
        mxx = (wi * qi_xx + wj * qj_xx[k]) / ds
        mxy = (wi * qi_xy + wj * qj_xy[k]) / ds
        rx = relative_positions[k, 0]
        ry = relative_positions[k, 1]
        rz = relative_positions[k, 2]
        quad = (
            (1.0 - mxx) * rx ** 2
            + (1.0 + mxx) * ry ** 2
            - 2.0 * mxy * rx * ry
            + rz ** 2
        )
        # Deliberately identical to original overlap kernel, even though
        # absolute overlap maximum is unused by the deformation criterion.
        imax = 4.0 * area ** 2 / (np.pi * np.sqrt(beta))
        norm = np.exp(-(ds / beta) * quad)
        if np.isnan(norm):
            return np.nan
        if norm > maximum:
            maximum = norm
    return maximum


def elongate_from_round_fast(self, cell_index: int) -> bool:
    """Drop-in alternative for Culture.elongate_from_round (Numba overlaps).

    For each proposal, derive geometry locally. Do not modify Culture arrays,
    nematic tensors or the Cell until choosing an accepted proposal.
    """
    if getattr(self, 'overlap_backend', 'numpy') != 'numba':
        raise RuntimeError('Elongation fastpath requires overlap_backend=numba')

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

    # Exact expression/order from Cell.update_shape_parameters.
    d_i = (self.cell_area / np.pi) * (
        new_aspect_ratio + 1 / new_aspect_ratio
    )
    eps_i = (new_aspect_ratio ** 2 - 1) / (new_aspect_ratio ** 2 + 1)

    candidates_by_bucket = {}
    best_score = np.inf
    best_proposals = []

    for _ in range(n_attempts):
        # Preserve the scalar RNG calls and the original tie-breaking draw.
        new_phi = self.rng.uniform(low=0, high=2 * np.pi)
        new_position = np.mod(
            old_position
            + np.array([
                displacement_length * np.cos(new_phi),
                displacement_length * np.sin(new_phi),
                0.0,
            ]),
            self.side,
        )
        proposal_bucket = self.grid.get_hash_key(new_position)
        if proposal_bucket not in candidates_by_bucket:
            candidate_neighbors = [
                neighbor_index
                for neighbor_index in self.grid.find_neighbors(position=new_position)
                if neighbor_index != cell_index
            ]
            idx = np.asarray(candidate_neighbors, dtype=np.intp)
            if len(idx):
                neighbors = [self.cells[int(j)] for j in idx]
                d_j = np.array([n.squared_diagonal for n in neighbors], dtype=np.float64)
                eps_j = np.array([n.anisotropy for n in neighbors], dtype=np.float64)
                phi_j = np.asarray(self.cell_phies[idx], dtype=np.float64)
                q_j = self.nematic_tensors[idx]
                qj_xx = np.ascontiguousarray(q_j[:, 0, 0])
                qj_xy = np.ascontiguousarray(q_j[:, 0, 1])
            else:
                d_j = np.empty(0, dtype=np.float64)
                eps_j = np.empty(0, dtype=np.float64)
                phi_j = np.empty(0, dtype=np.float64)
                qj_xx = np.empty(0, dtype=np.float64)
                qj_xy = np.empty(0, dtype=np.float64)
            candidates_by_bucket[proposal_bucket] = (
                idx, d_j, eps_j, phi_j, qj_xx, qj_xy
            )

        idx, d_j, eps_j, phi_j, qj_xx, qj_xy = candidates_by_bucket[proposal_bucket]
        if len(idx):
            relative = self.calculate_relative_positions(
                new_position, self.cell_positions[idx]
            )
            # Exact expressions from Culture.update_nematic_tensors.
            qi_xx = np.cos(2 * new_phi)
            qi_xy = np.sin(2 * new_phi)
            score = float(max_normalized_overlap_proposal(
                float(d_i), float(eps_i), float(new_phi),
                float(qi_xx), float(qi_xy),
                d_j, eps_j, phi_j, qj_xx, qj_xy,
                np.asarray(relative, dtype=np.float64), float(self.cell_area),
            ))
        else:
            score = 0.0

        if score <= self.overlap_threshold_ratio:
            if score < best_score:
                best_score = score
                best_proposals = [(new_phi, new_position)]
            elif score == best_score:
                best_proposals.append((new_phi, new_position))

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
