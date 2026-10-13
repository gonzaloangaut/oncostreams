"""Optional Numba implementation for anisotropic Grosmann pair interactions.

Imported lazily, only when Anisotropic_Grosmann.force_backend == "numba".
The algorithm leaves mobilities, stochastic draws and shrinking in Python.
No fastmath and no parallel reduction; the order of neighbors is preserved.
"""

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "force_backend='numba' requires Numba. "
        "Install it with: python -m pip install numba"
    ) from exc


# The NumPy reference uses nextafter(1, 0), not an arbitrary epsilon.
XI_POWER_MAX = float(np.nextafter(1.0, 0.0))


@njit(cache=True, fastmath=False)
def grosmann_force_torque_kernel(
    d_i, eps_i, phi_i, qi_xx, qi_xy,
    d_j, eps_j, phi_j, qj_xx, qj_xy,
    relative_positions, normalized_overlaps,
    k_rep, gamma, lambda_core, xi_power_max,
):
    """Accumulate 3D pair forces and torque; preserve ordered neighbor iteration.

    Qxx=-Qyy, Qxy=Qyx and every tensor's z row/column is zero. This is the
    invariant maintained by Cell/ Culture. Positions themselves may have z != 0.

    Note that xi comes from the upstream overlap cache, *not* recomputed here.
    """
    force = np.zeros(3, dtype=np.float64)
    torque = 0.0
    w_i = d_i * eps_i

    for k in range(len(d_j)):
        dj = d_j[k]
        ej = eps_j[k]
        w_j = dj * ej
        ds = d_i + dj
        relative_angle = phi_i - phi_j[k]
        cos_rel = np.cos(relative_angle)

        mxx = (w_i * qi_xx + w_j * qj_xx[k]) / ds
        mxy = (w_i * qi_xy + w_j * qj_xy[k]) / ds

        beta = (ds ** 2 - (w_i - w_j) ** 2
                - 4.0 * d_i * eps_i * dj * ej * cos_rel ** 2)

        xi_power = normalized_overlaps[k] ** gamma
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

        rx = relative_positions[k, 0]
        ry = relative_positions[k, 1]
        rz = relative_positions[k, 2]

        cx = (1.0 - mxx) * rx - mxy * ry
        cy = -mxy * rx + (1.0 + mxx) * ry
        cz = rz

        force[0] += kernel * cx
        force[1] += kernel * cy
        force[2] += kernel * cz

        theta = np.arctan2(ry, rx)
        r_dot_c = rx * cx + ry * cy + rz * cz
        r_squared = rx * rx + ry * ry + rz * rz
        torque += kernel * (
            ((2.0 * d_i * eps_i * dj * ej * np.sin(-2.0 * relative_angle))
             / beta) * r_dot_c
            + (d_i * eps_i / ds) * r_squared
            * np.sin(2.0 * (phi_i - theta))
        )

    return force, torque


def prepare_force_arrays(cells, phies, cell_index, neighbors_indexes,
                         nematic_tensors):
    """Python bridge data only; no changes to cell caches or neighbor ordering."""
    cell = cells[cell_index]
    idx = np.asarray(neighbors_indexes, dtype=np.intp)
    if idx.ndim != 1:
        raise ValueError("neighbors_indexes must be a 1D sequence")
    n = len(idx)
    d_j = np.empty(n, dtype=np.float64)
    eps_j = np.empty(n, dtype=np.float64)
    relative_positions = np.empty((n, 3), dtype=np.float64)
    xis = np.empty(n, dtype=np.float64)
    for k, neighbor_index in enumerate(idx):
        index = int(neighbor_index)
        neighbor = cells[index]
        d_j[k] = neighbor.squared_diagonal
        eps_j[k] = neighbor.anisotropy
        relative_positions[k] = cell.neighbors_relative_pos[index]
        xis[k] = cell.neighbors_normalized_overlap[index]
    # Advanced indexing copies; the small arrays are intentional in this
    # first experiment. Their cost is included in the full benchmark.
    phies_j = np.asarray(phies[idx], dtype=np.float64)
    q_j = np.asarray(nematic_tensors[idx], dtype=np.float64)
    return (
        float(cell.squared_diagonal), float(cell.anisotropy),
        float(phies[cell_index]),
        float(nematic_tensors[cell_index, 0, 0]),
        float(nematic_tensors[cell_index, 0, 1]),
        d_j, eps_j, phies_j,
        np.ascontiguousarray(q_j[:, 0, 0]),
        np.ascontiguousarray(q_j[:, 0, 1]),
        relative_positions, xis,
    )


def calculate_interaction_numba(force_model, cells, phies, cell_index,
                                delta_t, area, neighbors_indexes,
                                nematic_tensors):
    """Drop-in adapter: preserve mobilities, noise and shrink decisions."""
    cell = cells[cell_index]
    q_cell = nematic_tensors[cell_index]
    mP, mS, mR = force_model.calculate_mobilities(cell, area)
    # For isolated cells avoid array extraction and a compiled-kernel call.
    # The remaining (mobility/noise/shrinking) work is unchanged.
    if len(neighbors_indexes) == 0:
        force = np.zeros(3, dtype=np.float64)
        torque = 0.0
    else:
        data = prepare_force_arrays(
            cells, phies, cell_index, neighbors_indexes, nematic_tensors
        )
        force, torque = grosmann_force_torque_kernel(
            *data, float(force_model.kRep), float(force_model.bExp),
            float(force_model.lambda_core), XI_POWER_MAX,
        )

    identity = np.identity(3)
    dif_velocity = np.matmul(
        ((mP + mS) / 2) * identity + ((mP - mS) / 2) * q_cell,
        force,
    )
    dif_position = (cell.velocity() + dif_velocity) * delta_t
    dif_phi = mR * torque * delta_t

    if force_model.translational_noise_enabled or force_model.rotational_noise_enabled:
        translational_noise, rotational_noise = force_model.calculate_noise(
            cells, phies, cell_index, area, delta_t, mP, mS, mR,
        )
        dif_position += translational_noise
        dif_phi += rotational_noise

    if force_model.shrinking is True:
        force_model.check_shrink_condition(cell, dif_velocity)

    return dif_position, dif_phi
