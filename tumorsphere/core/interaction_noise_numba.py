"""Compiled noise increments for the production batch interaction sweep.

The RNG remains NumPy's Generator, advancing in the reference cell
order. Only deterministic transformations of the sampled random
values execute in Numba. ``fastmath`` is disabled, but exact binary
agreement across different library versions is not guaranteed.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _numba_noise_all(
    phies,
    mobility_p,
    mobility_s,
    mobility_r,
    eta,
    rotational_strength,
    dt,
    randoms,
    has_translational,
    has_rotational,
):
    """Noise algebra in exactly the scalar order used by calculate_noise().

    Array 3rd component preserves signed-zero arithmetic from numpy's 3D
    direction and perpendicular arrays. There is no RNG in this function.
    """
    n = len(phies)
    translation = np.empty((n, 3), dtype=np.float64)
    rotation = np.empty(n, dtype=np.float64)
    k = 0
    for i in range(n):
        if has_translational:
            phi = phies[i]
            # Match: eta * sqrt(mP*dt), then multiply by the first scalar draw.
            s_np = eta * np.sqrt(mobility_p[i] * dt)
            np_ = s_np * randoms[k]
            k += 1
            parallel_x = np_ * np.cos(phi)
            parallel_y = np_ * np.sin(phi)
            parallel_z = np_ * 0.0
            # Match: eta * sqrt(mS*dt), then second scalar draw.
            s_ns = eta * np.sqrt(mobility_s[i] * dt)
            ns_ = s_ns * randoms[k]
            k += 1
            perpendicular_x = ns_ * np.cos(phi + np.pi / 2)
            perpendicular_y = ns_ * np.sin(phi + np.pi / 2)
            perpendicular_z = ns_ * 0.0
            translation[i, 0] = parallel_x + perpendicular_x
            translation[i, 1] = parallel_y + perpendicular_y
            translation[i, 2] = parallel_z + perpendicular_z
        else:
            translation[i, 0] = 0.0
            translation[i, 1] = 0.0
            translation[i, 2] = 0.0

        if has_rotational:
            s_nr = np.sqrt(2 * rotational_strength * mobility_r[i] * dt)
            rotation[i] = s_nr * randoms[k]
            k += 1
        else:
            rotation[i] = 0.0
    return translation, rotation


def compiled_noise_for_sweep(culture, mobilities, delta_t):
    """Return (translation[n,3], rotation[n]) and consume exact scalar RNG order.

    The mechanical pair sweep and cell deformation have already completed.
    No intermediate RNG consumers occur within post-force processing.
    """
    force = culture.force
    trans = bool(force.translational_noise_enabled)
    rot = bool(force.rotational_noise_enabled)
    if not (trans or rot):
        raise ValueError("The compiled noise path requires enabled noise")
    n = len(culture.cells)
    n_draws = n * (2 * int(trans) + int(rot))
    randoms = culture.rng.normal(0, 1, size=n_draws)
    return _numba_noise_all(
        culture.cell_phies,
        np.ascontiguousarray(mobilities[:, 0]),
        np.ascontiguousarray(mobilities[:, 1]),
        np.ascontiguousarray(mobilities[:, 2]),
        float(force.noise_eta) if trans else 0.0,
        float(force.d_phi) if rot else 0.0,
        float(delta_t),
        randoms,
        trans,
        rot,
    )
