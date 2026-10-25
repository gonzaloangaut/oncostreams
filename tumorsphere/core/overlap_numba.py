"""Numba implementation of Gaussian overlap calculations.

This module is loaded lazily by the optimized fixed-N backend. The
default NumPy backend does not require Numba to be imported.
"""

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "The optional Numba overlap backend requires Numba. "
        "Install it in the active virtualenv using: python -m pip install numba"
    ) from exc


@njit(cache=True, fastmath=False)
def calculate_overlap_numba(
    d_i,
    d_j,
    eps_i,
    eps_j,
    phi_i,
    phi_j,
    qi_xx,
    qi_xy,
    qj_xx,
    qj_xy,
    r,
    area,
):
    """Return (absolute, normalized, maximum) overlaps for one reference cell.

    The cell's nematic tensor has Qyy=-Qxx, Qxy=Qyx, Qxz=Qyz=Qzz=0;
    therefore the 3D quadratic form can be evaluated without 3x3 matrices.

    Input arrays are float64 and r has shape (n_neighbors, 3). `fastmath`
    is disabled to avoid extra floating-point re-associations.
    """
    n = len(d_j)
    absolute = np.empty(n, dtype=np.float64)
    normalized = np.empty(n, dtype=np.float64)
    maxima = np.empty(n, dtype=np.float64)

    for k in range(n):
        dj = d_j[k]
        ej = eps_j[k]
        ds = d_i + dj
        delta = phi_i - phi_j[k]
        cos_delta = np.cos(delta)
        beta = (
            ds**2
            - (d_i * eps_i - dj * ej) ** 2
            - 4.0 * d_i * dj * eps_i * ej * (cos_delta**2)
        )

        wi = d_i * eps_i
        wj = dj * ej
        mxx = (wi * qi_xx + wj * qj_xx[k]) / ds
        mxy = (wi * qi_xy + wj * qj_xy[k]) / ds

        rx = r[k, 0]
        ry = r[k, 1]
        rz = r[k, 2]
        quad = (
            (1.0 - mxx) * rx**2
            + (1.0 + mxx) * ry**2
            - 2.0 * mxy * rx * ry
            + rz**2
        )

        imax = 4.0 * area**2 / (np.pi * np.sqrt(beta))
        norm = np.exp(-(ds / beta) * quad)
        absolute[k] = imax * norm
        normalized[k] = norm
        maxima[k] = imax

    return absolute, normalized, maxima
