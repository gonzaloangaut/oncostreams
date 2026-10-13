"""Optional Numba backend for planar periodic relative displacements.

This is the same scalar kernel as the successful isolated experiment.
The x/y minimum-image rule uses strictly ``abs(d) > L/2``. The z component
is deliberately NOT wrapped, matching Culture.calculate_relative_positions.

The import is lazy from Culture; running the default NumPy backend does not
require Numba to be installed.
"""

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "relative_positions_backend='numba' requires Numba. "
        "Install it with: python -m pip install numba"
    ) from exc


@njit(cache=True, fastmath=False)
def periodic_relative_positions_numba(cell_pos, neighbors_pos, side):
    """Return ``cell_pos - neighbors_pos`` using the original x/y PBC rule.

    Inputs in Oncostreams are float64 arrays shaped (3,) and (n, 3).
    Output is float64 (n, 3); an empty input returns shape (0, 3).
    """
    result = np.empty((len(neighbors_pos), 3), dtype=np.float64)
    half_side = 0.5 * side
    for k in range(len(neighbors_pos)):
        dx = cell_pos[0] - neighbors_pos[k, 0]
        dy = cell_pos[1] - neighbors_pos[k, 1]
        dz = cell_pos[2] - neighbors_pos[k, 2]
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
        result[k, 0] = dx
        result[k, 1] = dy
        result[k, 2] = dz
    return result
