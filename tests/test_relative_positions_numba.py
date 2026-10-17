"""Reference equivalence tests for periodic relative positions.

NumPy is the reference. The x/y minimum-image rule and unwrapped z
must remain unchanged when selecting backend="numba".
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tumorsphere.core.culture import Culture


def numpy_reference(cell, neighbors, side):
    dummy = SimpleNamespace(side=side, backend="numpy")
    return Culture.calculate_relative_positions(dummy, cell, neighbors)


def actual_backend(cell, neighbors, side):
    pytest.importorskip("numba")
    dummy = SimpleNamespace(side=side, backend="numba")
    return Culture.calculate_relative_positions(dummy, cell, neighbors)


SEEDS = (0, 1, 7, 1234, 20261013)
NEIGHBOR_COUNTS = (0, 1, 2, 8, 24, 64)
SIDES = (7.0, 12.5, 101.0)

# Balanced deterministic coverage: 60 cases instead of 180. Every seed
# and planar/nonplanar mode sees all three box sizes across neighbor counts.
RELATIVE_POSITION_CASES = [
    (seed, n, SIDES[(seed_i + count_i + int(nonplanar)) % len(SIDES)],
     nonplanar)
    for seed_i, seed in enumerate(SEEDS)
    for count_i, n in enumerate(NEIGHBOR_COUNTS)
    for nonplanar in (False, True)
]

@pytest.mark.parametrize("seed,n,side,nonplanar", RELATIVE_POSITION_CASES)
def test_random_float64_reference_bitwise(seed, n, side, nonplanar):
    rng = np.random.default_rng(seed)
    cell = rng.uniform(0, side, size=3).astype(np.float64)
    neighbors = rng.uniform(0, side, size=(n, 3)).astype(np.float64)
    if not nonplanar:
        cell[2] = 0.0
        neighbors[:, 2] = 0.0
    original_cell = cell.copy()
    original_neighbors = neighbors.copy()
    expected = numpy_reference(cell, neighbors, side)
    obtained = actual_backend(cell, neighbors, side)
    assert expected.shape == obtained.shape == (n, 3)
    np.testing.assert_array_equal(obtained, expected)
    # The computation must not modify the input arrays.
    np.testing.assert_array_equal(cell, original_cell)
    np.testing.assert_array_equal(neighbors, original_neighbors)


@pytest.mark.parametrize("side", [7.0, 12.5, 101.0])
def test_boundaries_and_empty(side):
    half = side / 2
    above = np.nextafter(half, float("inf"))
    below = np.nextafter(half, float("-inf"))
    cell = np.array([0.0, 0.0, 3.5], dtype=np.float64)
    neighbors = np.array([
        [half, half, 1.0],
        [-half, -half, 2.0],
        [above, above, 4.0],
        [-above, -above, 5.0],
        [below, below, 7.0],
        [-below, -below, 8.0],
        [side, -side, 0.0],
        [0.0, 0.0, 3.5],
    ], dtype=np.float64)
    np.testing.assert_array_equal(
        actual_backend(cell, neighbors, side),
        numpy_reference(cell, neighbors, side),
    )
    assert actual_backend(cell, neighbors[:0], side).shape == (0, 3)


def test_unknown_backend_rejected():
    cell = np.zeros(3)
    neighbors = np.zeros((0, 3))
    with pytest.raises(ValueError, match="Unknown backend"):
        Culture.calculate_relative_positions(
            SimpleNamespace(side=10.0, backend="cuda"),
            cell, neighbors,
        )
