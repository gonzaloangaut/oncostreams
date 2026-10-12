"""Optional Numba-backend checks; skips cleanly without Numba installed.

Install as tests/test_numba_overlap.py. Only the reference (NumPy) path is
required for normal Oncostreams operation.
"""

from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("numba")
from tumorsphere.core.culture import Culture


@pytest.mark.parametrize("nonplanar", [False, True])
@pytest.mark.parametrize("n_neighbors", [0, 1, 2, 5, 10, 20])
@pytest.mark.parametrize("seed", [1, 5])
def test_numba_vs_numpy_overlap_components(seed, n_neighbors, nonplanar):
    """Numerical equivalence for mixed shapes, orientations and off-plane r."""
    rng = np.random.default_rng(seed * 100 + n_neighbors)
    n = n_neighbors + 1
    area = np.pi
    aspect = rng.choice([1.0, 1.1, 1.6, 2.7], size=n)
    phies = rng.uniform(-np.pi, np.pi, size=n)
    cells = [
        SimpleNamespace(
            squared_diagonal=(area / np.pi) * (a + 1.0 / a),
            anisotropy=(a * a - 1.0) / (a * a + 1.0),
        )
        for a in aspect
    ]
    q = np.zeros((n, 3, 3), dtype=np.float64)
    q[:, 0, 0] = np.cos(2.0 * phies)
    q[:, 0, 1] = np.sin(2.0 * phies)
    q[:, 1, 0] = q[:, 0, 1]
    q[:, 1, 1] = -q[:, 0, 0]
    relative_positions = rng.normal(size=(n_neighbors, 3))
    if not nonplanar:
        relative_positions[:, 2] = 0.0
    if n_neighbors:
        relative_positions[0] = 0.0
    culture = Culture.__new__(Culture)
    culture.cells = cells
    culture.cell_area = area
    culture.cell_phies = phies
    culture.nematic_tensors = q
    neighbors = list(range(1, n))

    culture.overlap_backend = "numpy"
    ref = culture.calculate_overlap_components(0, neighbors, relative_positions)
    culture.overlap_backend = "numba"
    got = culture.calculate_overlap_components(0, neighbors, relative_positions)
    for reference, actual in zip(ref, got):
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-14)
        assert actual.shape == (n_neighbors,)


def test_unknown_backend_rejected():
    culture = Culture.__new__(Culture)
    culture.cells = [SimpleNamespace(squared_diagonal=2.0, anisotropy=0.0)]
    culture.cell_area = np.pi
    culture.cell_phies = np.zeros(1)
    culture.nematic_tensors = np.zeros((1, 3, 3))
    culture.overlap_backend = "nonexistent"
    with pytest.raises(ValueError, match="Unknown overlap backend"):
        culture.calculate_overlap_components(0, [], np.empty((0, 3)))
