"""Numerical reference tests for Culture.calculate_overlap_components.

The scalar reference follows the model's analytic Gaussian-overlap formula
without constructing 3x3 nematic matrices. It supports nonzero z as well.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tumorsphere.core.culture import Culture


ASPECT_RATIOS = (1.0, 2.7, 1.15, 1.8, 1.0, 2.4)
ANGLES = (0.0, 0.37, -0.51, 1.38, math.pi / 2, 2.74)


def make_culture(area=math.pi):
    """Build only the attributes needed by the overlap routine."""
    culture = Culture.__new__(Culture)
    culture.backend = "numpy"
    culture.cell_area = area
    culture.cell_phies = np.asarray(ANGLES, dtype=float)
    culture.cells = []
    tensors = []

    for aspect, phi in zip(ASPECT_RATIOS, ANGLES):
        culture.cells.append(
            SimpleNamespace(
                squared_diagonal=(area / math.pi) * (aspect + 1.0 / aspect),
                anisotropy=(aspect * aspect - 1.0) / (aspect * aspect + 1.0),
            )
        )
        tensors.append(
            [
                [math.cos(2.0 * phi), math.sin(2.0 * phi), 0.0],
                [math.sin(2.0 * phi), -math.cos(2.0 * phi), 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

    culture.nematic_tensors = np.asarray(tensors, dtype=float)
    return culture


def scalar_overlap_reference(culture, index, neighbors, relative_positions):
    """Independent, pair-by-pair reference derived from the analytic formula."""
    i = culture.cells[index]
    phi_i = culture.cell_phies[index]
    d_i, eps_i = i.squared_diagonal, i.anisotropy
    absolute, normalized, maxima = [], [], []

    for j, (x, y, z) in zip(neighbors, relative_positions):
        other = culture.cells[j]
        phi_j = culture.cell_phies[j]
        d_j, eps_j = other.squared_diagonal, other.anisotropy
        total_d = d_i + d_j

        beta = (
            total_d**2
            - (d_i * eps_i - d_j * eps_j) ** 2
            - 4.0 * d_i * d_j * eps_i * eps_j
            * math.cos(phi_i - phi_j) ** 2
        )

        m_xx = (
            d_i * eps_i * math.cos(2.0 * phi_i)
            + d_j * eps_j * math.cos(2.0 * phi_j)
        ) / total_d
        m_xy = (
            d_i * eps_i * math.sin(2.0 * phi_i)
            + d_j * eps_j * math.sin(2.0 * phi_j)
        ) / total_d

        # Q_xx=-Q_yy and Q_xz=Q_yz=Q_zz=0.
        # Consequently, the z contribution to r^T (I-M) r is z^2.
        quadratic = (
            (1.0 - m_xx) * x * x
            + (1.0 + m_xx) * y * y
            - 2.0 * m_xy * x * y
            + z * z
        )
        norm = math.exp(-(total_d / beta) * quadratic)
        maximum = 4.0 * culture.cell_area**2 / (math.pi * math.sqrt(beta))
        normalized.append(norm)
        maxima.append(maximum)
        absolute.append(norm * maximum)

    return tuple(np.asarray(values, dtype=float) for values in (
        absolute, normalized, maxima
    ))


def assert_overlap_matches_reference(culture, index, neighbors, positions):
    actual = culture.calculate_overlap_components(index, neighbors, positions)
    expected = scalar_overlap_reference(culture, index, neighbors, positions)

    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=2e-12, atol=1e-14)
        assert result.shape == (len(neighbors),)
        assert np.all(np.isfinite(result))


@pytest.mark.parametrize("area", [math.pi, 1.96 * math.pi])
@pytest.mark.parametrize("reference_index", range(len(ASPECT_RATIOS)))
@pytest.mark.parametrize("nonzero_z", [False, True])
def test_overlap_reference_for_mixed_shapes(area, reference_index, nonzero_z):
    """Round/elongated cells, orientations and off-plane positions."""
    culture = make_culture(area)
    neighbors = [i for i in range(len(ASPECT_RATIOS)) if i != reference_index]
    rng = np.random.default_rng(20261012 + reference_index)
    relative_positions = rng.normal(size=(len(neighbors), 3)) * 2.0
    if not nonzero_z:
        relative_positions[:, 2] = 0.0

    assert_overlap_matches_reference(
        culture, reference_index, neighbors, relative_positions
    )


@pytest.mark.parametrize("reference_index,neighbor_index", [(0, 1), (1, 2), (3, 5), (4, 0)])
def test_overlap_threshold_decisions(reference_index, neighbor_index):
    """The two implementations agree on either side of exp(-1)."""
    culture = make_culture()
    i, j = culture.cells[reference_index], culture.cells[neighbor_index]
    phi_i = culture.cell_phies[reference_index]
    phi_j = culture.cell_phies[neighbor_index]
    d_i, d_j = i.squared_diagonal, j.squared_diagonal
    eps_i, eps_j = i.anisotropy, j.anisotropy
    total_d = d_i + d_j
    beta = (
        total_d**2 - (d_i * eps_i - d_j * eps_j) ** 2
        - 4.0 * d_i * d_j * eps_i * eps_j * np.cos(phi_i - phi_j) ** 2
    )
    m_xx = (
        d_i * eps_i * np.cos(2.0 * phi_i)
        + d_j * eps_j * np.cos(2.0 * phi_j)
    ) / total_d
    coefficient = total_d / beta * (1.0 - m_xx)
    threshold = math.exp(-1.0)
    distance = math.sqrt(-math.log(threshold) / coefficient)
    positions = np.array([
        [distance * (1.0 - 1e-5), 0.0, 0.0],
        [distance * (1.0 + 1e-5), 0.0, 0.0],
    ])
    neighbors = [neighbor_index, neighbor_index]

    assert_overlap_matches_reference(culture, reference_index, neighbors, positions)
    _, normalized, _ = culture.calculate_overlap_components(
        reference_index, neighbors, positions
    )
    assert normalized[0] > threshold
    assert normalized[1] < threshold


def test_overlap_at_zero_separation():
    culture = make_culture()
    neighbors = [1, 2, 3]
    zeros = np.zeros((len(neighbors), 3))
    overlaps, normalized, maxima = culture.calculate_overlap_components(
        0, neighbors, zeros
    )
    np.testing.assert_allclose(normalized, 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(overlaps, maxima, rtol=0, atol=1e-14)


def test_overlap_empty_neighbors():
    culture = make_culture()
    actual = culture.calculate_overlap_components(
        0, [], np.empty((0, 3), dtype=float)
    )
    for result in actual:
        assert result.shape == (0,)


# Compiled-overlap equivalence tests (formerly test_numba_overlap.py).

@pytest.mark.parametrize("nonplanar", [False, True])
@pytest.mark.parametrize("n_neighbors", [0, 1, 2, 5, 10, 20])
@pytest.mark.parametrize("seed", [1, 5])
def test_numba_vs_numpy_overlap_components(seed, n_neighbors, nonplanar):
    """Numerical equivalence for mixed shapes, orientations and off-plane r."""
    pytest.importorskip("numba")
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

    culture.backend = "numpy"
    ref = culture.calculate_overlap_components(0, neighbors, relative_positions)
    culture.backend = "numba"
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
    culture.backend = "nonexistent"
    with pytest.raises(ValueError, match="Unknown backend"):
        culture.calculate_overlap_components(0, [], np.empty((0, 3)))
