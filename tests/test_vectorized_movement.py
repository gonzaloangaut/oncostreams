"""Bitwise regression: vectorized Culture.move versus old per-cell bucket comparison.

The physics arithmetic and order of hash mutations must be unchanged.
"""

import numpy as np
import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


def make_minimal_culture(dimension, seed):
    """Create a controlled periodic culture with boundary-adjacent cells."""
    rng = np.random.default_rng(seed)
    n = 32
    culture = Culture.__new__(Culture)
    culture.side = 12.0
    culture.active_cell_indexes = list(range(n))
    culture.cell_positions = np.zeros((n, 3), dtype=np.float64)
    culture.cell_positions[:, :2] = rng.uniform(0.0, culture.side, (n, 2))
    culture.cell_positions[:4, :2] = [
        [0.01, 11.99], [11.99, 0.01], [2.99, 3.01], [6.01, 8.99]
    ]
    culture.cell_phies = rng.uniform(-np.pi, np.pi, n)
    culture.nematic_tensors = np.empty((n, 3, 3), dtype=np.float64)
    culture.update_nematic_tensors()
    culture.grid = SpatialHashGrid(
        culture=culture, bounds=culture.side, cube_size=3.0,
        torus=True, spatial_dimension=dimension,
    )
    for i, p in enumerate(culture.cell_positions):
        culture.grid.add_cell_to_hash_table(i, p)
    return culture


def original_move(culture, dif_positions, dif_phies):
    """Reference code from Culture.move before vectorization."""
    old_positions = culture.cell_positions.copy()
    culture.cell_positions = culture.cell_positions + dif_positions
    culture.cell_phies = culture.cell_phies + dif_phies
    culture.update_nematic_tensors()
    culture.cell_positions = np.mod(culture.cell_positions, culture.side)
    for cell_index in culture.active_cell_indexes:
        old_key = culture.grid.get_hash_key(old_positions[cell_index])
        new_key = culture.grid.get_hash_key(culture.cell_positions[cell_index])
        if old_key != new_key:
            culture.grid.remove_cell_from_hash_table(
                cell_index, old_positions[cell_index]
            )
            culture.grid.add_cell_to_hash_table(
                cell_index, culture.cell_positions[cell_index]
            )


def grid_signature(culture):
    """Capture bucket keys and ordered membership for exact comparisons."""
    return [(key, tuple(indices)) for key, indices in culture.grid.hash_table.items()]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("seed", [17, 29, 43])
def test_vectorized_movement_matches_original_bitwise(dimension, seed):
    old = make_minimal_culture(dimension, seed)
    new = make_minimal_culture(dimension, seed)
    rng = np.random.default_rng(seed + 999)
    for step in range(12):
        dp = np.zeros_like(old.cell_positions)
        dp[:, :2] = rng.uniform(-4, 4, size=(len(dp), 2))
        if step % 3 == 0:
            dp[:3, 0] += 12.0  # periodic wrapping
        da = rng.uniform(-0.8, 0.8, size=len(old.cell_phies))
        original_move(old, dp, da)
        new.move(dp, da)
        assert old.cell_positions.tobytes() == new.cell_positions.tobytes()
        assert old.cell_phies.tobytes() == new.cell_phies.tobytes()
        assert old.nematic_tensors.tobytes() == new.nematic_tensors.tobytes()
        assert grid_signature(old) == grid_signature(new)


def test_vectorized_movement_rejects_nonplanar_2d():
    old = make_minimal_culture(2, 17)
    new = make_minimal_culture(2, 17)
    dp = np.zeros_like(old.cell_positions)
    dp[0, 2] = 0.25
    da = np.zeros_like(old.cell_phies)
    with pytest.raises(ValueError, match="requires z=0"):
        original_move(old, dp, da)
    with pytest.raises(ValueError, match="requires z=0"):
        new.move(dp, da)
