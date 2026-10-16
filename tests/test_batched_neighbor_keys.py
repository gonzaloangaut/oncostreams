"""Verify vectorized spatial keys preserve the exact ordered neighbor CSR."""
from copy import deepcopy

import numpy as np
import pytest

from tests._helpers import make_batch_culture as make_culture, run_steps as run


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("rho", [0.30, 0.84, 0.90])
@pytest.mark.parametrize("eta", [0.0, 0.033])
def test_ordered_csr_matches_scalar_candidate_queries(dimension, rho, eta):
    culture = make_culture(
        'batch_cached_noise_numba', seed=17, dimension=dimension,
        rho=rho, eta=eta, n=64,
    )
    run(culture, 0, 12)
    grid = culture.grid
    assert grid._batch_neighbor_cache is not None

    # These caches start with the same entries and invalidation versions.
    fast = deepcopy(grid._batch_neighbor_cache)
    reference = deepcopy(grid._batch_neighbor_cache)

    # The vectorized keys must be byte-for-byte identical in both dimensions.
    buckets = np.floor(
        culture.cell_positions[:, :dimension] / grid.cube_size
    ).astype(int)
    assert all(
        bucket.tobytes() == grid.get_hash_key(position)
        for bucket, position in zip(buckets, culture.cell_positions)
    )

    ptr, neighbors = fast.ordered_csr(grid, culture.cell_positions)
    ref_ptr = [0]
    ref_neighbors = []
    for index, position in enumerate(culture.cell_positions):
        ref_neighbors.extend(reference.candidates(grid, index, position))
        ref_ptr.append(len(ref_neighbors))

    assert ptr == ref_ptr
    assert neighbors == ref_neighbors
    assert (fast.hits, fast.misses) == (reference.hits, reference.misses)
    assert fast._versions == reference._versions
    assert fast._entries == reference._entries


def test_ordered_csr_after_grid_membership_change():
    culture = make_culture(
        'batch_cached_noise_numba', seed=29, dimension=2,
        rho=0.84, eta=0.0, n=64,
    )
    run(culture, 0, 10)
    grid = culture.grid
    position = culture.cell_positions[0].copy()

    # Remove/add invalidates nearby entries, even when position is unchanged.
    grid.remove_cell_from_hash_table(0, position)
    grid.add_cell_to_hash_table(0, position)
    fast = deepcopy(grid._batch_neighbor_cache)
    reference = deepcopy(grid._batch_neighbor_cache)

    ptr, neighbors = fast.ordered_csr(grid, culture.cell_positions)
    ref_ptr = [0]
    ref_neighbors = []
    for index, pos in enumerate(culture.cell_positions):
        ref_neighbors.extend(reference.candidates(grid, index, pos))
        ref_ptr.append(len(ref_neighbors))
    assert (ptr, neighbors) == (ref_ptr, ref_neighbors)
    assert (fast.hits, fast.misses) == (reference.hits, reference.misses)
    assert fast._entries == reference._entries
