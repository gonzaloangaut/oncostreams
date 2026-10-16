"""Regression tests for transient grouped-elongation caches."""
import pickle

import numpy as np
import pytest

from tests._helpers import (
    make_elongation_culture as build_culture,
    assert_cultures_bitwise_equal as compare_bitwise,
    run_steps as run,
)
from tumorsphere.core.elongation_grouped_numba import (
    _cached_elongation_neighbors, _shared_elongation_geometry,
)


def assert_geometry_cache_current(culture):
    cache = getattr(culture.grid, '_elongation_geometry_cache', None)
    if cache is not None:
        np.testing.assert_array_equal(
            cache[0], np.asarray([c.squared_diagonal for c in culture.cells]))
        np.testing.assert_array_equal(
            cache[1], np.asarray([c.anisotropy for c in culture.cells]))


def assert_bucket_cache_current(culture):
    # Check bucket membership AND the original ordered iteration sequence.
    cache = getattr(culture.grid, '_elongation_bucket_cache', None)
    if cache:
        for key, ordered in cache.items():
            # A cached key may be inactive if there have been no mutations;
            # reconstruct a point in that bucket without changing the cache.
            bucket = np.frombuffer(key, dtype=np.int64)
            sample = (bucket.astype(float) + 0.5) * culture.grid.cube_size
            expected = tuple(culture.grid.find_neighbors(position=sample))
            assert expected == ordered


@pytest.mark.parametrize('rho', [0.22, 0.84, 0.90])
@pytest.mark.parametrize('attempts', [1, 5, 10])
@pytest.mark.parametrize('dimension', [2, 3])
def test_grouped_caches_preserve_bitwise_steps(rho, attempts, dimension):
    # Compare persistent caches against a run that deliberately rebuilds
    # elongation caches every step. The optimized physics must not change.
    rebuilt = build_culture('grouped_early_numba', n=64, rho=rho,
                            attempts=attempts, dimension=dimension)
    optimized = build_culture('grouped_early_numba', n=64, rho=rho,
                              attempts=attempts, dimension=dimension)
    for tic in range(1, 21):
        rebuilt.grid._elongation_geometry_cache = None
        rebuilt.grid._elongation_bucket_cache = None
        run(rebuilt, tic - 1, tic)
        run(optimized, tic - 1, tic)
        compare_bitwise(rebuilt, optimized)
        assert_geometry_cache_current(optimized)
        assert_bucket_cache_current(optimized)


def test_caches_update_and_invalidate_directly():
    culture = build_culture('grouped_early_numba', n=64, rho=0.5)
    run(culture, 0, 0)
    grid = culture.grid
    geometry = _shared_elongation_geometry(culture)
    cell = culture.cells[0]
    previous = cell.aspect_ratio
    cell.set_aspect_ratio(1.0 if previous > 1.0 else 2.7)
    assert geometry[0][0] == cell.squared_diagonal
    assert geometry[1][0] == cell.anisotropy
    cell.set_aspect_ratio(previous)
    assert_geometry_cache_current(culture)

    position = culture.cell_positions[0].copy()
    key = grid.get_hash_key(position)
    expected = tuple(grid.find_neighbors(position=position))
    assert _cached_elongation_neighbors(grid, key, position) == expected
    assert grid._elongation_bucket_cache
    grid.remove_cell_from_hash_table(0, position)
    assert not grid._elongation_bucket_cache
    grid.add_cell_to_hash_table(0, position)
    assert not grid._elongation_bucket_cache
    refreshed = _cached_elongation_neighbors(grid, key, position)
    assert refreshed == tuple(grid.find_neighbors(position=position))


def test_elongation_caches_not_checkpointed(tmp_path):
    continuous = build_culture('grouped_early_numba', n=64, rho=0.84, attempts=5)
    paused = build_culture('grouped_early_numba', n=64, rho=0.84, attempts=5)
    run(continuous, 0, 30)
    run(paused, 0, 13)
    assert getattr(paused.grid, '_elongation_geometry_cache', None) is not None
    checkpoint = tmp_path / 'reuse.pkl'
    paused.save_checkpoint(str(checkpoint), tic=13)
    with checkpoint.open('rb') as handle:
        restored, start, state = pickle.load(handle)
    assert getattr(restored.grid, '_elongation_geometry_cache', None) is None
    assert getattr(restored.grid, '_elongation_bucket_cache', None) is None
    restored.rng.bit_generator.state = state
    run(restored, start, 30)
    compare_bitwise(continuous, restored)
