"""Final grouped elongation regressions and independent kernel checks."""
import pickle

import numpy as np
import pytest

from tumorsphere.core.elongation_grouped_numba import (
    _cached_elongation_neighbors, _shared_elongation_geometry,
)

from tests._helpers import (
    make_elongation_culture as build_culture,
    assert_cultures_bitwise_equal as compare_bitwise,
    run_steps as run,
)


@pytest.mark.parametrize("rho,dimension,attempts", [
    (0.22, 2, 1), (0.84, 3, 10), (0.90, 2, 5),
])
def test_grouped_elongation_determinism(rho, dimension, attempts):
    a = build_culture("numba", rho=rho,
                      dimension=dimension, attempts=attempts)
    b = build_culture("numba", rho=rho,
                      dimension=dimension, attempts=attempts)
    for tic in range(1, 21):
        run(a, tic - 1, tic)
        run(b, tic - 1, tic)
        compare_bitwise(a, b)


def test_real_elongation_successes_are_recorded():
    culture = build_culture("numba", rho=0.22, n=64, seed=17)
    run(culture, 0, 10)
    assert culture.deformation_event_counts["round_elongation_successes"] > 0


def test_grouped_checkpoint_resume(tmp_path):
    continuous = build_culture("numba", seed=42, n=64, rho=0.5)
    paused = build_culture("numba", seed=42, n=64, rho=0.5)
    run(continuous, 0, 45)
    run(paused, 0, 17)
    checkpoint = tmp_path / "grouped_elongation.pkl"
    paused.save_checkpoint(str(checkpoint), tic=17)
    with checkpoint.open("rb") as stream:
        restored, start, saved_rng_state = pickle.load(stream)
    assert start == 17
    assert restored.backend == "numba"
    assert getattr(restored.grid, "_batch_neighbor_cache", None) is None
    restored.rng.bit_generator.state = saved_rng_state
    run(restored, start, 45)
    compare_bitwise(continuous, restored)


def test_zero_attempts_matches_standard_path():
    """With no proposals, either elongation backend must do the same thing."""
    reference = build_culture("numpy", attempts=0)
    optimized = build_culture("numba", attempts=0)
    # Isolate deformation: reference and optimized interactions need not
    # produce byte-identical continuous trajectories.
    run(reference, 0, 0)
    run(optimized, 0, 0)
    for index, cell in enumerate(reference.cells):
        if cell.is_round:
            assert not reference.elongate_from_round(index)
            assert not optimized.elongate_from_round(index)
    compare_bitwise(reference, optimized)
    assert optimized.deformation_event_counts["round_elongation_successes"] == 0


@pytest.mark.parametrize('seed', [17, 42])
@pytest.mark.parametrize('attempts', [1, 5, 10])
def test_proposal_angles_bulk_rng_matches_scalar(seed, attempts):
    """Bulk generation must reproduce scalar draws and RNG state."""
    scalar = np.random.default_rng(seed)
    bulk = np.random.default_rng(seed)
    expected = np.array([
        scalar.uniform(low=0, high=2 * np.pi) for _ in range(attempts)
    ])
    actual = bulk.uniform(low=0, high=2 * np.pi, size=attempts)
    np.testing.assert_array_equal(actual, expected)
    assert repr(scalar.bit_generator.state) == repr(bulk.bit_generator.state)


@pytest.mark.parametrize('attempts', [1, 5, 10, 127])
def test_compiled_proposal_geometry_is_bitwise_identical(attempts):
    """Proposals preserve scalar NumPy results, including periodic crossings."""
    from tumorsphere.core.elongation_grouped_numba import compiled_proposals

    rng = np.random.default_rng(10834 + attempts)
    for side in (5.5, 38.4, 121.9):
        for displacement_length in (0.0, 0.015, 0.25):
            for old_position in (
                np.array([0.0, 0.0, 0.0]),
                np.array([side - 1e-12, 2e-12, 0.0]),
                np.array([side / 2, side / 4, 0.0]),
            ):
                phis = rng.uniform(0.0, 2 * np.pi, size=attempts)
                expected_positions = np.empty((attempts, 3))
                expected_qxx = np.empty(attempts)
                expected_qxy = np.empty(attempts)
                for k, phi in enumerate(phis):
                    expected_positions[k] = np.mod(
                        old_position + np.array([
                            displacement_length * np.cos(phi),
                            displacement_length * np.sin(phi),
                            0.0,
                        ]), side,
                    )
                    expected_qxx[k] = np.cos(2 * phi)
                    expected_qxy[k] = np.sin(2 * phi)
                positions, qxx, qxy = compiled_proposals(
                    old_position, displacement_length, phis, side,
                )
                for expected, actual in (
                    (expected_positions, positions),
                    (expected_qxx, qxx),
                    (expected_qxy, qxy),
                ):
                    assert expected.dtype == actual.dtype
                    assert expected.shape == actual.shape
                    assert expected.tobytes() == actual.tobytes()


# Grouped-elongation cache regressions (formerly test_elongation_reuse_cache.py).

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
    rebuilt = build_culture('numba', n=64, rho=rho,
                            attempts=attempts, dimension=dimension)
    optimized = build_culture('numba', n=64, rho=rho,
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
    culture = build_culture('numba', n=64, rho=0.5)
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
    continuous = build_culture('numba', n=64, rho=0.84, attempts=5)
    paused = build_culture('numba', n=64, rho=0.84, attempts=5)
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
