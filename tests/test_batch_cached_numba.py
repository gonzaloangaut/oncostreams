"""Regression tests for the *optional* ordered candidate-list cache backend.

Run after installation: pytest -q tests/test_batch_cached_numba.py
"""
from __future__ import annotations

import pickle

import pytest

from tests._helpers import (
    assert_cultures_bitwise_equal,
    make_batch_culture,
    run_steps,
)


def build_culture(backend, *, seed=17, dimension=3, output_dir=None):
    return make_batch_culture(
        backend, seed=seed, dimension=dimension, out=output_dir,
    )


_run = run_steps


def _equal_state(a, b):
    # Preserve the original cache-test assertion scope (no overlap counters).
    assert_cultures_bitwise_equal(a, b, check_overlap_statistics=False)


def _check_candidate_lists(culture):
    cache = culture.grid._batch_neighbor_cache
    assert cache is not None
    for i in range(len(culture.cells)):
        pos = culture.cell_positions[i]
        candidates = [int(j) for j in culture.grid.find_neighbors(pos) if int(j) != i]
        expected = tuple(sorted(j for j in candidates if j < i)
                         + [j for j in candidates if j > i])
        assert cache.candidates(culture.grid, i, pos) == expected


@pytest.mark.parametrize('dimension', [2, 3])
def test_cached_batch_bitwise_and_order(dimension):
    baseline = build_culture('batch_numba', dimension=dimension)
    optimized = build_culture('batch_cached_numba', dimension=dimension)
    # Compare every step, not only the final state.
    for tic in range(1, 23):
        _run(baseline, tic - 1, tic)
        _run(optimized, tic - 1, tic)
        _equal_state(baseline, optimized)
    _check_candidate_lists(optimized)
    assert optimized.grid._batch_neighbor_cache.hits > 0


def test_checkpoint_round_trip_recreates_cache_without_changing_trajectory(tmp_path):
    baseline = build_culture('batch_numba', seed=42)
    optimized = build_culture('batch_cached_numba', seed=42)
    _run(baseline, 0, 24)
    _run(optimized, 0, 11)
    assert optimized.grid._batch_neighbor_cache is not None

    # Test the model's own atomic checkpoint writer and the same payload shape
    # used by Simulation's checkpoint recovery path.
    checkpoint = tmp_path / 'culture.pkl'
    optimized.save_checkpoint(str(checkpoint), tic=11)
    with checkpoint.open('rb') as stream:
        restored, tic, state = pickle.load(stream)
    assert tic == 11
    assert restored.grid._batch_neighbor_cache is None
    restored.rng.bit_generator.state = state
    _run(restored, tic, 24)
    _equal_state(baseline, restored)
    _check_candidate_lists(restored)


def test_dat_outputs_are_byte_identical(tmp_path):
    baseline_dir = tmp_path / 'baseline'
    optimized_dir = tmp_path / 'cached'
    baseline = build_culture('batch_numba', seed=31, output_dir=baseline_dir)
    optimized = build_culture('batch_cached_numba', seed=31, output_dir=optimized_dir)
    _run(baseline, 0, 16)
    _run(optimized, 0, 16)
    _equal_state(baseline, optimized)

    def outputs(root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob('*.dat')}

    b_files = outputs(baseline_dir)
    c_files = outputs(optimized_dir)
    assert b_files, 'The test must actually create .dat files'
    assert b_files == c_files
