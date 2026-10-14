"""Regression tests for the *optional* ordered candidate-list cache backend.

Run after installation: pytest -q tests/test_batch_cached_numba.py
"""
from __future__ import annotations

import math
import pickle

import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import (
    OutputDemux,
    DatOutput_order_parameters,
    DatOutput_position_aspectratio,
)
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


def build_culture(backend, *, seed=17, dimension=3, output_dir=None):
    n = 64
    rho = 0.84
    side = math.sqrt(n * math.pi / rho)
    threshold = math.exp(-1)
    ar_max = 2.7
    grid = SpatialHashGrid(
        culture=None,
        bounds=side,
        cube_size=2 * math.sqrt(ar_max * (-math.log(threshold))),
        torus=True,
        spatial_dimension=dimension,
    )
    force = Anisotropic_Grosmann(
        kRep=2.5, bExp=3, noise_eta=0.033, shrinking=True, lambda_core=1.0
    )
    force.force_backend = 'numba'
    outputs = []
    if output_dir is not None:
        outputs = [
            DatOutput_order_parameters('identical', str(output_dir), save_step=4),
            DatOutput_position_aspectratio('identical', str(output_dir), save_step=4),
        ]
    culture = Culture(
        output=OutputDemux('identical', outputs),
        force=force,
        grid=grid,
        rng_seed=seed,
        initial_number_of_cells=n,
        initial_fraction_elongated=0.5,
        reproduction=False,
        movement=True,
        deformation=True,
        stabilization_time=0,
        overlap_threshold_ratio=threshold,
        delta_t=0.005,
        deformation_attempt_period=0.01,
        aspect_ratio_max=ar_max,
        trabajo_final=True,
        cell_max_def_attempts=1,
        interaction_backend=backend,
        elongation_backend='fast_numba',
    )
    culture.overlap_backend = 'numba'
    culture.relative_positions_backend = 'numba'
    return culture


def _run(culture, start, end):
    culture.simulate(num_times=end, start_tic=start, checkpoint_path=None)


def _equal_state(a, b):
    for attr in ('cell_positions', 'cell_phies', 'nematic_tensors',
                 'cell_instantaneous_velocities'):
        x, y = getattr(a, attr), getattr(b, attr)
        assert x.dtype == y.dtype
        assert x.shape == y.shape
        assert x.tobytes() == y.tobytes(), attr
    for attr in ('aspect_ratio', 'anisotropy', 'squared_diagonal', 'is_round', 'shrink'):
        assert [getattr(c, attr) for c in a.cells] == [getattr(c, attr) for c in b.cells], attr
    assert repr(a.rng.bit_generator.state) == repr(b.rng.bit_generator.state)
    assert a.deformation_event_counts == b.deformation_event_counts
    assert [(key, list(v)) for key, v in a.grid.hash_table.items()] == [
        (key, list(v)) for key, v in b.grid.hash_table.items()
    ]


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
