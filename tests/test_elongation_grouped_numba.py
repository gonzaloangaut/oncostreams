"""Bitwise regression tests for optional grouped_early_numba round elongation.

Run: python -m pytest -q tests/test_elongation_grouped_numba.py

The reference is the already validated fast_numba elongation backend. All
other settings, initial conditions, and RNG seeds are identical.
"""
import math
import pickle

import numpy as np
import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import (
    OutputDemux,
    DatOutput_order_parameters,
    DatOutput_position_aspectratio,
)
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


def build_culture(
    backend, *, seed=17, n=64, rho=0.5, dimension=3,
    attempts=10, eta=0.033, output_dir=None,
):
    side = math.sqrt(n * math.pi / rho)
    threshold = math.exp(-1)
    ar_max = 2.7
    grid = SpatialHashGrid(
        culture=None, bounds=side,
        cube_size=2 * math.sqrt(ar_max * (-math.log(threshold))),
        torus=True, spatial_dimension=dimension,
    )
    force = Anisotropic_Grosmann(
        kRep=2.5, bExp=3, noise_eta=eta,
        shrinking=True, lambda_core=1.0,
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
        force=force, grid=grid, rng_seed=seed,
        initial_number_of_cells=n, initial_fraction_elongated=0.5,
        reproduction=False, movement=True, deformation=True,
        stabilization_time=0, overlap_threshold_ratio=threshold,
        delta_t=0.005, deformation_attempt_period=0.005,
        aspect_ratio_max=ar_max, trabajo_final=True,
        cell_max_def_attempts=attempts,
        interaction_backend='batch_cached_noise_numba',
        elongation_backend=backend,
    )
    culture.overlap_backend = 'numba'
    culture.relative_positions_backend = 'numba'
    return culture


def run(culture, start, end):
    culture.simulate(num_times=end, start_tic=start, checkpoint_path=None)


def compare_bitwise(a, b):
    for attr in (
        'cell_positions', 'cell_phies', 'nematic_tensors',
        'cell_instantaneous_velocities',
    ):
        x = getattr(a, attr)
        y = getattr(b, attr)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes(), attr
    for attr in (
        'aspect_ratio', 'anisotropy', 'squared_diagonal',
        'is_round', 'shrink',
    ):
        assert [getattr(c, attr) for c in a.cells] == [
            getattr(c, attr) for c in b.cells
        ], attr
    assert repr(a.rng.bit_generator.state) == repr(b.rng.bit_generator.state)
    assert a.deformation_event_counts == b.deformation_event_counts
    assert [(key, list(bucket)) for key, bucket in a.grid.hash_table.items()] == [
        (key, list(bucket)) for key, bucket in b.grid.hash_table.items()
    ]
    for attr in (
        'max_normalized_overlap_interval',
        'normalized_overlap_sum_interval',
        'normalized_overlap_count_interval',
    ):
        assert getattr(a, attr) == getattr(b, attr), attr


@pytest.mark.parametrize('rho', [0.5, 0.84])
@pytest.mark.parametrize('dimension', [2, 3])
def test_grouped_elongation_matches_fast_numba_every_step(rho, dimension):
    baseline = build_culture('fast_numba', rho=rho, dimension=dimension)
    candidate = build_culture('grouped_early_numba', rho=rho, dimension=dimension)
    for tic in range(1, 27):
        run(baseline, tic - 1, tic)
        run(candidate, tic - 1, tic)
        compare_bitwise(baseline, candidate)


def test_accepts_actual_round_elongations_without_changing_choices():
    # Sparse culture makes accepted orientations common; the dense tests above
    # predominantly exercise rejected proposals.
    baseline = build_culture('fast_numba', rho=0.22, n=64, seed=17)
    candidate = build_culture('grouped_early_numba', rho=0.22, n=64, seed=17)
    run(baseline, 0, 10)
    run(candidate, 0, 10)
    compare_bitwise(baseline, candidate)
    assert baseline.deformation_event_counts['round_elongation_successes'] > 0


def test_checkpoint_resume_matches_reference(tmp_path):
    baseline = build_culture('fast_numba', seed=42, n=64, rho=0.5)
    optimized = build_culture('grouped_early_numba', seed=42, n=64, rho=0.5)
    run(baseline, 0, 45)
    run(optimized, 0, 17)
    checkpoint = tmp_path / 'grouped_elongation_checkpoint.pkl'
    optimized.save_checkpoint(str(checkpoint), tic=17)
    with checkpoint.open('rb') as f:
        restored, start, saved_rng_state = pickle.load(f)
    assert start == 17
    assert restored.elongation_backend == 'grouped_early_numba'
    # Interaction candidate caches must not be serialized; rebuild lazily.
    assert getattr(restored.grid, '_batch_neighbor_cache', None) is None
    restored.rng.bit_generator.state = saved_rng_state
    run(restored, start, 45)
    compare_bitwise(baseline, restored)


def test_dat_outputs_are_byte_identical(tmp_path):
    ref_dir = tmp_path / 'reference'
    new_dir = tmp_path / 'grouped'
    baseline = build_culture('fast_numba', seed=23, n=64, rho=0.50,
                             output_dir=ref_dir)
    optimized = build_culture('grouped_early_numba', seed=23, n=64, rho=0.50,
                              output_dir=new_dir)
    run(baseline, 0, 26)
    run(optimized, 0, 26)
    compare_bitwise(baseline, optimized)

    def dat_files(root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob('*.dat')}

    reference = dat_files(ref_dir)
    candidate = dat_files(new_dir)
    assert reference, 'The test must produce .dat files'
    assert reference == candidate


def test_zero_attempts_does_not_consume_rng():
    a = build_culture('fast_numba', attempts=0)
    b = build_culture('grouped_early_numba', attempts=0)
    run(a, 0, 3)
    run(b, 0, 3)
    compare_bitwise(a, b)
