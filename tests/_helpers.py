"""Reusable fixtures for fixed-N Grosmann backend regression tests.

Keep these helpers independent of individual test modules. They intentionally
preserve the constructor arguments and exact bytewise assertions used before
this refactor; they are not part of the production API.
"""
from __future__ import annotations

import math

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import (
    DatOutput_order_parameters,
    DatOutput_position_aspectratio,
    OutputDemux,
)
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


def fixed_n_grosmann_culture(
    *, interaction_backend, elongation_backend, seed, n, rho,
    dimension, eta, d_phi, attempts, deformation_period, output_dir,
):
    """Create the exact model configuration used by the backend tests."""
    threshold = math.exp(-1)
    ar_max = 2.7
    side = math.sqrt(n * math.pi / rho)
    grid = SpatialHashGrid(
        culture=None, bounds=side,
        cube_size=2 * math.sqrt(ar_max * (-math.log(threshold))),
        torus=True, spatial_dimension=dimension,
    )
    force = Anisotropic_Grosmann(
        kRep=2.5, bExp=3, noise_eta=eta, d_phi=d_phi,
        shrinking=True, lambda_core=1.0,
    )
    force.force_backend = 'numba'
    outputs = [] if output_dir is None else [
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
        deformation_attempt_period=deformation_period,
        aspect_ratio_max=ar_max,
        trabajo_final=True,
        cell_max_def_attempts=attempts,
        interaction_backend=interaction_backend,
        elongation_backend=elongation_backend,
    )
    culture.overlap_backend = 'numba'
    culture.relative_positions_backend = 'numba'
    return culture


def make_batch_culture(
    backend, *, seed=17, dimension=3, n=64, eta=0.033,
    d_phi=None, out=None, rho=0.84,
):
    """Use the final grouped-elongation implementation in interaction tests."""
    return fixed_n_grosmann_culture(
        interaction_backend=backend, elongation_backend='grouped_early_numba',
        seed=seed, n=n, rho=rho, dimension=dimension,
        eta=eta, d_phi=d_phi, attempts=1,
        deformation_period=0.01, output_dir=out,
    )


def make_elongation_culture(
    backend, *, seed=17, n=64, rho=0.5, dimension=3,
    attempts=10, eta=0.033, output_dir=None,
):
    """Preserve the existing grouped-elongation test defaults."""
    return fixed_n_grosmann_culture(
        interaction_backend='batch_cached_noise_numba',
        elongation_backend=backend, seed=seed, n=n, rho=rho,
        dimension=dimension, eta=eta, d_phi=None, attempts=attempts,
        deformation_period=0.005, output_dir=output_dir,
    )


def run_steps(culture, start, end):
    """Advance a culture, including t=0 initialization when start=end=0."""
    culture.simulate(num_times=end, start_tic=start, checkpoint_path=None)


def assert_cultures_bitwise_equal(a, b):
    """Compare physical state, cell state, ordered grid, and RNG exactly."""
    for attr in (
        'cell_positions', 'cell_phies', 'nematic_tensors',
        'cell_instantaneous_velocities',
    ):
        x, y = getattr(a, attr), getattr(b, attr)
        assert x.dtype == y.dtype and x.shape == y.shape, attr
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
