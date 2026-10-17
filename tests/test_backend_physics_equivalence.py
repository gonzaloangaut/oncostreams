"""Scientific equivalence checks between reference and production backends.

Compare the SAME initial microstate and RNG state before advancing. These
checks are independent of checkpoint correctness and of the frozen Numba
trajectory fixture. They deliberately do not require identical floating-point
results across NumPy and Numba.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import fixed_n_grosmann_culture
from tumorsphere.core.interaction_batch_numba import batch_interaction_step

pytest.importorskip("numba")


def paired_cultures(*, seed, rho, dimension, attempts=1, eta=None, d_phi=None,
                    lambda_core=1.0):
    """Two independent cultures with identical initial positions and RNG."""
    options = dict(
        interaction_backend="batch_cached_noise_numba",
        elongation_backend="grouped_early_numba",
        seed=seed,
        n=64,
        rho=rho,
        dimension=dimension,
        eta=eta,
        d_phi=d_phi,
        attempts=attempts,
        deformation_period=0.005,
        output_dir=None,
    )
    reference = fixed_n_grosmann_culture(**options)
    optimized = fixed_n_grosmann_culture(**options)

    # Initialize cells, grid, orientation tensors, and RNG identically, without
    # any movement or deformation. Do not consume new draws when switching.
    reference.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    optimized.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    np.testing.assert_array_equal(reference.cell_positions, optimized.cell_positions)
    np.testing.assert_array_equal(reference.cell_phies, optimized.cell_phies)
    assert reference.rng.bit_generator.state == optimized.rng.bit_generator.state

    reference.backend = "numpy"
    reference.force.force_backend = "numpy"
    reference.overlap_backend = "numpy"
    reference.relative_positions_backend = "numpy"
    reference.interaction_backend = "cached"
    reference.elongation_backend = "standard"
    optimized.backend = "numba"

    reference.force.lambda_core = lambda_core
    optimized.force.lambda_core = lambda_core
    return reference, optimized


def ordered_grid(culture):
    return [(key, tuple(bucket)) for key, bucket in culture.grid.hash_table.items()]


@pytest.mark.parametrize("seed,rho,dimension,lambda_core", [
    (17, 0.22, 2, 0.0),
    (23, 0.84, 3, 1.0),
    (42, 0.90, 2, 1.0),
    (29, 0.84, 2, 0.35),
])
@pytest.mark.parametrize("eta,d_phi", [
    (None, None),
    (0.033, None),
    (None, 0.13),
    (0.033, 0.13),
])
def test_same_microstate_has_equivalent_interaction_sweep(
    seed, rho, dimension, lambda_core, eta, d_phi,
):
    """Reference per-cell and production batch forces/torques/noise agree.

    This isolates interactions from feedback, grid mutations and deformation;
    it compares state increments, contraction decisions and RNG consumption.
    """
    ref, opt = paired_cultures(
        seed=seed, rho=rho, dimension=dimension,
        eta=eta, d_phi=d_phi, lambda_core=lambda_core,
    )
    n = len(ref.cells)
    displacement_ref = np.zeros((n, 3), dtype=np.float64)
    rotation_ref = np.zeros(n, dtype=np.float64)
    for i in ref.active_cell_indexes:
        displacement_ref[i], rotation_ref[i] = ref.interaction(
            cell_index=i, delta_t=ref.delta_t,
        )

    displacement_opt, rotation_opt = batch_interaction_step(opt, opt.delta_t)

    assert np.all(np.isfinite(displacement_ref))
    assert np.all(np.isfinite(displacement_opt))
    assert np.all(np.isfinite(rotation_ref))
    assert np.all(np.isfinite(rotation_opt))
    np.testing.assert_allclose(
        displacement_opt, displacement_ref, rtol=1e-9, atol=2e-11,
        err_msg="Different displacements from the same microstate",
    )
    np.testing.assert_allclose(
        rotation_opt, rotation_ref, rtol=1e-9, atol=2e-11,
        err_msg="Different angular increments from the same microstate",
    )
    assert [c.shrink for c in ref.cells] == [c.shrink for c in opt.cells]
    assert ref.rng.bit_generator.state == opt.rng.bit_generator.state


@pytest.mark.parametrize("seed,rho,dimension,attempts", [
    (17, 0.22, 2, 1),
    (42, 0.22, 3, 10),
    (23, 0.84, 3, 1),
    (29, 0.84, 2, 5),
    (37, 0.90, 2, 10),
])
def test_same_microstate_has_equivalent_round_elongation_choices(
    seed, rho, dimension, attempts,
):
    """Proposal acceptance, selected geometry and RNG match the reference.

    Check consecutive cells, so accepted deformations can affect later
    candidate sets and ordered grid membership.
    """
    ref, opt = paired_cultures(
        seed=seed, rho=rho, dimension=dimension, attempts=attempts,
    )
    round_indices = [i for i, cell in enumerate(ref.cells) if cell.is_round][:6]
    assert len(round_indices) == 6
    for i in round_indices:
        accepted_ref = ref.elongate_from_round(i)
        accepted_opt = opt.elongate_from_round(i)
        assert accepted_ref == accepted_opt, (
            f"Elongation acceptance differs for seed={seed}, rho={rho}, "
            f"attempts={attempts}, cell={i}"
        )
        np.testing.assert_allclose(
            opt.cell_positions, ref.cell_positions, rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            opt.cell_phies, ref.cell_phies, rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            [cell.aspect_ratio for cell in opt.cells],
            [cell.aspect_ratio for cell in ref.cells], rtol=0, atol=0,
        )
        assert ref.rng.bit_generator.state == opt.rng.bit_generator.state
        assert ordered_grid(ref) == ordered_grid(opt)
