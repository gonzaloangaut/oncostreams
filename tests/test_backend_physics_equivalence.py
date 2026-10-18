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
    reference = fixed_n_grosmann_culture(backend="numpy", **options)
    optimized = fixed_n_grosmann_culture(backend="numba", **options)

    # Initialize cells, grid, orientation tensors and RNG identically,
    # without movement or deformation. No backend mutation is needed.
    reference.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    optimized.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    np.testing.assert_array_equal(reference.cell_positions, optimized.cell_positions)
    np.testing.assert_array_equal(reference.cell_phies, optimized.cell_phies)
    assert reference.rng.bit_generator.state == optimized.rng.bit_generator.state

    reference.force.lambda_core = lambda_core
    optimized.force.lambda_core = lambda_core
    return reference, optimized


def ordered_grid(culture):
    """Capture bucket membership and iteration order for exact comparisons."""
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
# ---------------------------------------------------------------------------
# Controlled two-cell tests at physical decision boundaries.
# These exercise the actual production NumPy and Numba dispatch paths.
# ---------------------------------------------------------------------------


def _threshold_pair(backend, *, seed=17, lambda_core=1.0):
    """Initialize two cells and reduce both to round, noiseless cells."""
    culture = fixed_n_grosmann_culture(
        backend=backend, seed=seed, n=2, rho=0.02, dimension=2,
        eta=None, d_phi=None, attempts=1, deformation_period=0.005,
        output_dir=None,
    )
    culture.simulate(num_times=0, start_tic=0, checkpoint_path=None)
    assert len(culture.cells) == 2
    target = next(i for i, cell in enumerate(culture.cells) if cell.is_round)
    other = 1 - target
    culture.cells[other].set_aspect_ratio(1.0)
    culture.cell_phies[:] = 0.0
    culture.update_nematic_tensors()
    culture.force.lambda_core = lambda_core
    return culture, target, other


def _relocate_pair(culture, target, other, relative_displacement):
    """Move the pair coherently, including grid membership and pair caches."""
    for index in (target, other):
        culture.grid.remove_cell_from_hash_table(
            index, culture.cell_positions[index].copy()
        )
    middle = 0.5 * culture.side
    target_position = np.array([middle, middle, 0.0], dtype=float)
    culture.cell_positions[target] = target_position
    culture.cell_positions[other] = np.mod(
        target_position + np.asarray(relative_displacement, dtype=float),
        culture.side,
    )
    for index in (target, other):
        culture.grid.add_cell_to_hash_table(
            index, culture.cell_positions[index]
        )
        culture.cells[index].neighbors_relative_pos.clear()
        culture.cells[index].neighbors_overlap.clear()
        culture.cells[index].neighbors_normalized_overlap.clear()


def _compare_pair_interactions(reference, optimized, target):
    """Compare full reference neighbor selection with production batch sweep."""
    from tumorsphere.core.interaction_batch_numba import batch_interaction_step

    reference_displacement, reference_rotation = reference.interaction(
        target, reference.delta_t,
    )
    optimized_displacements, optimized_rotations = batch_interaction_step(
        optimized, optimized.delta_t,
    )
    np.testing.assert_allclose(
        optimized_displacements[target], reference_displacement,
        rtol=1e-9, atol=2e-11,
    )
    np.testing.assert_allclose(
        optimized_rotations[target], reference_rotation,
        rtol=1e-9, atol=2e-11,
    )
    assert np.all(np.isfinite(reference_displacement))
    assert np.all(np.isfinite(optimized_displacements))
    assert np.all(np.isfinite(optimized_rotations))
    assert reference.rng.bit_generator.state == optimized.rng.bit_generator.state
    return reference_displacement, optimized_displacements[target]


@pytest.mark.parametrize("factor", [1.0 - 1e-5, 1.0 + 1e-5])
def test_two_round_cells_cross_interaction_threshold(factor):
    """Interaction uses strict xi > exp(-1) in BOTH production backends."""
    reference, i, j = _threshold_pair("numpy")
    optimized, i_opt, j_opt = _threshold_pair("numba")
    assert (i, j) == (i_opt, j_opt)

    threshold = reference.overlap_threshold_ratio
    xi = threshold * factor
    # For two round cells of area pi, normalized overlap is exp(-r**2/4).
    separation = 2.0 * np.sqrt(-np.log(xi))
    for culture in (reference, optimized):
        _relocate_pair(culture, i, j, [separation, 0.0, 0.0])

    actual_neighbors = reference._get_significant_neighbors(
        i, update_overlap_diagnostic=False,
    )
    measured_xi = reference.cells[i].neighbors_normalized_overlap[j]
    np.testing.assert_allclose(measured_xi, xi, rtol=1e-12, atol=1e-14)
    should_interact = factor > 1.0
    assert (j in actual_neighbors) == should_interact

    d_ref, d_opt = _compare_pair_interactions(reference, optimized, i)
    assert (np.linalg.norm(d_ref) > 0.0) == should_interact
    assert (np.linalg.norm(d_opt) > 0.0) == should_interact


@pytest.mark.parametrize("lambda_core", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("xi", [1.0 - 1e-3, 1.0 - 1e-6, 1.0])
def test_core_regularization_near_unity_is_finite_and_equivalent(
    lambda_core, xi,
):
    """Near xi=1, the force amplification is finite and matches both paths."""
    reference, i, j = _threshold_pair("numpy", lambda_core=lambda_core)
    optimized, i_opt, j_opt = _threshold_pair("numba", lambda_core=lambda_core)
    assert (i, j) == (i_opt, j_opt)
    separation = 2.0 * np.sqrt(-np.log(xi))
    for culture in (reference, optimized):
        _relocate_pair(culture, i, j, [separation, 0.0, 0.0])

    amplification = reference.force.calculate_core_amplification(xi ** 3.0)
    assert np.isfinite(amplification)
    assert amplification >= 1.0
    if lambda_core == 0.0:
        assert amplification == 1.0

    _compare_pair_interactions(reference, optimized, i)


@pytest.mark.parametrize("factor", [1.0 - 1e-5, 1.0 + 1e-5])
@pytest.mark.parametrize("seed", [17, 42])
def test_marginal_round_elongation_decision_matches_backends(seed, factor):
    """One geometrically controlled proposal is accepted iff xi <= threshold."""
    from copy import deepcopy

    reference, i, j = _threshold_pair("numpy", seed=seed)
    optimized, i_opt, j_opt = _threshold_pair("numba", seed=seed)
    assert (i, j) == (i_opt, j_opt)
    assert reference.rng.bit_generator.state == optimized.rng.bit_generator.state

    # Read the next proposal angle without consuming either culture's RNG.
    probe = np.random.default_rng()
    probe.bit_generator.state = deepcopy(reference.rng.bit_generator.state)
    phi = probe.uniform(low=0.0, high=2 * np.pi)
    direction = np.array([np.cos(phi), np.sin(phi), 0.0])
    new_aspect = reference.aspect_ratio_max
    displacement = np.sqrt(new_aspect) - 1.0  # area=pi, old aspect=1

    # The other cell is round; along the proposed major axis,
    # xi = exp(-coefficient * separation**2). Derive that coefficient
    # analytically rather than by querying either implementation.
    di = new_aspect + 1.0 / new_aspect
    ei = (new_aspect**2 - 1.0) / (new_aspect**2 + 1.0)
    dj = 2.0
    ds = di + dj
    beta = ds**2 - (di * ei)**2
    parallel_weight = di * ei / ds
    coefficient = ds / beta * (1.0 - parallel_weight)
    target_xi = reference.overlap_threshold_ratio * factor
    separation = np.sqrt(-np.log(target_xi) / coefficient)
    np.testing.assert_allclose(
        np.exp(-coefficient * separation**2), target_xi,
        rtol=1e-12, atol=1e-14,
    )
    for culture in (reference, optimized):
        _relocate_pair(
            culture, i, j, (displacement + separation) * direction,
        )
        proposed_position = np.mod(
            culture.cell_positions[i] + displacement * direction, culture.side
        )
        assert j in list(culture.grid.find_neighbors(position=proposed_position))

    original_positions = reference.cell_positions.copy()
    success_numpy = reference.elongate_from_round(i)
    success_numba = optimized.elongate_from_round(i)
    expected_success = factor < 1.0
    assert success_numpy == success_numba == expected_success
    np.testing.assert_allclose(
        optimized.cell_positions, reference.cell_positions, rtol=0, atol=2e-12,
    )
    np.testing.assert_allclose(
        optimized.cell_phies, reference.cell_phies, rtol=0, atol=2e-12,
    )
    assert [c.aspect_ratio for c in reference.cells] == [
        c.aspect_ratio for c in optimized.cells
    ]
    if expected_success:
        assert not reference.cells[i].is_round
        assert reference.cells[i].aspect_ratio == new_aspect
        assert not np.array_equal(reference.cell_positions, original_positions)
    else:
        assert reference.cells[i].is_round
        np.testing.assert_array_equal(reference.cell_positions, original_positions)
    assert reference.rng.bit_generator.state == optimized.rng.bit_generator.state
    assert ordered_grid(reference) == ordered_grid(optimized)
