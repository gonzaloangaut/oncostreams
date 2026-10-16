"""Bitwise regression tests for optional grouped_early_numba round elongation.

Run: python -m pytest -q tests/test_elongation_grouped_numba.py

The reference is the already validated fast_numba elongation backend. All
other settings, initial conditions, and RNG seeds are identical.
"""
import pickle

import numpy as np
import pytest

from tests._helpers import (
    assert_cultures_bitwise_equal,
    make_elongation_culture,
    run_steps,
)


# Preserve the public helper names imported by the cache regression tests.
build_culture = make_elongation_culture
run = run_steps
compare_bitwise = assert_cultures_bitwise_equal


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

@pytest.mark.parametrize('seed', [17, 42])
@pytest.mark.parametrize('attempts', [1, 5, 10])
def test_proposal_angles_bulk_rng_matches_scalar(seed, attempts):
    """One sweep consumes the same random values and leaves identical RNG state."""
    scalar = np.random.default_rng(seed)
    bulk = np.random.default_rng(seed)
    expected = np.array([
        scalar.uniform(low=0, high=2 * np.pi) for _ in range(attempts)
    ])
    actual = bulk.uniform(low=0, high=2 * np.pi, size=attempts)
    np.testing.assert_array_equal(actual, expected)
    assert repr(scalar.bit_generator.state) == repr(bulk.bit_generator.state)


@pytest.mark.parametrize('attempts', [1, 5, 10])
def test_grouped_bulk_angle_rng_preserves_full_trajectory(attempts):
    """The optimized grouped path agrees bitwise with scalar-RNG fast_numba."""
    reference = build_culture('fast_numba', attempts=attempts, n=64, rho=0.84)
    optimized = build_culture('grouped_early_numba', attempts=attempts,
                              n=64, rho=0.84)
    for tic in range(1, 17):
        run(reference, tic - 1, tic)
        run(optimized, tic - 1, tic)
        compare_bitwise(reference, optimized)


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
