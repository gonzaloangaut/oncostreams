"""Final grouped elongation regressions and independent kernel checks."""
import pickle

import numpy as np
import pytest

from tests._helpers import (
    make_elongation_culture as build_culture,
    assert_cultures_bitwise_equal as compare_bitwise,
    run_steps as run,
)


@pytest.mark.parametrize("rho,dimension,attempts", [
    (0.22, 2, 1), (0.84, 3, 10), (0.90, 2, 5),
])
def test_grouped_elongation_determinism(rho, dimension, attempts):
    a = build_culture("grouped_early_numba", rho=rho,
                      dimension=dimension, attempts=attempts)
    b = build_culture("grouped_early_numba", rho=rho,
                      dimension=dimension, attempts=attempts)
    for tic in range(1, 21):
        run(a, tic - 1, tic)
        run(b, tic - 1, tic)
        compare_bitwise(a, b)


def test_real_elongation_successes_are_recorded():
    culture = build_culture("grouped_early_numba", rho=0.22, n=64, seed=17)
    run(culture, 0, 10)
    assert culture.deformation_event_counts["round_elongation_successes"] > 0


def test_grouped_checkpoint_resume(tmp_path):
    continuous = build_culture("grouped_early_numba", seed=42, n=64, rho=0.5)
    paused = build_culture("grouped_early_numba", seed=42, n=64, rho=0.5)
    run(continuous, 0, 45)
    run(paused, 0, 17)
    checkpoint = tmp_path / "grouped_elongation.pkl"
    paused.save_checkpoint(str(checkpoint), tic=17)
    with checkpoint.open("rb") as stream:
        restored, start, saved_rng_state = pickle.load(stream)
    assert start == 17
    assert restored.elongation_backend == "grouped_early_numba"
    assert getattr(restored.grid, "_batch_neighbor_cache", None) is None
    restored.rng.bit_generator.state = saved_rng_state
    run(restored, start, 45)
    compare_bitwise(continuous, restored)


def test_zero_attempts_matches_standard_path():
    """With no proposals, either elongation backend must do the same thing."""
    reference = build_culture("standard", attempts=0)
    optimized = build_culture("grouped_early_numba", attempts=0)
    run(reference, 0, 3)
    run(optimized, 0, 3)
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
