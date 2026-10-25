"""Production batch interaction regression, noise/RNG, checkpoint and IO."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tests._helpers import (
    assert_cultures_bitwise_equal as assert_same,
    make_batch_culture as make_culture,
    run_steps as run,
)
from tumorsphere.core.interaction_noise_numba import _numba_noise_all


BACKEND = "numba"


def test_synthetic_noise_algebra_and_rng_order():
    """Cover all combinations of enabled translational and rotational noise."""
    for k, (trans, rot) in enumerate(
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ]
    ):
        gen = np.random.default_rng(42 + k)
        n = 137
        phi = gen.uniform(-8 * np.pi, 8 * np.pi, n)
        mobilities = gen.uniform(0.1, 1.8, (n, 3))
        a = np.random.default_rng(123 + k)
        b = np.random.default_rng(123 + k)
        length = n * (2 * int(trans) + int(rot))
        scalar_draws = np.asarray([a.normal(0, 1) for _ in range(length)])
        bulk_draws = b.normal(0, 1, size=length)
        assert scalar_draws.tobytes() == bulk_draws.tobytes()
        assert repr(a.bit_generator.state) == repr(b.bit_generator.state)

        tr_ref = np.zeros((n, 3), dtype=np.float64)
        rot_ref = np.zeros(n, dtype=np.float64)
        cursor = 0
        for i in range(n):
            mP, mS, mR = mobilities[i]
            if trans:
                vec = np.array([np.cos(phi[i]), np.sin(phi[i]), 0])
                nP = (0.033 * np.sqrt(mP * 0.005)) * scalar_draws[cursor]
                cursor += 1
                perp = np.array(
                    [np.cos(phi[i] + np.pi / 2), np.sin(phi[i] + np.pi / 2), 0]
                )
                nS = (0.033 * np.sqrt(mS * 0.005)) * scalar_draws[cursor]
                cursor += 1
                tr_ref[i] = nP * vec + nS * perp
            if rot:
                rot_ref[i] = (
                    np.sqrt(2 * 0.12 * mR * 0.005) * scalar_draws[cursor]
                )
                cursor += 1
        tr, rr = _numba_noise_all(
            phi,
            np.ascontiguousarray(mobilities[:, 0]),
            np.ascontiguousarray(mobilities[:, 1]),
            np.ascontiguousarray(mobilities[:, 2]),
            0.033,
            0.12,
            0.005,
            bulk_draws,
            trans,
            rot,
        )
        assert tr_ref.tobytes() == tr.tobytes()
        assert rot_ref.tobytes() == rr.tobytes()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize(
    "eta,d_phi",
    [
        (0.033, None),
        (None, 0.13),
        (0.033, 0.13),
        (None, None),
        (0.0, 0.0),
    ],
)
def test_production_determinism_each_step(dimension, eta, d_phi):
    """Same seed and optimized backend reproduce the complete dynamic state."""
    a = make_culture(BACKEND, dimension=dimension, eta=eta, d_phi=d_phi)
    b = make_culture(BACKEND, dimension=dimension, eta=eta, d_phi=d_phi)
    for tic in range(1, 17):
        run(a, tic - 1, tic)
        run(b, tic - 1, tic)
        assert_same(a, b)


@pytest.mark.parametrize(
    "eta,d_phi",
    [
        (0.033, None),
        (None, 0.13),
        (0.033, 0.13),
        (None, None),
    ],
)
def test_production_checkpoint_resume(eta, d_phi, tmp_path):
    """Uninterrupted and resumed optimized runs must agree bit-for-bit."""
    continuous = make_culture(BACKEND, seed=42, eta=eta, d_phi=d_phi)
    paused = make_culture(BACKEND, seed=42, eta=eta, d_phi=d_phi)
    run(continuous, 0, 24)
    run(paused, 0, 11)
    assert paused.grid._batch_neighbor_cache is not None
    checkpoint = tmp_path / "culture.pkl"
    paused.save_checkpoint(str(checkpoint), tic=11)
    with checkpoint.open("rb") as handle:
        restored, tic, state = pickle.load(handle)
    assert tic == 11
    assert restored.grid._batch_neighbor_cache is None
    restored.rng.bit_generator.state = state
    run(restored, tic, 24)
    assert_same(continuous, restored)


def test_production_outputs_are_deterministic(tmp_path):
    """Production output bytes are reproducible for fixed seed and config."""
    dir_a = tmp_path / "first"
    dir_b = tmp_path / "second"
    a = make_culture(BACKEND, seed=31, out=dir_a)
    b = make_culture(BACKEND, seed=31, out=dir_b)
    run(a, 0, 20)
    run(b, 0, 20)
    assert_same(a, b)
    files_a = {
        p.relative_to(dir_a): p.read_bytes() for p in dir_a.rglob("*.dat")
    }
    files_b = {
        p.relative_to(dir_b): p.read_bytes() for p in dir_b.rglob("*.dat")
    }
    assert files_a, "The test must write data files"
    assert files_a == files_b
