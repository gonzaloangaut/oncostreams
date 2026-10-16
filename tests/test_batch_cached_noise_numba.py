"""Bitwise regression tests for optional compiled batch noise.

Run: python -m pytest -q tests/test_batch_cached_noise_numba.py
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from tests._helpers import (
    assert_cultures_bitwise_equal,
    make_batch_culture,
    run_steps,
)
from tumorsphere.core.interaction_noise_numba import _numba_noise_all


BASELINE = 'batch_cached_numba'
CANDIDATE = 'batch_cached_noise_numba'


# Keep names used by existing tests; construction and assertions are shared.
make_culture = make_batch_culture
run = run_steps
assert_same = assert_cultures_bitwise_equal


def test_synthetic_noise_algebra_and_rng_order():
    """Cover all combinations of enabled translational and rotational noise."""
    for k, (trans, rot) in enumerate([
        (True, False), (False, True), (True, True), (False, False),
    ]):
        gen = np.random.default_rng(42 + k)
        n = 137
        phi = gen.uniform(-8 * np.pi, 8 * np.pi, n)
        mobilities = gen.uniform(.1, 1.8, (n, 3))
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
                perp = np.array([np.cos(phi[i] + np.pi / 2),
                                 np.sin(phi[i] + np.pi / 2), 0])
                nS = (0.033 * np.sqrt(mS * 0.005)) * scalar_draws[cursor]
                cursor += 1
                tr_ref[i] = nP * vec + nS * perp
            if rot:
                rot_ref[i] = np.sqrt(2 * 0.12 * mR * 0.005) * scalar_draws[cursor]
                cursor += 1
        tr, rr = _numba_noise_all(
            phi, np.ascontiguousarray(mobilities[:, 0]),
            np.ascontiguousarray(mobilities[:, 1]),
            np.ascontiguousarray(mobilities[:, 2]),
            0.033, 0.12, 0.005, bulk_draws, trans, rot,
        )
        assert tr_ref.tobytes() == tr.tobytes()
        assert rot_ref.tobytes() == rr.tobytes()


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('eta,d_phi', [
    (0.033, None),
    (None, 0.13),
    (0.033, 0.13),
    (None, None),
    (0.0, None),
    (None, 0.0),
    (0.0, 0.0),
])
def test_bitwise_trajectory_each_step(dimension, eta, d_phi):
    base = make_culture(BASELINE, dimension=dimension, eta=eta, d_phi=d_phi)
    new = make_culture(CANDIDATE, dimension=dimension, eta=eta, d_phi=d_phi)
    for tic in range(1, 17):
        run(base, tic - 1, tic)
        run(new, tic - 1, tic)
        assert_same(base, new)


@pytest.mark.parametrize('rho', [0.30, 0.84, 0.90])
@pytest.mark.parametrize('eta,d_phi', [
    (0.033, None), (None, 0.13), (0.033, 0.13),
])
def test_vectorized_postprocessing_varied_conditions(rho, eta, d_phi):
    """Extended comparison against the existing per-cell batch postprocessing."""
    baseline = make_culture(BASELINE, seed=29, n=96, rho=rho,
                            eta=eta, d_phi=d_phi)
    candidate = make_culture(CANDIDATE, seed=29, n=96, rho=rho,
                             eta=eta, d_phi=d_phi)
    run(baseline, 0, 75)
    run(candidate, 0, 75)
    assert_same(baseline, candidate)


@pytest.mark.parametrize('eta,d_phi', [
    (0.033, None), (None, 0.13), (0.033, 0.13), (None, None),
])
def test_checkpoint_resume(eta, d_phi, tmp_path):
    base = make_culture(BASELINE, seed=42, eta=eta, d_phi=d_phi)
    new = make_culture(CANDIDATE, seed=42, eta=eta, d_phi=d_phi)
    run(base, 0, 24)
    run(new, 0, 11)
    assert new.grid._batch_neighbor_cache is not None
    path = tmp_path / 'checkpoint.pkl'
    new.save_checkpoint(str(path), tic=11)
    with path.open('rb') as handle:
        restored, tic, state = pickle.load(handle)
    assert tic == 11
    assert restored.grid._batch_neighbor_cache is None
    restored.rng.bit_generator.state = state
    run(restored, tic, 24)
    assert_same(base, restored)


def test_output_bytes(tmp_path):
    baseline_dir = tmp_path / 'reference'
    optimized_dir = tmp_path / 'optimized'
    a = make_culture(BASELINE, seed=31, out=baseline_dir)
    b = make_culture(CANDIDATE, seed=31, out=optimized_dir)
    run(a, 0, 20)
    run(b, 0, 20)
    assert_same(a, b)
    fa = {p.relative_to(baseline_dir): p.read_bytes()
          for p in baseline_dir.rglob('*.dat')}
    fb = {p.relative_to(optimized_dir): p.read_bytes()
          for p in optimized_dir.rglob('*.dat')}
    assert fa, 'The experiment should write at least one .dat file'
    assert fa == fb
