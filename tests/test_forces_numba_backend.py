"""Numerical regression tests for the optional Numba Grosmann backend."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")

from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.forces_numba import (
    XI_POWER_MAX,
    grosmann_force_torque_kernel,
    prepare_force_arrays,
)


class DummyCell:
    def __init__(self, culture, index, aspect):
        self.culture = culture
        self._index = index
        self.aspect_ratio = float(aspect)
        self.is_round = bool(np.isclose(aspect, 1.0))
        self.squared_diagonal = float(culture.area / np.pi * (aspect + 1/aspect))
        self.anisotropy = float((aspect * aspect - 1) / (aspect * aspect + 1))
        self.neighbors_relative_pos = {}
        self.neighbors_normalized_overlap = {}
        self.shrink = False

    def velocity(self):
        phi = self.culture.phies[self._index]
        speed = (self.aspect_ratio - 1) / (2.7 - 1)
        return speed * np.array([np.cos(phi), np.sin(phi), 0.0])


def make_case(n, seed, nonplanar, zero_distance=False, xi_override=None):
    rng = np.random.default_rng(seed)
    area = np.pi
    aspects = rng.choice([1.0, 1.2, 1.8, 2.7], size=n+1)
    phies = rng.uniform(-np.pi, np.pi, n+1)
    q = np.zeros((n+1, 3, 3), dtype=float)
    q[:, 0, 0] = np.cos(2*phies)
    q[:, 0, 1] = q[:, 1, 0] = np.sin(2*phies)
    q[:, 1, 1] = -q[:, 0, 0]
    culture = SimpleNamespace(area=area, phies=phies,
                              rng=np.random.default_rng(seed+10000))
    cells = [DummyCell(culture, i, aspects[i]) for i in range(n+1)]
    r = rng.normal(0.0, 1.5, (n, 3))
    if not nonplanar:
        r[:, 2] = 0.0
    if n and zero_distance:
        r[0] = 0.0
    for k in range(n):
        neighbor = cells[k+1]
        center = cells[0]
        ds = center.squared_diagonal + neighbor.squared_diagonal
        beta = (ds**2 - (center.squared_diagonal*center.anisotropy -
                         neighbor.squared_diagonal*neighbor.anisotropy)**2
                - 4*center.squared_diagonal*center.anisotropy *
                neighbor.squared_diagonal*neighbor.anisotropy *
                np.cos(phies[0]-phies[k+1])**2)
        M = ((center.squared_diagonal*center.anisotropy*q[0]
              +neighbor.squared_diagonal*neighbor.anisotropy*q[k+1])/ds)
        xi = np.exp(-(ds/beta)*np.dot(r[k],(np.identity(3)-M)@r[k]))
        if xi_override is not None and k == 0:
            xi = xi_override
        center.neighbors_relative_pos[k+1] = r[k]
        center.neighbors_normalized_overlap[k+1] = xi
    return cells, phies, q, area, list(range(1, n+1))


def run_pair(n, seed, lam, nonplanar, noise, shrinking=False,
             zero_distance=False, xi_override=None):
    kw = dict(kRep=2.5, bExp=3.0, lambda_core=lam,
              shrinking=shrinking,
              noise_eta=0.033 if noise in ('both', 'trans') else None,
              d_phi=0.11 if noise in ('both', 'rot') else None)
    c0, phies0, q0, area, indices = make_case(
        n, seed, nonplanar, zero_distance, xi_override)
    c1, phies1, q1, _, indices1 = make_case(
        n, seed, nonplanar, zero_distance, xi_override)
    ref = Anisotropic_Grosmann(**kw)
    candidate = Anisotropic_Grosmann(**kw)
    a = ref.calculate_interaction(c0, phies0, 0, 0.005, area, indices, q0)
    candidate.force_backend = "numba"
    b = candidate.calculate_interaction(c1, phies1, 0, 0.005,
                                        area, indices1, q1)
    for x,y in zip(a,b):
        np.testing.assert_allclose(x, y, atol=1e-12, rtol=2e-12)
    assert c0[0].shrink == c1[0].shrink
    # Exactly the same noise draws must be consumed, in the same order.
    assert c0[0].culture.rng.bit_generator.state == c1[0].culture.rng.bit_generator.state
    return a,b


@pytest.mark.parametrize('n',[0,1,2,5,10,20])
@pytest.mark.parametrize('lam',[0.0, 0.35, 1.0])
@pytest.mark.parametrize('nonplanar',[False,True])
@pytest.mark.parametrize('seed',[23,97])
def test_forces_and_torque_match_model(n, lam, nonplanar, seed):
    run_pair(n,seed,lam,nonplanar,noise='none')


@pytest.mark.parametrize('noise',['none','trans','rot','both'])
@pytest.mark.parametrize('lam',[0.0,1.0])
@pytest.mark.parametrize('n',[0,1,5])
def test_noise_shrinking_and_rng_are_unchanged(noise,lam,n):
    run_pair(n,717,lam,False,noise,shrinking=True)


@pytest.mark.parametrize('xi',[1.0, np.nextafter(1.0,0.0), 0.0, 0.999999, 0.4])
def test_core_amplification_extremes(xi):
    # Zero separation creates xi=1. Explicit overrides exercise numerical edge cases.
    run_pair(1,5,1.0,False,'none',zero_distance=True,xi_override=xi)


def test_kernel_signature_and_empty_neighbors():
    cells, phies, q, _, indices = make_case(0,18,False)
    data = prepare_force_arrays(cells, phies, 0, indices, q)
    total, torque = grosmann_force_torque_kernel(
        *data, 2.5, 3.0, 1.0, XI_POWER_MAX)
    np.testing.assert_array_equal(total,np.zeros(3))
    assert torque == 0
    assert grosmann_force_torque_kernel.nopython_signatures
