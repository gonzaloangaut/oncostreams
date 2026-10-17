"""Unified backend selection and real multiprocessing checkpoint regressions.

New checkpoints only: no compatibility with earlier component-selector formats.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from tests._helpers import assert_cultures_bitwise_equal
from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann, Spring_Force
from tumorsphere.core.simulation import Simulation


COMPONENTS = {
    "numpy": ("numpy", "numpy", "numpy", "cached", "standard"),
    "numba": ("numba", "numba", "numba", "batch_cached_noise_numba",
              "grouped_early_numba"),
}


def make_simulation(backend, steps, *, noise=False):
    return Simulation(
        forces=[Anisotropic_Grosmann(
            kRep=2.5, bExp=3.0, lambda_core=1.0, shrinking=True,
            noise_eta=0.033 if noise else None,
            d_phi=0.13 if noise else None,
        )],
        backend=backend,
        rng_seed=13579,
        num_of_realizations=1,
        num_of_steps_per_realization=steps,
        initial_number_of_cells=[48],
        initial_fraction_elongated=[0.5],
        initial_density=[0.84],
        cell_max_def_attempts=1,
        reproduction=False,
        movement=True,
        deformation=True,
        stabilization_time=0,
        overlap_threshold_ratio=np.exp(-1),
        delta_t=0.005,
        deformation_attempt_period=0.005,
        initial_aspect_ratio=1.0,
        aspect_ratio_max=2.7,
        cell_speed_max=1.0,
        trabajo_final=True,
        spatial_dimension=3,
    )


def run(backend, steps, output_dir, *, noise=False):
    if backend == "numba":
        pytest.importorskip("numba")
    sim = make_simulation(backend, steps, noise=noise)
    sim.simulate_parallel(
        sql=False, number_of_processes=1, output_dir=str(output_dir)
    )
    paths = list((output_dir / "checkpoints").glob("*.pkl"))
    assert len(paths) == 1
    with paths[0].open("rb") as handle:
        culture, tic, rng_state = pickle.load(handle)
    assert tic == steps
    assert culture.backend == backend
    config = culture._checkpoint_model_config
    assert config["backend"] == backend
    assert config["checkpoint_config_version"] == 3
    for retired_key in (
        "overlap_backend", "force_backend", "relative_positions_backend",
        "interaction_backend", "elongation_backend",
    ):
        assert retired_key not in config
    return culture, paths[0]


def modify_checkpoint(path, modify):
    with path.open("rb") as handle:
        culture, tic, state = pickle.load(handle)
    modify(culture)
    with path.open("wb") as handle:
        pickle.dump((culture, tic, state), handle, pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_backend_selects_expected_components(backend):
    force = Anisotropic_Grosmann()
    sim = Simulation(forces=[force], backend=backend)
    actual = (
        sim.overlap_backend, sim.force_backend, sim.relative_positions_backend,
        sim.interaction_backend, sim.elongation_backend,
    )
    assert actual == COMPONENTS[backend]
    assert sim.backend == backend
    assert Simulation().backend == "numpy"


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("noise", [False, True])
def test_resume_matches_uninterrupted_bitwise(backend, noise, tmp_path):
    part = tmp_path / "partial"
    run(backend, 7, part, noise=noise)
    resumed, _ = run(backend, 13, part, noise=noise)
    uninterrupted, _ = run(
        backend, 13, tmp_path / "uninterrupted", noise=noise
    )
    assert_cultures_bitwise_equal(resumed, uninterrupted)


@pytest.mark.parametrize("old,new", [("numpy", "numba"), ("numba", "numpy")])
def test_backend_change_on_resume_is_rejected(old, new, tmp_path):
    run(old, 4, tmp_path)
    with pytest.raises(ValueError, match="backend"):
        run(new, 8, tmp_path)


@pytest.mark.parametrize("tamper", ["metadata", "serialized", "components"])
def test_checkpoint_backend_tampering_is_rejected(tamper, tmp_path):
    _, path = run("numpy", 4, tmp_path)

    def corrupt(culture):
        if tamper == "metadata":
            culture._checkpoint_model_config["backend"] = "numba"
        elif tamper == "serialized":
            culture.backend = "numba"
        else:
            culture.relative_positions_backend = "numba"

    modify_checkpoint(path, corrupt)
    with pytest.raises(ValueError, match="backend"):
        run("numpy", 8, tmp_path)


def test_old_checkpoint_format_is_not_accepted(tmp_path):
    _, path = run("numpy", 4, tmp_path)
    modify_checkpoint(
        path,
        lambda culture: culture._checkpoint_model_config.update(
            checkpoint_config_version=2
        ),
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        run("numpy", 8, tmp_path)


@pytest.mark.parametrize("backend", ["CUDA", "fast", "NUMBA", "cached", None])
def test_invalid_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="backend must"):
        Simulation(backend=backend)


@pytest.mark.parametrize("retired", [
    "overlap_backend", "force_backend", "relative_positions_backend",
    "interaction_backend", "elongation_backend",
])
def test_individual_selector_arguments_are_removed(retired):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Simulation(**{retired: "numpy"})


def test_numba_requires_supported_model():
    with pytest.raises(ValueError, match="Anisotropic_Grosmann"):
        Simulation(backend="numba", forces=[Spring_Force()])
    with pytest.raises(ValueError, match="Anisotropic_Grosmann"):
        Simulation(backend="numba")
    with pytest.raises(ValueError, match="fixed-N movement"):
        Simulation(backend="numba", forces=[Anisotropic_Grosmann()], reproduction=True)
    with pytest.raises(ValueError, match="fixed-N movement"):
        Simulation(backend="numba", forces=[Anisotropic_Grosmann()], movement=False)
    assert Simulation(backend="numpy", forces=[Spring_Force()])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_new_culture_uses_complete_backend_configuration(backend, tmp_path):
    culture, _ = run(backend, 2, tmp_path)
    components = (
        culture.overlap_backend, culture.force.force_backend,
        culture.relative_positions_backend, culture.interaction_backend,
        culture.elongation_backend,
    )
    assert components == COMPONENTS[backend]


@pytest.mark.parametrize("backend", [
    "direct_numba", "batch_numba", "batch_cached_numba",
    "fast_numba", None,
])
def test_direct_culture_rejects_removed_backends(backend):
    with pytest.raises(ValueError, match="backend must"):
        Culture(output=None, force=None, grid=None, backend=backend)
