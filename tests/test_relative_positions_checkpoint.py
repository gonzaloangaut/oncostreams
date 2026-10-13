"""End-to-end tests for backend selection, checkpoint restoration and RNG.

Exercises the real Simulation.simulate_parallel worker with small cultures.
"""

import pickle

import numpy as np
import pytest

from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.simulation import Simulation


def make_simulation(backend="numpy", steps=8, *, noise=False):
    return Simulation(
        forces=[Anisotropic_Grosmann(
            kRep=2.5, bExp=3.0, lambda_core=1.0,
            shrinking=True, noise_eta=0.05 if noise else None,
            d_phi=0.01 if noise else None,
        )],
        rng_seed=13579,
        num_of_realizations=1,
        num_of_steps_per_realization=steps,
        initial_number_of_cells=[48],
        initial_fraction_elongated=[0.5],
        initial_density=[0.84],
        cell_max_def_attempts=1,
        reproduction=False, movement=True, deformation=True,
        stabilization_time=0, overlap_threshold_ratio=np.exp(-1),
        delta_t=0.005, deformation_attempt_period=0.05,
        initial_aspect_ratio=1.0, aspect_ratio_max=2.7,
        cell_speed_max=1.0, trabajo_final=True,
        spatial_dimension=3,
        overlap_backend="numba", force_backend="numba",
        relative_positions_backend=backend,
    )


def run(backend, steps, out, *, noise=False):
    sim = make_simulation(backend, steps, noise=noise)
    sim.simulate_parallel(sql=False, number_of_processes=1, output_dir=str(out))
    paths = list((out / "checkpoints").glob("*.pkl"))
    assert len(paths) == 1
    with paths[0].open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    assert tic == steps
    assert culture.relative_positions_backend == backend
    assert culture._checkpoint_model_config["relative_positions_backend"] == backend
    return culture, paths[0]


def assert_exact_state(a, b):
    np.testing.assert_array_equal(a.cell_positions, b.cell_positions)
    np.testing.assert_array_equal(a.cell_phies, b.cell_phies)
    np.testing.assert_array_equal(
        np.array([c.aspect_ratio for c in a.cells]),
        np.array([c.aspect_ratio for c in b.cells]),
    )
    np.testing.assert_array_equal(
        np.array([c.is_round for c in a.cells]),
        np.array([c.is_round for c in b.cells]),
    )
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


@pytest.mark.parametrize("invalid", ["CUDA", "fast", "NUMBA", None])
def test_invalid_backend_rejected(invalid):
    with pytest.raises(ValueError, match="relative_positions_backend"):
        make_simulation(invalid)


def test_numpy_defaults():
    assert Simulation().relative_positions_backend == "numpy"


@pytest.mark.parametrize("noise", [False, True])
def test_numpy_numba_exact_simulation(noise, tmp_path):
    pytest.importorskip("numba")
    numpy_state, _ = run("numpy", 14, tmp_path / "numpy", noise=noise)
    numba_state, _ = run("numba", 14, tmp_path / "numba", noise=noise)
    assert_exact_state(numpy_state, numba_state)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_resume_matches_uninterrupted(backend, tmp_path):
    if backend == "numba":
        pytest.importorskip("numba")
    short_path = tmp_path / "partial"
    run(backend, 4, short_path)
    resumed, _ = run(backend, 8, short_path)
    full, _ = run(backend, 8, tmp_path / "uninterrupted")
    assert_exact_state(resumed, full)


@pytest.mark.parametrize("old,new", [("numpy", "numba"), ("numba", "numpy")])
def test_backend_switch_on_resume_rejected(old, new, tmp_path):
    pytest.importorskip("numba")
    run(old, 4, tmp_path)
    with pytest.raises(ValueError, match="relative_positions_backend"):
        run(new, 8, tmp_path)


def test_legacy_checkpoint_without_backend_defaults_to_numpy(tmp_path):
    _, checkpoint = run("numpy", 4, tmp_path)
    with checkpoint.open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    culture._checkpoint_model_config.pop("relative_positions_backend", None)
    del culture.relative_positions_backend
    with checkpoint.open("wb") as stream:
        pickle.dump((culture, tic, state), stream, pickle.HIGHEST_PROTOCOL)
    resumed, _ = run("numpy", 8, tmp_path)
    assert resumed.relative_positions_backend == "numpy"
    assert resumed._checkpoint_model_config["relative_positions_backend"] == "numpy"


def test_inconsistent_checkpoint_metadata_rejected(tmp_path):
    _, checkpoint = run("numpy", 4, tmp_path)
    with checkpoint.open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    culture._checkpoint_model_config["relative_positions_backend"] = "numba"
    with checkpoint.open("wb") as stream:
        pickle.dump((culture, tic, state), stream, pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="relative_positions_backend.*disagrees"):
        run("numpy", 8, tmp_path)
