"""Integration tests for selecting overlap backends and restoring checkpoints.

Place at tests/test_overlap_backend_checkpoint.py after applying
apply_overlap_backend_config.py. These run a very small real simulation,
including the multiprocessing worker used by Simulation.simulate_parallel.
"""

import pickle

import numpy as np
import pytest

from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.simulation import Simulation


def make_simulation(backend, steps):
    return Simulation(
        forces=[Anisotropic_Grosmann(
            kRep=2.5,
            bExp=3.0,
            lambda_core=1.0,
            shrinking=True,
            noise_eta=None,
            d_phi=None,
        )],
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
        deformation_attempt_period=0.05,
        initial_aspect_ratio=1.0,
        aspect_ratio_max=2.7,
        cell_speed_max=1.0,
        trabajo_final=True,
        spatial_dimension=3,
        overlap_backend=backend,
    )


def run_simulation(backend, steps, output_dir):
    sim = make_simulation(backend, steps)
    sim.simulate_parallel(
        sql=False,
        number_of_processes=1,
        output_dir=str(output_dir),
    )
    checkpoints = list((output_dir / "checkpoints").glob("*.pkl"))
    assert len(checkpoints) == 1
    with checkpoints[0].open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    assert tic == steps
    assert culture.overlap_backend == backend
    assert culture._checkpoint_model_config["overlap_backend"] == backend
    return culture, checkpoints[0]


@pytest.mark.parametrize("bad_backend", ["invalid", "NUMBA", "cuda", None])
def test_reject_invalid_overlap_backend(bad_backend):
    with pytest.raises(ValueError, match="overlap_backend"):
        make_simulation(bad_backend, 1)


def test_numpy_is_the_default():
    assert Simulation().overlap_backend == "numpy"


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_backend_survives_checkpoint_resume(backend, tmp_path):
    if backend == "numba":
        pytest.importorskip("numba")

    # A real 4-step checkpoint, then resume to step 8.
    partial = tmp_path / "partial"
    _, checkpoint = run_simulation(backend, 4, partial)
    resumed, _ = run_simulation(backend, 8, partial)

    # Compare to 8 uninterrupted steps with the same seed.
    full, _ = run_simulation(backend, 8, tmp_path / "full")
    np.testing.assert_allclose(
        resumed.cell_positions, full.cell_positions, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        resumed.cell_phies, full.cell_phies, rtol=0, atol=2e-12
    )
    np.testing.assert_array_equal(
        [cell.aspect_ratio for cell in resumed.cells],
        [cell.aspect_ratio for cell in full.cells],
    )
    assert checkpoint.exists()


@pytest.mark.parametrize("old_backend,new_backend", [
    ("numpy", "numba"), ("numba", "numpy"),
])
def test_cannot_silently_switch_backend_on_resume(
    old_backend, new_backend, tmp_path
):
    if "numba" in (old_backend, new_backend):
        pytest.importorskip("numba")
    run_simulation(old_backend, 4, tmp_path)
    with pytest.raises(ValueError, match="overlap_backend"):
        run_simulation(new_backend, 8, tmp_path)


def test_legacy_checkpoint_without_backend_defaults_to_numpy(tmp_path):
    _, checkpoint_path = run_simulation("numpy", 4, tmp_path)

    # Emulate an older checkpoint that predates the new backend metadata.
    with checkpoint_path.open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    culture._checkpoint_model_config.pop("overlap_backend", None)
    del culture.overlap_backend
    with checkpoint_path.open("wb") as stream:
        pickle.dump((culture, tic, state), stream, pickle.HIGHEST_PROTOCOL)

    resumed, _ = run_simulation("numpy", 8, tmp_path)
    assert resumed.overlap_backend == "numpy"
    assert resumed._checkpoint_model_config["overlap_backend"] == "numpy"
