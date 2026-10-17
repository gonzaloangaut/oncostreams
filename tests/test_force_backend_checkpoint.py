"""Checkpoint and configuration regressions for reference and batch forces.

Each backend runs a real simulation through the multiprocessing worker.
"""

import pickle

import numpy as np
import pytest

from tests._helpers import assert_cultures_bitwise_equal
from tumorsphere.core.forces import Anisotropic_Grosmann, Spring_Force
from tumorsphere.core.simulation import Simulation


def make_simulation(force_backend, steps, *, use_selector=False):
    optimized = force_backend == "numba"
    backend_options = (
        {"backend": force_backend}
        if use_selector
        else {
            "overlap_backend": "numba" if optimized else "numpy",
            "force_backend": force_backend,
            "relative_positions_backend": "numba" if optimized else "numpy",
            "interaction_backend": (
                "batch_cached_noise_numba" if optimized else "cached"
            ),
            "elongation_backend": (
                "grouped_early_numba" if optimized else "standard"
            ),
        }
    )
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
        initial_number_of_cells=[48],  # At least 3 periodic grid buckets.
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
        **backend_options,
    )


def run_simulation(force_backend, steps, output_dir, *, use_selector=False):
    simulation = make_simulation(force_backend, steps, use_selector=use_selector)
    simulation.simulate_parallel(
        sql=False, number_of_processes=1, output_dir=str(output_dir)
    )
    paths = list((output_dir / "checkpoints").glob("*.pkl"))
    assert len(paths) == 1
    with paths[0].open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    assert tic == steps
    assert getattr(culture.force, "force_backend", "numpy") == force_backend
    assert culture._checkpoint_model_config["force_backend"] == force_backend
    return culture, paths[0]


@pytest.mark.parametrize("invalid", ["bad", "NUMBA", "cuda", None])
def test_force_backend_validation(invalid):
    with pytest.raises(ValueError, match="force_backend"):
        make_simulation(invalid, steps=2)


def test_force_backend_defaults_to_numpy():
    assert Simulation().force_backend == "numpy"
    assert Anisotropic_Grosmann().force_backend == "numpy"


def test_individual_numba_force_backend_is_retired():
    with pytest.raises(ValueError, match="interaction_backend"):
        Simulation(
            forces=[Anisotropic_Grosmann()],
            force_backend="numba",
            interaction_backend="cached",
        )


def test_numba_force_backend_rejects_other_force_classes():
    with pytest.raises(ValueError, match="Anisotropic_Grosmann"):
        Simulation(forces=[Spring_Force()], force_backend="numba")
    assert Simulation(forces=[Spring_Force()], force_backend="numpy")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_force_backend_checkpoint_resume(backend, tmp_path):
    if backend == "numba":
        pytest.importorskip("numba")
    short_path = tmp_path / "short"
    run_simulation(backend, 4, short_path)
    resumed, _ = run_simulation(backend, 8, short_path)
    uninterrupted, _ = run_simulation(backend, 8, tmp_path / "full")
    np.testing.assert_allclose(
        resumed.cell_positions, uninterrupted.cell_positions,
        rtol=0, atol=2e-12,
    )
    np.testing.assert_allclose(
        resumed.cell_phies, uninterrupted.cell_phies,
        rtol=0, atol=2e-12,
    )
    np.testing.assert_array_equal(
        [c.aspect_ratio for c in resumed.cells],
        [c.aspect_ratio for c in uninterrupted.cells],
    )
    assert resumed.rng.bit_generator.state == uninterrupted.rng.bit_generator.state


@pytest.mark.parametrize("old,new", [("numpy", "numba"), ("numba", "numpy")])
def test_changing_force_backend_on_resume_is_rejected(old, new, tmp_path):
    pytest.importorskip("numba")
    run_simulation(old, 4, tmp_path)
    with pytest.raises(ValueError, match="force_backend"):
        run_simulation(new, 8, tmp_path)


def test_inconsistent_force_backend_metadata_is_rejected(tmp_path):
    _, path = run_simulation("numpy", 4, tmp_path)
    with path.open("rb") as stream:
        culture, tic, state = pickle.load(stream)
    culture._checkpoint_model_config["force_backend"] = "numba"
    with path.open("wb") as stream:
        pickle.dump((culture, tic, state), stream, pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="force_backend.*disagrees"):
        run_simulation("numpy", 8, tmp_path)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_unified_selector_preserves_exact_dynamics(backend, tmp_path):
    if backend == "numba":
        pytest.importorskip("numba")
    old, _ = run_simulation(backend, 8, tmp_path / "explicit")
    selected, _ = run_simulation(
        backend, 8, tmp_path / "selector", use_selector=True,
    )
    assert_cultures_bitwise_equal(old, selected)
    assert make_simulation(backend, 8, use_selector=True).backend == backend


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_unified_selector_checkpoint_resume(backend, tmp_path):
    if backend == "numba":
        pytest.importorskip("numba")
    partial = tmp_path / "partial"
    run_simulation(backend, 4, partial, use_selector=True)
    resumed, _ = run_simulation(backend, 8, partial, use_selector=True)
    full, _ = run_simulation(
        backend, 8, tmp_path / "full", use_selector=True,
    )
    assert_cultures_bitwise_equal(resumed, full)


@pytest.mark.parametrize("invalid", ["CUDA", "fast", "NUMBA", "cached"])
def test_unified_selector_rejects_invalid_backend(invalid):
    with pytest.raises(ValueError, match="backend must"):
        Simulation(backend=invalid)


def test_unified_selector_rejects_component_override():
    with pytest.raises(ValueError, match="cannot be combined"):
        Simulation(
            backend="numba",
            overlap_backend="numba",
            forces=[Anisotropic_Grosmann()],
        )
