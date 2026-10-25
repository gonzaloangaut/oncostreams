"""Regressions for per-step motion statistics and unwrapped MSD.

These tests use analytic trajectories to validate observable definitions, and
one small end-to-end checkpoint run to verify the MSD time origin on resume.
They do not test convergence of the physical integrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import DatOutput_motion_parameters
from tumorsphere.core.simulation import Simulation


def _position(x: float, y: float = 0.0) -> np.ndarray:
    """Return a single planar cell position with the production (N, 3) shape."""
    return np.array([[x, y, 0.0]], dtype=float)


def _saved_motion(path, step: int) -> np.ndarray:
    """Read the five motion observables from the file for a given step."""
    matches = list(path.glob(f"motion_*_step={step:05d}.dat"))
    assert len(matches) == 1, (step, matches)
    data = np.loadtxt(matches[0], delimiter=",", skiprows=1, ndmin=1)
    assert data.shape == (5,)
    return data


def test_constant_velocity_gives_ballistic_msd():
    """The per-step displacement stays fixed while MSD grows as time squared."""
    output = DatOutput_motion_parameters("ballistic")
    dt, speed, side = 0.25, 1.2, 20.0
    step_length = speed * dt

    assert output.calculate_motion_parameters(_position(1.0), side) == (
        0.0, 0.0, 0.0, 0.0, 0.0,
    )

    for step in range(1, 6):
        mean_dr, mean_dr2, p95_dr, max_dr, msd = (
            output.calculate_motion_parameters(
                _position(1.0 + step * step_length), side,
            )
        )
        np.testing.assert_allclose(
            (mean_dr, mean_dr2, p95_dr, max_dr, msd),
            (
                step_length,
                step_length**2,
                step_length,
                step_length,
                (speed * step * dt) ** 2,
            ),
            rtol=0,
            atol=1e-12,
        )


def test_periodic_crossing_preserves_unwrapped_displacement():
    """Crossing x=L does not generate an artificial jump in displacement or MSD."""
    output = DatOutput_motion_parameters("periodic")
    side = 10.0

    output.calculate_motion_parameters(_position(9.8), side)
    for step, wrapped_x in enumerate((0.1, 0.4), start=1):
        mean_dr, mean_dr2, p95_dr, max_dr, msd = (
            output.calculate_motion_parameters(_position(wrapped_x), side)
        )
        np.testing.assert_allclose(
            (mean_dr, mean_dr2, p95_dr, max_dr, msd),
            (0.3, 0.09, 0.3, 0.3, (0.3 * step) ** 2),
            rtol=0,
            atol=1e-12,
        )


def _record_synthetic_motion(output, *, step: int, wrapped_x: float, side: float):
    """Feed the motion output using the same recording entry point as Culture."""
    output.record_culture_state(
        tic=step,
        cells=[],  # This output only consumes cell_positions and side.
        cell_positions=_position(wrapped_x),
        cell_phies=np.zeros(1),
        active_cell_indexes=[0],
        side=side,
        cell_area=np.pi,
    )


def test_save_interval_does_not_change_msd_or_step_definition(tmp_path):
    """Skipping files must not skip unwrapping intermediate timesteps."""
    side, x0, step_length = 3.0, 0.25, 0.7
    dense_dir = tmp_path / "dense"
    sparse_dir = tmp_path / "sparse"
    dense = DatOutput_motion_parameters("dense", str(dense_dir), save_step=1)
    sparse = DatOutput_motion_parameters("sparse", str(sparse_dir), save_step=4)

    for step in range(9):
        wrapped_x = (x0 + step * step_length) % side
        for output in (dense, sparse):
            _record_synthetic_motion(
                output, step=step, wrapped_x=wrapped_x, side=side,
            )

    dense_files = dense_dir / "dat_motion_parameters"
    sparse_files = sparse_dir / "dat_motion_parameters"
    assert len(list(dense_files.glob("motion_*.dat"))) == 9
    assert len(list(sparse_files.glob("motion_*.dat"))) == 3

    for step in (0, 4, 8):
        dense_record = _saved_motion(dense_files, step)
        sparse_record = _saved_motion(sparse_files, step)
        np.testing.assert_allclose(sparse_record, dense_record, rtol=0, atol=1e-12)
        if step:
            np.testing.assert_allclose(
                sparse_record,
                (step_length, step_length**2, step_length,
                 step_length, (step * step_length) ** 2),
                rtol=0,
                atol=1e-12,
            )


def _run_checkpoint_case(output_dir, *, steps: int, backend: str) -> None:
    """Exercise the public simulation/checkpoint path with motion output."""
    simulation = Simulation(
        forces=[Anisotropic_Grosmann(
            kRep=2.5,
            bExp=3.0,
            noise_eta=0.033,
            d_phi=0.13,
            shrinking=True,
            lambda_core=1.0,
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
    simulation.simulate_parallel(
        sql=False,
        dat_motion_par=True,
        save_step_dat_motion_par=2,
        number_of_processes=1,
        output_dir=str(output_dir),
    )


@pytest.mark.parametrize("backend", ("numpy", "numba"))
def test_checkpoint_resume_preserves_msd_origin(tmp_path, backend):
    """A resumed physical run must match an uninterrupted t=0 MSD history."""
    if backend == "numba":
        pytest.importorskip("numba")

    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"
    _run_checkpoint_case(resumed, steps=5, backend=backend)
    assert len(list((resumed / "checkpoints").glob("*.pkl"))) == 1
    _run_checkpoint_case(resumed, steps=10, backend=backend)
    _run_checkpoint_case(uninterrupted, steps=10, backend=backend)

    resumed_data = resumed / "dat_motion_parameters"
    uninterrupted_data = uninterrupted / "dat_motion_parameters"
    for step in (6, 8, 10):
        np.testing.assert_allclose(
            _saved_motion(resumed_data, step),
            _saved_motion(uninterrupted_data, step),
            rtol=0,
            atol=1e-12,
        )

    assert _saved_motion(resumed_data, 10)[-1] > 0.0
