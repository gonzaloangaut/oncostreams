"""Regression tests for the physical-time deformation schedule."""

from __future__ import annotations

import numpy as np
import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import (
    DatOutput_deformation_parameters,
    OutputDemux,
)
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


@pytest.mark.parametrize(
    "delta_t",
    (0.05, 0.025, 0.01, 0.005, 0.001),
)
def test_first_deformation_sweep_uses_state_at_physical_period(
    tmp_path,
    delta_t,
):
    """The first sweep must act on the state at t=T_def for every delta_t."""
    deformation_period = 0.1
    interval_steps = int(round(deformation_period / delta_t))

    threshold = np.exp(-1)
    aspect_ratio_max = 2.7
    side = 20.0
    cube_size = 2 * np.sqrt(
        aspect_ratio_max * (-np.log(threshold))
    )

    output = DatOutput_deformation_parameters(
        "schedule",
        str(tmp_path),
        save_step=1,
    )
    grid = SpatialHashGrid(
        culture=None,
        bounds=side,
        cube_size=cube_size,
        torus=True,
        spatial_dimension=2,
    )
    force = Anisotropic_Grosmann(
        kRep=2.5,
        bExp=3.0,
        shrinking=True,
        lambda_core=1.0,
    )

    culture = Culture(
        output=OutputDemux("schedule", [output]),
        force=force,
        grid=grid,
        rng_seed=13579,
        initial_number_of_cells=1,
        initial_fraction_elongated=0.0,
        reproduction=False,
        movement=True,
        deformation=True,
        stabilization_time=0,
        overlap_threshold_ratio=threshold,
        delta_t=delta_t,
        deformation_attempt_period=deformation_period,
        aspect_ratio_max=aspect_ratio_max,
        cell_speed_max=1.0,
        trabajo_final=True,
        cell_max_def_attempts=1,
        backend="numpy",
    )

    # The sweep at state tic=interval_steps happens before integrating
    # dynamic step interval_steps + 1.
    culture.simulate(
        num_times=interval_steps + 1,
        start_tic=0,
        checkpoint_path=None,
    )

    output_dir = tmp_path / "dat_deformation_parameters"

    before = np.genfromtxt(
        output_dir
        / f"deformation_events_schedule_step={interval_steps:05d}.dat",
        delimiter=",",
        names=True,
    )
    after = np.genfromtxt(
        output_dir
        / (
            "deformation_events_schedule_step="
            f"{interval_steps + 1:05d}.dat"
        ),
        delimiter=",",
        names=True,
    )

    assert int(before["round_elongation_attempts"]) == 0
    assert int(after["round_elongation_attempts"]) == 1

    state_time_of_first_sweep = interval_steps * delta_t
    assert state_time_of_first_sweep == pytest.approx(
        deformation_period,
        rel=0,
        abs=1e-12,
    )
