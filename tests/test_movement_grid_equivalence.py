"""Regression tests comparing planar dynamics with 2D and 3D grids."""

import numpy as np
import pytest

from tumorsphere.core.culture import Culture
from tumorsphere.core.forces import Anisotropic_Grosmann
from tumorsphere.core.output import create_output_demux
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


@pytest.mark.parametrize("seed", [17, 42])
@pytest.mark.parametrize("initial_elongated", [0.0, 0.5])
def test_2d_and_3d_grids_produce_identical_dynamics(
    tmp_path, seed, initial_elongated
):
    """Changing only the grid dimension must preserve the trajectory."""

    n = 48
    rho = 0.84
    dt = 0.005
    aspect_max = 2.7
    overlap_threshold = np.exp(-1)

    side = np.sqrt(n * np.pi / rho)

    cube_size = 2.0 * np.sqrt(
        aspect_max * (-np.log(overlap_threshold))
    )

    cultures = []

    for dimension in (2, 3):
        output = create_output_demux(
            culture_name=f"test_{dimension}d",
            requested_outputs=[],
            output_dir=str(tmp_path),
        )

        force = Anisotropic_Grosmann(
            kRep=2.5,
            bExp=3.0,
            lambda_core=1.0,
            shrinking=True,
            noise_eta=None,
            d_phi=None,
        )

        grid = SpatialHashGrid(
            bounds=side,
            cube_size=cube_size,
            torus=True,
            spatial_dimension=dimension,
        )

        culture = Culture(
            output=output,
            force=force,
            grid=grid,
            initial_number_of_cells=n,
            initial_fraction_elongated=initial_elongated,
            rng_seed=seed,
            reproduction=False,
            movement=True,
            deformation=True,
            stabilization_time=0,
            overlap_threshold_ratio=overlap_threshold,
            delta_t=dt,
            deformation_attempt_period=0.05,
            initial_aspect_ratio=1.0,
            aspect_ratio_max=aspect_max,
            cell_speed_max=1.0,
            cell_max_def_attempts=1,
            trabajo_final=True,
            initialization_mode="random",
        )

        culture.simulate(
            num_times=100,
            start_tic=0,
            checkpoint_path=None,
        )

        cultures.append(culture)

    culture_2d, culture_3d = cultures

    # Exact equality is expected: only the neighbor search changed.
    np.testing.assert_array_equal(
        culture_2d.cell_positions,
        culture_3d.cell_positions,
    )

    np.testing.assert_array_equal(
        culture_2d.cell_phies,
        culture_3d.cell_phies,
    )

    np.testing.assert_array_equal(
        culture_2d.nematic_tensors,
        culture_3d.nematic_tensors,
    )

    np.testing.assert_array_equal(
        [cell.aspect_ratio for cell in culture_2d.cells],
        [cell.aspect_ratio for cell in culture_3d.cells],
    )

    assert (
        [cell.is_round for cell in culture_2d.cells]
        ==
        [cell.is_round for cell in culture_3d.cells]
    )