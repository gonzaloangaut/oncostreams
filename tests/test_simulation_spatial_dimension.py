"""Tests for spatial dimension configuration in Simulation."""

import pytest

from tumorsphere.core.simulation import Simulation


@pytest.mark.parametrize("dimension", [2, 3])
def test_simulation_accepts_spatial_dimension(dimension):
    """Simulation stores the requested spatial dimension."""

    sim = Simulation(
        spatial_dimension=dimension,
        reproduction=False,
    )

    assert sim.spatial_dimension == dimension


@pytest.mark.parametrize("dimension", [0, 1, 4, True])
def test_simulation_rejects_invalid_dimension(dimension):
    """Only integer dimensions 2 and 3 are supported."""

    with pytest.raises(ValueError, match="spatial_dimension"):
        Simulation(spatial_dimension=dimension)


def test_2d_simulation_rejects_reproduction():
    """Current reproduction dynamics require a 3D grid."""

    with pytest.raises(ValueError, match="reproduction"):
        Simulation(
            spatial_dimension=2,
            reproduction=True,
        )