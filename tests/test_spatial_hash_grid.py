"""Regression tests for spatial neighbor searches."""

from itertools import chain, product

import numpy as np
import pytest

from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


@pytest.mark.parametrize("n", [3, 4, 8])
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_periodic_neighbors_match_original(n, radius):
    """Preserve the original neighbors and their iteration order."""

    cube_size = 2.0
    bounds = n * cube_size

    grid = SpatialHashGrid(
        bounds=bounds,
        cube_size=cube_size,
        torus=True,
    )

    # Place one cell at the center of every bucket.
    for index, coords in enumerate(product(range(n), repeat=3)):
        position = cube_size * (
            np.asarray(coords, dtype=float) + 0.5
        )
        grid.add_cell_to_hash_table(index, position)

    # Test positions are chosen to be near the corners of the domain, 
    # as well as a random position.
    positions = [
        np.array([0.01, 0.01, 0.01]),
        np.array([bounds - 1e-6, 0.01, 0.01]),
        np.array([bounds - 1e-6] * 3),
        np.array([1.2, 3.4, 5.6]),
    ]

    offsets = np.array(
        list(product(range(-radius, radius + 1), repeat=3))
    )

    # Test that the neighbors returned by the new implementation 
    # match the original implementation.
    for position in positions:
        bucket = grid.get_bucket_position(position)

        # Reference: original implementation using np.unique.
        adjacent = np.mod(
            bucket + offsets,
            grid.number_of_buckets,
        )
        adjacent = np.unique(adjacent, axis=0)

        expected = list(chain.from_iterable(
            grid.hash_table[b.tobytes()] for b in adjacent
        ))

        obtained = list(
            grid.find_neighbors(position, bucket_radius=radius)
        )

        assert obtained == expected