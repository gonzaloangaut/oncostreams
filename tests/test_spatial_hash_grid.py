"""Regression tests for spatial neighbor searches."""

import pickle
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


def test_neighbor_cache_tracks_moving_cells():
    """Cached buckets must reflect changes in cell positions."""

    grid = SpatialHashGrid(
        bounds=12.0,
        cube_size=2.0,
        torus=True,
    )

    # Add a cell to the grid and query for neighbors.
    query = np.array([0.5, 0.5, 0.5])
    near = np.array([2.5, 0.5, 0.5])
    far = np.array([6.5, 0.5, 0.5])

    grid.add_cell_to_hash_table(10, near)

    # First query populates the cache.
    assert list(grid.find_neighbors(query)) == [10]
    assert len(grid._neighbor_bucket_cache) == 1

    cached_keys = next(
        iter(grid._neighbor_bucket_cache.values())
    )

    # Repeated query reuses the same cached keys.
    assert list(grid.find_neighbors(query)) == [10]
    assert next(
        iter(grid._neighbor_bucket_cache.values())
    ) is cached_keys

    # Move the cell outside the queried neighborhood.
    grid.remove_cell_from_hash_table(10, near)
    grid.add_cell_to_hash_table(10, far)

    assert list(grid.find_neighbors(query)) == []

    # Add another cell inside the neighborhood.
    grid.add_cell_to_hash_table(11, near)

    assert list(grid.find_neighbors(query)) == [11]

    # No new cache entry should have been necessary.
    assert len(grid._neighbor_bucket_cache) == 1


def test_neighbor_cache_checkpoint_compatibility():
    """The cache must be reconstructed after deserialization."""

    grid = SpatialHashGrid(
        bounds=12.0,
        cube_size=2.0,
        torus=True,
    )

    position = np.array([0.5, 0.5, 0.5])

    grid.add_cell_to_hash_table(10, position)

    # Populate the cache.
    assert list(grid.find_neighbors(position)) == [10]
    assert len(grid._neighbor_bucket_cache) == 1

    # Serialize and restore the grid.
    restored = pickle.loads(pickle.dumps(grid))

    assert restored._neighbor_bucket_cache == {}
    assert list(restored.find_neighbors(position)) == [10]

    # Simulate restoring the state of an older checkpoint.
    old_state = grid.__dict__.copy()
    old_state.pop("_neighbor_bucket_cache")

    old_grid = SpatialHashGrid.__new__(SpatialHashGrid)
    old_grid.__setstate__(old_state)

    assert old_grid._neighbor_bucket_cache == {}
    assert list(old_grid.find_neighbors(position)) == [10]


@pytest.mark.parametrize("n", [3, 4, 8])
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_2d_grid_matches_3d_for_planar_cells(n, radius):
    """2D and 3D grids must find the same neighbors when z=0."""

    cube_size = 2.0
    bounds = n * cube_size

    grid_2d = SpatialHashGrid(
        bounds=bounds,
        cube_size=cube_size,
        torus=True,
        spatial_dimension=2,
    )

    grid_3d = SpatialHashGrid(
        bounds=bounds,
        cube_size=cube_size,
        torus=True,
        spatial_dimension=3,
    )

    # Place one cell in each XY bucket, always with z=0.
    for index, (i, j) in enumerate(
        product(range(n), repeat=2)
    ):
        position = np.array([
            cube_size * (i + 0.5),
            cube_size * (j + 0.5),
            0.0,
        ])

        grid_2d.add_cell_to_hash_table(index, position)
        grid_3d.add_cell_to_hash_table(index, position)

    positions = [
        np.array([0.01, 0.01, 0.0]),
        np.array([bounds - 1e-6, 0.01, 0.0]),
        np.array([bounds - 1e-6, bounds - 1e-6, 0.0]),
        np.array([1.2, 3.4, 0.0]),
    ]

    for position in positions:
        neighbors_2d = list(
            grid_2d.find_neighbors(position, bucket_radius=radius)
        )

        neighbors_3d = list(
            grid_3d.find_neighbors(position, bucket_radius=radius)
        )

        assert neighbors_2d == neighbors_3d


def test_spatial_grid_dimension():
    """Check that the grid uses the requested spatial dimension."""

    position = np.array([5.0, 7.0, 0.0])

    grid_2d = SpatialHashGrid(
        spatial_dimension=2
    )

    grid_3d = SpatialHashGrid(
        spatial_dimension=3
    )

    assert grid_2d.get_bucket_position(position).shape == (2,)
    assert grid_3d.get_bucket_position(position).shape == (3,)

    assert len(grid_2d.offsets) == 9
    assert len(grid_3d.offsets) == 27


def test_2d_grid_rejects_nonzero_z():
    """A planar grid must not silently ignore a nonzero z."""

    grid = SpatialHashGrid(
        spatial_dimension=2
    )

    position = np.array([1.0, 2.0, 0.5])

    with pytest.raises(ValueError, match="z=0"):
        grid.get_bucket_position(position)


def test_legacy_grid_defaults_to_3d():
    """Old checkpoints without dimension metadata must remain 3D."""

    grid = SpatialHashGrid(
        spatial_dimension=3
    )

    old_state = grid.__dict__.copy()
    old_state.pop("spatial_dimension")

    restored = SpatialHashGrid.__new__(SpatialHashGrid)
    restored.__setstate__(old_state)

    assert restored.spatial_dimension == 3
    assert len(restored.offsets) == 27